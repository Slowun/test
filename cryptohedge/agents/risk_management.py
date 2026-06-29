"""Risk Management Agent.

Role: enforce the risk budget. Computes VaR, CVaR and Expected Shortfall of the
hedged book, checks them against configured limits (VaR, drawdown, leverage),
and produces adaptive stop-loss levels (ATR + VaR + Heston-vol aware) together
with a dynamic trailing-stop trajectory.

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       7 / 11 — RiskManagementAgent.
НАЗНАЧЕНИЕ:  следит за бюджетом риска. Считает VaR/CVaR/ES хеджированной книги,
             проверяет лимиты (VaR, просадка, плечо) и строит адаптивный стоп-лосс
             (ATR + VaR + волатильность Хестона) и траекторию трейлинг-стопа.
ВХОД (consumes):  PORTFOLIO_READY (от агента 6).
ВЫХОД (produces): RISK_ASSESSMENT → агенту backtesting.
КЛАДЁТ НА ДОСКУ:  risk_assessment, stop_level, trailing_stops, risk_returns.
ИМПОРТИРУЕТ:
  - domain.decisions.RiskAssessment       : структура оценки риска.
  - services.metrics (как mx)             : VaR/CVaR/просадка.
  - services.stops.*                      : ATR, адаптивный стоп, трейлинг-стоп.
КОНФИГ:  config.risk (методы/лимиты/стопы), config.investment.capital_usd.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import numpy as np                                       # числовые операции (diff, sqrt)
import pandas as pd                                       # таблицы истории/баров

from cryptohedge.core.agent import BaseAgent             # контракт агента
from cryptohedge.core.context import AgentContext        # контекст
from cryptohedge.core.message import Message, MessageType # сообщения
from cryptohedge.domain.decisions import RiskAssessment  # структура оценки риска
from cryptohedge.services import metrics as mx           # сервис метрик (псевдоним mx)
from cryptohedge.services.stops import TrailingStop, adaptive_stop, average_true_range  # стопы и ATR


class RiskManagementAgent(BaseAgent):
    name = "risk_management"                             # имя агента / id этапа
    consumes = [MessageType.PORTFOLIO_READY]             # принимает PORTFOLIO_READY
    produces = MessageType.RISK_ASSESSMENT               # выпускает RISK_ASSESSMENT
    checkpoint_keys = ["risk_assessment", "stop_level", "trailing_stops", "risk_returns"]  # ключи чекпойнта

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        rcfg = context.config.risk                       # секция конфига риска
        inv = context.config.investment                  # секция инвестиций (капитал)
        history: pd.DataFrame = context.require("hedge_history")  # история хеджа (PnL по шагам)
        spot_bars: pd.DataFrame = context.require("spot_bars")   # OHLCV-бары (для ATR)
        primary = context.require("primary_symbol")      # первичный актив
        calibr: pd.DataFrame = context.require("calibr_data")    # калибровка (для волатильности Хестона)

        # ---- hedged PnL returns (per capital)           # --- доходности хеджированного PnL (к капиталу) ---
        pnl = history["pnl"].to_numpy()                  # кумулятивный PnL по шагам
        pnl_changes = np.diff(pnl, prepend=pnl[0]) / inv.capital_usd  # приростные доходности (к капиталу)
        var = mx.value_at_risk(pnl_changes, rcfg.var_confidence, rcfg.var_method)  # VaR
        cvar = mx.conditional_var(pnl_changes, rcfg.cvar_confidence)  # CVaR
        equity = inv.capital_usd + (pnl - pnl[0])        # кривая капитала
        mdd = mx.max_drawdown(equity)                    # максимальная просадка

        breached = []                                    # список нарушенных лимитов
        if var > rcfg.var_limit_pct:                     # превышен лимит VaR?
            breached.append("VaR")                       #   → отмечаем
        if abs(mdd) > rcfg.max_drawdown_limit_pct:       # превышен лимит просадки?
            breached.append("MaxDrawdown")               #   → отмечаем

        utilization = {                                  # утилизация лимитов (доля от лимита)
            "var_vs_limit": float(var / rcfg.var_limit_pct) if rcfg.var_limit_pct else 0.0,  # VaR/лимит
            "drawdown_vs_limit": float(abs(mdd) / rcfg.max_drawdown_limit_pct) if rcfg.max_drawdown_limit_pct else 0.0,  # просадка/лимит
        }
        assessment = RiskAssessment(                     # формируем объект оценки риска
            var=var, cvar=cvar, expected_shortfall=cvar, max_drawdown=mdd,
            within_limits=(len(breached) == 0), breached_limits=breached, utilization=utilization,
        )

        # ---- adaptive stop-loss on the primary BTC exposure  # --- адаптивный стоп по BTC-экспозиции ---
        bars = spot_bars[spot_bars["symbol"] == primary].sort_values("timestamp")  # бары первичного актива
        atr = average_true_range(bars["high"].to_numpy(), bars["low"].to_numpy(),  # ATR по барам
                                 bars["close"].to_numpy(), rcfg.stop_loss.atr_window)
        atr_last = float(np.nan_to_num(atr[-1]))         # последний ATR (без NaN)
        ref_price = float(bars["close"].iloc[-1])        # опорная цена (последний close)
        btc_returns = np.diff(np.log(bars["close"].to_numpy()))  # лог-доходности BTC
        daily_var = mx.value_at_risk(btc_returns, rcfg.var_confidence, "historical")  # дневной VaR BTC
        v0_last = float(calibr.sort_values("sample_idx")["v0"].iloc[-1])  # последняя дисперсия v0 (Хестон)
        heston_daily_vol = float(np.sqrt(max(v0_last, 0.0) / context.config.horizons.trading_days_per_year))  # дн. вола Хестона

        stop = adaptive_stop(ref_price, atr_last, daily_var, heston_daily_vol, "long", rcfg.stop_loss)  # адаптивный стоп

        # ---- dynamic trailing stop trajectory           # --- траектория трейлинг-стопа ---
        trailing_rows = []                               # строки траектории трейлинга
        if rcfg.stop_loss.trailing:                      # если трейлинг включён…
            tstop = TrailingStop("long", float(bars["close"].iloc[0]), rcfg.stop_loss)  # инициализируем трейлинг
            closes = bars["close"].to_numpy()            # цены закрытия
            for i, px in enumerate(closes):              # по каждой цене…
                a = float(np.nan_to_num(atr[i])) if i < len(atr) else atr_last  # ATR на шаге
                lvl = tstop.update(float(px), a, daily_var, heston_daily_vol)   # обновляем уровень стопа
                trailing_rows.append({"ts": bars["timestamp"].iloc[i], "price": float(px),  # копим строку
                                      "stop_price": lvl.stop_price, "triggered": bool(tstop.triggered)})
        trailing = pd.DataFrame(trailing_rows)           # траектория трейлинга → DataFrame

        context.put("risk_assessment", assessment.to_dict())  # оценка риска → на доску
        context.put("stop_level", stop.to_dict())        # уровень стопа → на доску
        context.put("trailing_stops", trailing)          # траектория трейлинга → на доску
        context.put("risk_returns", pnl_changes)         # доходности для диагностики → на доску

        if not trailing.empty:                           # если траектория не пуста…
            trailing.to_parquet(context.results_path("trailing_stops.parquet"))  # сохраняем на диск

        log.decision("risk assessment", var=round(var, 5), cvar=round(cvar, 5),  # лог-итог оценки риска
                     max_drawdown=round(mdd, 4), within_limits=assessment.within_limits,
                     breached=breached, stop_price=round(stop.stop_price, 2),
                     stop_distance_pct=round(stop.distance_pct, 4))

        return Message(self.produces, self.name, "backtesting",  # RISK_ASSESSMENT следующему агенту
                       payload={"within_limits": assessment.within_limits, "var": var},
                       correlation_id=message.correlation_id)
