"""Hedging Decision Agent.

Role: form and execute the delta-vega hedging strategy with the Heston model.
Running the hedging engine over the in-sample history produces every required
output - spot, PnL, fees, delta-hedge, vega-hedge, spot/option positions, option
portfolio premium, portfolio delta and vega - plus a current actionable decision.

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       5 / 11 — HedgingDecisionAgent (СЕРДЦЕ хеджирования).
НАЗНАЧЕНИЕ:  формирует и «исполняет» Δ/ν-хедж по модели Хестона. Прогоняет
             движок хеджа (services.hedging_engine) по всей истории: на каждом
             шаге ребалансирует спот (Δ) и опцион (ν), считает PnL, комиссии,
             позиции, остаточные греки и формирует актуальное решение.
ВХОД (consumes):  GREEKS_READY (от агента 4).
ВЫХОД (produces): HEDGE_DECISION → агенту portfolio_optimization.
КЛАДЁТ НА ДОСКУ:  hedge_history, hedge_decisions, hedge_trades, latest_decision.
ИМПОРТИРУЕТ:
  - services.greeks.HestonGreeksEngine                 : движок греков.
  - services.hedging_engine.{HedgingEngine,FeeModel,StrategyConfig} : движок хеджа.
КОНФИГ:  config.investment (комиссии), config.greeks, config.hedging (допуски ε).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import pandas as pd                                       # таблицы истории хеджа

from cryptohedge.core.agent import BaseAgent             # контракт агента
from cryptohedge.core.context import AgentContext        # контекст
from cryptohedge.core.message import Message, MessageType # сообщения
from cryptohedge.services.greeks import HestonGreeksEngine  # движок греков (нужен движку хеджа)
from cryptohedge.services.hedging_engine import FeeModel, HedgingEngine, StrategyConfig  # движок хеджа и его части


class HedgingDecisionAgent(BaseAgent):
    name = "hedging_decision"                            # имя агента / id этапа
    consumes = [MessageType.GREEKS_READY]                # принимает GREEKS_READY
    produces = MessageType.HEDGE_DECISION                # выпускает HEDGE_DECISION
    checkpoint_keys = ["hedge_history", "hedge_decisions", "hedge_trades", "latest_decision"]  # ключи чекпойнта

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        inv = context.config.investment                  # секция инвестиций (комиссии)
        engine = HedgingEngine(                           # собираем движок хеджа из трёх частей:
            HestonGreeksEngine(context.config.greeks),   #   движок греков
            FeeModel(inv.transaction_fee_pct, inv.option_fee_pct, inv.option_fee_cap_pct),  # модель комиссий
            StrategyConfig(context.config.hedging.delta_eps, context.config.hedging.vega_eps),  # допуски ε
        )

        md: pd.DataFrame = context.require("market_data")  # опционные данные
        calibr: pd.DataFrame = context.require("calibr_data")  # параметры калибровки по срезам
        liability, vega_option = context.require("hedge_setup")  # книга обязательств + vega-опцион (от агента 4)

        with log.timer("hedging_run", n=len(calibr)):    # замеряем время прогона движка
            result = engine.run(md, calibr, liability, vega_option)  # ПРОГОН хеджа по всей истории

        history = result.history                         # таблица истории хеджа (по шагам)
        if history.empty:                                # пустая история — ошибка прогона…
            raise RuntimeError("Hedging engine produced no results")  #   → исключение

        history.to_parquet(context.results_path("hedging_history.parquet"))  # сохраняем историю на диск
        decisions_df = pd.DataFrame([d.to_dict() for d in result.decisions])  # решения хеджа → DataFrame
        if not decisions_df.empty:                       # если решения есть…
            decisions_df.to_parquet(context.results_path("hedge_decisions.parquet"))  # сохраняем на диск

        last = history.iloc[-1]                           # последняя строка истории (текущее состояние)
        latest = {                                       # сводка текущего решения хеджа
            "ts": str(last["ts"]), "spot": float(last["spot"]),  #   время и спот
            "portfolio_delta": float(last["delta"]), "portfolio_vega": float(last["vega"]),  # греки портфеля
            "delta_hedge": float(last["delta_hedge"]), "vega_hedge": float(last["vega_hedge"]),  # объёмы хеджа
            "residual_delta": float(last["delta"] - last["delta_hedge"]),  # остаточная дельта (после хеджа)
            "residual_vega": float(last["vega"] - last["vega_hedge"]),     # остаточная вега (после хеджа)
            "pos_spot": float(last["pos_spot"]), "pos_vega_option": float(last["pos_vega_option"]),  # позиции
            "pnl": float(last["pnl"]), "fees": float(last["fee"]),  # PnL и комиссии
            "n_trades": int(len(result.trades)),         #   число сделок
        }

        context.put("hedge_history", history)            # история хеджа → на доску
        context.put("hedge_decisions", [d.to_dict() for d in result.decisions])  # решения → на доску
        context.put("hedge_trades", result.trades)       # сделки → на доску
        context.put("latest_decision", latest)           # текущее решение → на доску

        log.decision("executed delta-vega hedge", n_trades=latest["n_trades"],  # лог-итог хеджа
                     residual_delta=round(latest["residual_delta"], 4),
                     residual_vega=round(latest["residual_vega"], 4),
                     pnl=round(latest["pnl"], 2), fees=round(latest["fees"], 2))

        return Message(self.produces, self.name, "portfolio_optimization",  # HEDGE_DECISION следующему агенту
                       payload={"pnl": latest["pnl"], "n_trades": latest["n_trades"]},
                       correlation_id=message.correlation_id)
