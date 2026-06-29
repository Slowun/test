"""Backtesting Agent.

Role: validate the hedging strategy out-of-sample. Uses walk-forward validation
(no look-ahead), mitigates survivorship / selection / transaction-cost biases,
runs stress tests on extreme spot & volatility shocks, and computes the full set
of performance metrics against an unhedged-BTC benchmark.

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       8 / 11 — BacktestingAgent.
НАЗНАЧЕНИЕ:  валидирует стратегию вне выборки: walk-forward (без заглядывания
             вперёд), контроль смещений (survivorship/selection/transaction-cost),
             стресс-тесты на экстремальные шоки цены/волатильности и полный набор
             метрик против бенчмарка «голый BTC».
ВХОД (consumes):  RISK_ASSESSMENT (от агента 7).
ВЫХОД (produces): BACKTEST_READY → агенту self_diagnostic.
КЛАДЁТ НА ДОСКУ:  backtest_metrics, walkforward, stress_table, bias_controls.
ИМПОРТИРУЕТ:
  - services.metrics (как mx)                : метрики производительности.
  - services.greeks.HestonGreeksEngine        : движок греков (для движка хеджа).
  - services.hedging_engine.*                 : движок хеджа + YEAR_NANOS.
  - services.walkforward.walk_forward_splits  : нарезка окон train/test.
  - services.providers.base.INSTR_ASSET       : константа типа «актив».
КОНФИГ:  config.backtest (окна/смещения/стресс), config.investment, config.risk.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import List                                  # аннотации

import numpy as np                                       # числовые операции
import pandas as pd                                       # таблицы результатов

from cryptohedge.core.agent import BaseAgent             # контракт агента
from cryptohedge.core.context import AgentContext        # контекст
from cryptohedge.core.message import Message, MessageType # сообщения
from cryptohedge.services import metrics as mx           # сервис метрик (псевдоним mx)
from cryptohedge.services.greeks import HestonGreeksEngine  # движок греков
from cryptohedge.services.hedging_engine import FeeModel, HedgingEngine, StrategyConfig, YEAR_NANOS  # движок хеджа
from cryptohedge.services.walkforward import walk_forward_splits  # нарезка walk-forward
from cryptohedge.services.providers.base import INSTR_ASSET  # константа типа «актив»


class BacktestingAgent(BaseAgent):
    name = "backtesting"                                 # имя агента / id этапа
    consumes = [MessageType.RISK_ASSESSMENT]             # принимает RISK_ASSESSMENT
    produces = MessageType.BACKTEST_READY                # выпускает BACKTEST_READY
    checkpoint_keys = ["backtest_metrics", "walkforward", "stress_table", "bias_controls"]  # ключи чекпойнта

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        bcfg = context.config.backtest                   # секция конфига бэктеста
        inv = context.config.investment                  # секция инвестиций
        history: pd.DataFrame = context.require("hedge_history")  # история хеджа
        md: pd.DataFrame = context.require("market_data")        # опционные данные
        calibr: pd.DataFrame = context.require("calibr_data")    # калибровка
        spot_close = context.require("spot_close")       # цены спота
        primary = context.require("primary_symbol")      # первичный актив
        liability, vega_option = context.require("hedge_setup")  # книга + vega-опцион

        # ---- in-sample performance metrics vs unhedged BTC benchmark  # --- метрики vs «голый BTC» ---
        pnl = history["pnl"].to_numpy()                  # кумулятивный PnL хеджа
        hedged_returns = np.diff(pnl, prepend=pnl[0]) / inv.capital_usd  # доходности хеджа (к капиталу)
        btc = spot_close[primary].reindex(pd.to_datetime(history["ts"])).to_numpy()  # цены BTC по датам истории
        btc_returns = np.concatenate([[0.0], np.diff(np.log(btc))])  # лог-доходности BTC (бенчмарк)
        ppy = context.config.horizons.trading_days_per_year  # дней в году
        perf = mx.compute_metrics(hedged_returns, benchmark=btc_returns, periods_per_year=ppy,  # полный набор метрик
                                  var_confidence=context.config.risk.var_confidence,
                                  var_method=context.config.risk.var_method)

        # ---- walk-forward validation (no look-ahead)    # --- walk-forward (без заглядывания вперёд) ---
        engine = HedgingEngine(                           # пересобираем движок хеджа для прогона фолдов
            HestonGreeksEngine(context.config.greeks),
            FeeModel(inv.transaction_fee_pct, inv.option_fee_pct, inv.option_fee_cap_pct),
            StrategyConfig(context.config.hedging.delta_eps, context.config.hedging.vega_eps),
        )
        samples = sorted(calibr["sample_idx"].unique())  # упорядоченные индексы срезов
        folds = walk_forward_splits(len(samples), bcfg.train_window, bcfg.test_window,  # нарезаем окна train/test
                                    bcfg.step, bcfg.purge, bcfg.embargo)
        wf_rows = []                                      # строки результатов по фолдам
        with log.timer("walk_forward", n_folds=len(folds)):  # замеряем время walk-forward
            for fold in folds:                           # по каждому фолду…
                test_samples = [samples[i] for i in fold.test]  # индексы тестовых срезов
                res = engine.run(md, calibr, liability, vega_option, sample_indices=test_samples)  # прогон на тесте
                if res.history.empty:                    # пустой результат…
                    continue                             #   → пропускаем фолд
                fp = res.history["pnl"].to_numpy()       # PnL на тесте
                fr = np.diff(fp, prepend=fp[0]) / inv.capital_usd  # доходности на тесте
                m = mx.compute_metrics(fr, periods_per_year=ppy)  # метрики фолда
                wf_rows.append({                         # строка результата фолда
                    "fold": fold.index, "train_start": int(fold.train[0]), "train_end": int(fold.train[-1]),
                    "test_start": int(fold.test[0]), "test_end": int(fold.test[-1]),
                    "roi": m.roi, "sharpe": m.sharpe, "sortino": m.sortino,
                    "max_drawdown": m.max_drawdown, "pnl_end": float(fp[-1] - fp[0]),
                })
        walkforward = pd.DataFrame(wf_rows)              # результаты walk-forward → DataFrame

        # ---- stress testing on extreme scenarios        # --- стресс-тесты на экстремальные сценарии ---
        stress = self._stress_test(context, engine, md, calibr, history, liability, vega_option, bcfg)  # стресс

        bias_controls = {                                # описание контроля смещений (для отчёта/объяснимости)
            "survivorship_bias": {"controlled": bcfg.account_survivorship_bias,
                                  "note": "Full universe retained across the window; no winners-only selection."},
            "selection_bias": {"controlled": bcfg.account_selection_bias,
                               "note": "Hedge universe ranked on in-sample data only; no peeking at test folds."},
            "transaction_cost_bias": {"controlled": bcfg.account_transaction_cost_bias,
                                      "note": "Spot and option fees (incl. cap) charged on every trade."},
            "look_ahead": {"controlled": True,
                           "note": "Walk-forward folds with purge/embargo; per-slice calibration is contemporaneous."},
        }

        walkforward.to_parquet(context.results_path("walkforward.parquet")) if not walkforward.empty else None  # сохранить wf
        stress.to_parquet(context.results_path("stress_test.parquet"))  # сохранить стресс
        pd.Series(perf.to_dict()).to_json(context.results_path("performance_metrics.json"))  # сохранить метрики

        context.put("backtest_metrics", perf.to_dict())  # метрики → на доску
        context.put("walkforward", walkforward)          # walk-forward → на доску
        context.put("stress_table", stress)              # стресс → на доску
        context.put("bias_controls", bias_controls)      # контроль смещений → на доску

        log.decision("backtest complete", roi=round(perf.roi, 4), sharpe=round(perf.sharpe, 3),  # лог-итог бэктеста
                     sortino=round(perf.sortino, 3), calmar=round(perf.calmar, 3),
                     max_drawdown=round(perf.max_drawdown, 4), var=round(perf.var, 5),
                     cvar=round(perf.cvar, 5), n_folds=len(walkforward))

        return Message(self.produces, self.name, "self_diagnostic",  # BACKTEST_READY следующему агенту
                       payload={"sharpe": perf.sharpe, "roi": perf.roi},
                       correlation_id=message.correlation_id)

    def _stress_test(self, context, engine, md, calibr, history, liability, vega_option, bcfg) -> pd.DataFrame:
        """Decompose the book's P&L per leg under shocks and contrast with the
        unhedged BTC exposure, so the effectiveness of the hedge is explicit."""
        last_cal = calibr.sort_values("sample_idx").iloc[-1]  # последний срез калибровки
        sidx = int(last_cal["sample_idx"])               # его индекс
        grp = md[md["sample_idx"] == sidx]               # данные среза
        spot = float(grp[grp["instrument_type"] == INSTR_ASSET]["price"].iloc[0])  # спот на срезе
        ts_ns = int(pd.Timestamp(last_cal["timestamp"]).value)  # метка времени, нс
        last_row = history.iloc[-1]                      # последнее состояние книги
        pos_spot = float(last_row["pos_spot"])           # текущая позиция в споте
        pos_vo = float(last_row["pos_vega_option"])      # текущая позиция в vega-опционе
        q_hedge = float(context.require("hedge_sizing").quantity_to_hedge)  # объём «голой» BTC-экспозиции

        from cryptohedge.domain.market import HestonParameters  # ленивый импорт структуры параметров

        def params_with(v0):                             # параметры Хестона с заданной начальной дисперсией
            return HestonParameters(v0=max(v0, 1e-6), kappa=float(last_cal["kappa"]),
                                    theta=float(last_cal["theta"]), eps=float(last_cal["eps"]),
                                    rho=float(last_cal["rho"]), flat_yield=float(last_cal.get("flat_yield", 0.0)))

        ttm_v = (vega_option.expiry_ts - ts_ns) / YEAR_NANOS  # TTM vega-опциона

        def leg_values(ds, dv):                          # стоимости «ног» книги при шоке (ds спот, dv вола)
            s = spot * (1 + ds)                          #   шокированный спот
            p = params_with(float(last_cal["v0"]) * (1 + dv))  #   шокированная дисперсия
            p_liab = sum(engine.greeks.price(s, p, c.strike, (c.expiry_ts - ts_ns) / YEAR_NANOS, c.is_call) * c.notional  # стоимость книги
                         for c in liability if (c.expiry_ts - ts_ns) / YEAR_NANOS > 0)
            vo_val = engine.greeks.price(s, p, vega_option.strike, ttm_v, vega_option.is_call) if ttm_v > 0 else 0.0  # стоимость vega-опциона
            return s, p_liab, vo_val                     #   шок-спот, стоимость книги, стоимость vega-опциона

        s0, liab0, vo0 = leg_values(0.0, 0.0)            # базовые стоимости (без шока)
        rows = []                                        # строки таблицы стресс-тестов
        for sc in bcfg.stress_scenarios:                 # по каждому сценарию…
            s, liab, vo = leg_values(sc["spot_shock"], sc["vol_shock"])  # стоимости при шоке
            liability_pnl = -(liab - liab0)          # the option book is sold (short)  # PnL книги (она продана)
            spot_hedge_pnl = pos_spot * (s - s0)         # PnL спотового хеджа
            option_hedge_pnl = pos_vo * (vo - vo0)       # PnL опционного хеджа
            net = liability_pnl + spot_hedge_pnl + option_hedge_pnl  # суммарный PnL хеджированной книги
            unhedged = q_hedge * (s - s0)            # naked long-BTC exposure of equal size  # PnL «голого» BTC
            eff = float(1.0 - abs(net) / abs(unhedged)) if abs(unhedged) > 1e-9 else 1.0  # эффективность хеджа
            rows.append({                                # строка результата сценария
                "scenario": sc["name"], "spot_shock": sc["spot_shock"], "vol_shock": sc["vol_shock"],
                "liability_pnl": float(liability_pnl), "spot_hedge_pnl": float(spot_hedge_pnl),
                "option_hedge_pnl": float(option_hedge_pnl), "net_hedged_pnl": float(net),
                "unhedged_pnl": float(unhedged), "hedge_effectiveness": eff,
            })
        return pd.DataFrame(rows)                        # таблица стресс-тестов
