"""Portfolio Optimization Agent.

Role: construct, *validate* and rebalance an investable portfolio.

The agent (1) selects a diversified, historically profitable investable universe,
(2) builds candidate portfolios with five optimisers (Mean-Variance, Risk Parity,
Minimum Variance, Maximum Diversification, CVaR), (3) runs a periodically
rebalanced backtest of each candidate accounting for transaction costs, and
(4) selects the method that is both profitable and well diversified. It publishes
the chosen portfolio's constituents, equity curve, rebalancing path and
diversification diagnostics so the dashboard can render them.

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       6 / 11 — PortfolioOptimizationAgent.
НАЗНАЧЕНИЕ:  строит инвест-портфель: (1) отбирает диверсифицированную прибыльную
             вселенную; (2) считает веса 5 методами; (3) бэктестит каждый метод с
             ребалансировкой и комиссиями; (4) АВТОМАТИЧЕСКИ выбирает лучший метод
             (прибыльность + диверсификация). Публикует состав, equity-кривую,
             путь весов и диагностику диверсификации для дашборда.
ВХОД (consumes):  HEDGE_DECISION (от агента 5).
ВЫХОД (produces): PORTFOLIO_READY → агенту risk_management.
КЛАДЁТ НА ДОСКУ:  optimization_results, rebalance_decision, opt_weights,
                  portfolio_universe/constituents/equity/weights_path/rebalances/costs,
                  diversification, method_comparison.
ИМПОРТИРУЕТ:
  - domain.decisions.RebalanceDecision   : структура решения о ребалансе.
  - services.optimization (как opt)       : 5 оптимизаторов + turnover/издержки.
  - services.portfolio_backtest (как pbt) : бэктест с ребалансировкой.
КОНФИГ:  config.optimization (методы/веса/ребаланс/авто-выбор), config.investment.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import Dict, List                            # аннотации

import numpy as np                                       # линейная алгебра/веса/метрики
import pandas as pd                                       # таблицы портфеля

from cryptohedge.core.agent import BaseAgent             # контракт агента
from cryptohedge.core.context import AgentContext        # контекст
from cryptohedge.core.message import Message, MessageType # сообщения
from cryptohedge.domain.decisions import RebalanceDecision  # структура решения о ребалансе
from cryptohedge.services import optimization as opt     # сервис оптимизации (псевдоним opt)
from cryptohedge.services import portfolio_backtest as pbt  # сервис бэктеста (псевдоним pbt)


class PortfolioOptimizationAgent(BaseAgent):
    name = "portfolio_optimization"                      # имя агента / id этапа
    consumes = [MessageType.HEDGE_DECISION]              # принимает HEDGE_DECISION
    produces = MessageType.PORTFOLIO_READY               # выпускает PORTFOLIO_READY
    checkpoint_keys = [                                  # ключи чекпойнта
        "optimization_results", "rebalance_decision", "opt_weights",
        "portfolio_universe", "portfolio_constituents", "portfolio_equity",
        "portfolio_weights_path", "portfolio_rebalances", "portfolio_costs",
        "diversification", "method_comparison",
    ]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        ocfg = context.config.optimization               # секция конфига оптимизации
        ppy = context.config.horizons.trading_days_per_year  # дней в году (аннуализация)
        fee = context.config.investment.transaction_fee_pct  # комиссия за сделку

        returns: pd.DataFrame = context.require("returns")  # матрица доходностей (обязательна)
        spot_close: pd.DataFrame = context.require("spot_close")  # цены (обязательны)
        primary: str = context.require("primary_symbol")  # первичный актив

        # ---- 1. select a diversified, historically profitable investable universe  # --- 1. выбор вселенной ---
        universe = self._select_universe(returns, ocfg, ppy)  # отбираем инструменты в портфель
        prices = spot_close[universe].dropna(axis=1, how="any")  # цены вселенной без пропусков
        universe = list(prices.columns)                  # финальный список (после отсева)
        n = len(universe)                                # число активов
        R = returns[universe].dropna()                   # доходности вселенной
        mu = R.mean().to_numpy() * ppy                   # годовые ожидаемые доходности
        Sigma = R.cov().to_numpy() * ppy                 # годовая ковариационная матрица
        scenarios = R.to_numpy()                         # сценарии доходностей (для CVaR)
        log.decision("selected investable universe", n=n, instruments=universe)  # лог выбора вселенной

        # ---- 2-3. optimise + rebalanced backtest for every method  # --- 2-3. оптимизация + бэктест по методам ---
        results: Dict[str, dict] = {}                    # сводные результаты по методам
        backtests: Dict[str, pbt.PortfolioBacktest] = {}  # бэктесты по методам
        for method in ocfg.methods:                      # по каждому методу оптимизации…
            w_full = self._optimise(method, mu, Sigma, scenarios, ocfg)  # веса на полной выборке
            fn = self._weight_fn(method, ocfg, ppy)      # функция весов для бэктеста (по train-окну)
            with log.timer(f"backtest_{method}"):        # замеряем время бэктеста метода
                bt = pbt.backtest_rebalanced(prices, fn, ocfg.rebalance_frequency_days,  # бэктест с ребалансом
                                             fee, ocfg.lookback_days, ppy)
            backtests[method] = bt                       # сохраняем бэктест
            results[method] = {                          # сводка по методу:
                "weights": {c: float(wi) for c, wi in zip(universe, w_full)},  #   веса по активам
                "expected_return": float(w_full @ mu),   #   ожидаемая доходность
                "expected_risk": float(np.sqrt(max(w_full @ Sigma @ w_full, 0.0))),  #   ожидаемый риск
                "sharpe": float((w_full @ mu) / np.sqrt(max(w_full @ Sigma @ w_full, 1e-12))),  #   Sharpe
                "backtest": bt.metrics,                  #   метрики бэктеста
            }

        # ---- 4. choose the method: profitable first, then most diversified / risk-adjusted  # --- 4. выбор метода ---
        chosen = self._select_method(results, ocfg)      # авто-выбор лучшего метода
        bt = backtests[chosen]                           # бэктест выбранного метода
        w_chosen = np.array([results[chosen]["weights"][c] for c in universe])  # веса выбранного метода
        log.decision("selected optimisation method", method=chosen,  # лог выбора метода с метриками
                     total_return=round(results[chosen]["backtest"]["total_return"], 4),
                     sharpe=round(results[chosen]["backtest"]["sharpe"], 3),
                     diversification_ratio=round(results[chosen]["backtest"]["diversification_ratio"], 3),
                     effective_n=round(results[chosen]["backtest"]["effective_n"], 2))

        # ---- equal-weight benchmark for a fair profitability comparison  # --- бенчмарк равных весов ---
        eq_bt = pbt.backtest_rebalanced(prices, lambda tr: np.ones(n) / n,  # портфель 1/n как бенчмарк
                                        ocfg.rebalance_frequency_days, fee, ocfg.lookback_days, ppy)

        # ---- assemble portfolio artefacts ------------------------------------  # --- сборка артефактов портфеля ---
        corr = returns[universe].corrwith(returns[primary]) if primary in returns.columns else pd.Series(dtype=float)  # корреляция с BTC
        constituents = self._constituents(universe, w_chosen, mu, Sigma, corr, primary)  # таблица состава портфеля
        equity = pd.DataFrame({                          # equity-кривая портфеля vs бенчмарк
            "ts": bt.equity.index,                       #   время
            "equity": bt.equity.to_numpy(),              #   капитал портфеля
            "benchmark": eq_bt.equity.reindex(bt.equity.index).to_numpy(),  #   капитал бенчмарка 1/n
        })
        equity["drawdown"] = bt.equity.to_numpy() / np.maximum.accumulate(bt.equity.to_numpy()) - 1.0  # просадка

        weights_path = bt.weights_path.copy()            # путь весов во времени
        weights_path.insert(0, "ts", weights_path.index)  # добавляем колонку времени
        weights_path = weights_path.reset_index(drop=True)  # сбрасываем индекс

        costs = pd.DataFrame({"ts": bt.cum_cost.index, "cum_cost": bt.cum_cost.to_numpy(),  # издержки и оборот
                              "turnover": bt.turnover.to_numpy()})
        rebalances = [str(d) for d in bt.rebalance_dates]  # даты ребалансировок

        diversification = {k: bt.metrics[k] for k in     # выборка метрик диверсификации
                           ["diversification_ratio", "avg_diversification_ratio", "effective_n",
                            "n_active", "max_weight", "hhi", "n_rebalances"]}
        diversification["n_assets"] = n                  # число активов
        diversification["benchmark_diversification_ratio"] = eq_bt.metrics.get("diversification_ratio", 1.0)  # бенчмарк DR

        method_comparison = pd.DataFrame([               # таблица сравнения методов
            {"method": m,
             "total_return": r["backtest"]["total_return"],
             "cagr": r["backtest"]["cagr"],
             "sharpe": r["backtest"]["sharpe"],
             "volatility": r["backtest"]["volatility"],
             "max_drawdown": r["backtest"]["max_drawdown"],
             "diversification_ratio": r["backtest"]["diversification_ratio"],
             "effective_n": r["backtest"]["effective_n"],
             "chosen": (m == chosen)}                    # пометка выбранного метода
            for m, r in results.items()
        ])

        # ---- rebalance decision (kept for the explainability/legacy contract)  # --- объект решения о ребалансе ---
        w_prev = np.ones(n) / n                          # «предыдущие» веса = равные (отправная точка)
        tn = opt.turnover(w_chosen, w_prev)              # оборот перехода к выбранным весам
        tcost = opt.transaction_cost(w_chosen, w_prev, fee, context.config.investment.capital_usd)  # издержки перехода
        decision = RebalanceDecision(                    # формируем объект-решение
            method=chosen,                               #   выбранный метод
            target_weights=results[chosen]["weights"],   #   целевые веса
            current_weights={c: float(wp) for c, wp in zip(universe, w_prev)},  #   текущие (равные) веса
            turnover=tn,                                 #   оборот
            expected_return=results[chosen]["expected_return"],  #   ожидаемая доходность
            expected_risk=results[chosen]["expected_risk"],      #   ожидаемый риск
            transaction_cost=tcost,                      #   издержки
            triggered=bool(tn > 0.5 * ocfg.max_turnover),  #   сработал ли порог ребаланса
            rationale=(f"Method '{chosen}' delivered total return "  # текстовое обоснование
                       f"{results[chosen]['backtest']['total_return']:.2%} with diversification ratio "
                       f"{results[chosen]['backtest']['diversification_ratio']:.2f} "
                       f"(effective {results[chosen]['backtest']['effective_n']:.1f} bets)."),
        )

        # ---- persist                                    # --- сохраняем артефакты на диск ---
        method_comparison.to_parquet(context.results_path("portfolio_methods.parquet"))  # сравнение методов
        constituents.to_parquet(context.results_path("portfolio_constituents.parquet"))  # состав портфеля
        equity.to_parquet(context.results_path("portfolio_equity.parquet"))              # equity-кривая
        weights_path.to_parquet(context.results_path("portfolio_weights_path.parquet"))  # путь весов

        context.put("optimization_results", results)     # все результаты → на доску
        context.put("rebalance_decision", decision.to_dict())  # решение о ребалансе → на доску
        context.put("opt_weights", results[chosen]["weights"])  # выбранные веса → на доску
        context.put("portfolio_universe", universe)      # вселенная → на доску
        context.put("portfolio_constituents", constituents)  # состав → на доску
        context.put("portfolio_equity", equity)          # equity → на доску
        context.put("portfolio_weights_path", weights_path)  # путь весов → на доску
        context.put("portfolio_rebalances", rebalances)  # даты ребалансов → на доску
        context.put("portfolio_costs", costs)            # издержки → на доску
        context.put("diversification", diversification)  # диагностика диверсификации → на доску
        context.put("method_comparison", method_comparison)  # сравнение методов → на доску

        log.decision("portfolio optimization", chosen=chosen, n_assets=n,  # лог-итог оптимизации
                     turnover=round(tn, 3), transaction_cost=round(tcost, 2),
                     profitable=bool(results[chosen]["backtest"]["total_return"] > 0),
                     diversification_ratio=round(diversification["diversification_ratio"], 3))

        return Message(self.produces, self.name, "risk_management",  # PORTFOLIO_READY следующему агенту
                       payload={"method": chosen, "n_assets": n,
                                "total_return": results[chosen]["backtest"]["total_return"]},
                       correlation_id=message.correlation_id)

    # ------------------------------------------------------------------ helpers
    def _select_universe(self, returns: pd.DataFrame, ocfg, ppy: int) -> List[str]:
        """Pick a diversified, historically profitable set of instruments.

        Profitable longs (positive mean return) ranked by risk-adjusted return;
        if too few are positive, fall back to the top names by mean return.
        """
        mu = returns.mean() * ppy                        # годовая средняя доходность по активам
        vol = returns.std() * np.sqrt(ppy)               # годовая волатильность по активам
        sharpe = (mu / vol.replace(0, np.nan)).fillna(0.0)  # Sharpe (защита от деления на 0)
        k = max(2, min(ocfg.portfolio_universe_size, returns.shape[1]))  # сколько активов взять

        profitable = sharpe[mu > ocfg.min_expected_return].sort_values(ascending=False)  # прибыльные по Sharpe
        if len(profitable) >= max(2, k // 2):            # если прибыльных достаточно…
            return list(profitable.head(k).index)        #   берём топ-k прибыльных
        return list(mu.sort_values(ascending=False).head(k).index)  # иначе — топ-k по средней доходности

    def _optimise(self, method, mu, Sigma, scenarios, ocfg) -> np.ndarray:
        n = len(mu)                                      # число активов
        # keep the bounds feasible for small universes (n * max_weight must be >= 1)
        max_weight = max(ocfg.max_weight, 1.0 / n + 1e-9)  # корректируем кап веса для малых вселенных
        try:
            return opt.optimize(method, mu, Sigma, scenarios=scenarios,  # вызываем оптимизатор метода
                                risk_aversion=ocfg.risk_aversion, cvar_alpha=ocfg.cvar_alpha,
                                long_only=ocfg.long_only, max_weight=max_weight)
        except Exception:                                # при ошибке оптимизации…
            return np.ones(n) / n                        #   откат к равным весам

    def _weight_fn(self, method, ocfg, ppy):
        def fn(train_returns: pd.DataFrame) -> np.ndarray:  # функция весов для бэктеста (по train-окну)
            mu = train_returns.mean().to_numpy() * ppy   #   годовые доходности на train-окне
            Sigma = train_returns.cov().to_numpy() * ppy #   годовая ковариация на train-окне
            return self._optimise(method, mu, Sigma, train_returns.to_numpy(), ocfg)  # веса на train-окне
        return fn                                        # возвращаем замыкание

    def _select_method(self, results: Dict[str, dict], ocfg) -> str:
        """Pick the best *profitable* method, balancing risk-adjusted return and
        diversification on a common (min-max normalised) scale so the two very
        different magnitudes (Sharpe ~ units, diversification ratio ~ 1-2) are
        weighted fairly. ``diversification_weight`` controls the trade-off."""
        if not ocfg.auto_select_method:                  # если авто-выбор отключён…
            return ocfg.method if ocfg.method in results else next(iter(results))  # …берём метод из конфига

        profitable = {m: r for m, r in results.items()   # отбираем прибыльные методы
                      if r["backtest"]["total_return"] > ocfg.min_expected_return}
        pool = profitable or results                     # если прибыльных нет — берём все
        if len(pool) == 1:                               # единственный кандидат…
            return next(iter(pool))                      #   → он и выбран

        def col(metric):                                 # выбрать метрику по всем методам пула
            return {m: float(r["backtest"][metric]) for m, r in pool.items()}

        def norm(vals: Dict[str, float]) -> Dict[str, float]:  # min-max нормировка к [0,1]
            lo, hi = min(vals.values()), max(vals.values())
            rng = hi - lo
            return {m: (v - lo) / rng if rng > 1e-12 else 0.5 for m, v in vals.items()}

        n_sharpe = norm(col("sharpe"))                   # нормированный Sharpe
        n_dr = norm(col("diversification_ratio"))        # нормированный diversification ratio
        n_eff = norm(col("effective_n"))                 # нормированное «эффективное число ставок»
        w = ocfg.diversification_weight                  # вес диверсификации в итоговом score
        div = {m: 0.5 * (n_dr[m] + n_eff[m]) for m in pool}  # совокупная мера диверсификации
        score = {m: (1.0 - w) * n_sharpe[m] + w * div[m] for m in pool}  # итоговый score
        return max(score, key=score.get)                 # метод с максимальным score

    def _constituents(self, universe, weights, mu, Sigma, corr, primary) -> pd.DataFrame:
        vol = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))  # годовые волатильности активов
        df = pd.DataFrame({                              # таблица состава портфеля
            "symbol": universe,                          #   тикер
            "weight": weights,                           #   вес
            "exp_return_annual": mu,                     #   ожидаемая годовая доходность
            "vol_annual": vol,                           #   годовая волатильность
            "relationship": [self._classify(s, primary, corr) for s in universe],  # тип связи с BTC
        })
        df = df[df["weight"] > 1e-4].sort_values("weight", ascending=False).reset_index(drop=True)  # значимые веса
        return df                                        # таблица состава

    @staticmethod
    def _classify(symbol, primary, corr) -> str:
        if symbol == primary:                            # сам первичный актив…
            return "primary"                             #   → метка primary
        c = float(corr.get(symbol, 0.0)) if corr is not None and len(corr) else 0.0  # корреляция с BTC
        if c >= 0.5:                                     # сильная положительная…
            return "positive"
        if c <= -0.3:                                    # отрицательная…
            return "inverse"
        if abs(c) < 0.2:                                 # около нуля…
            return "neutral"
        return "weak"                                    # слабая связь
