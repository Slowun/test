"""Market Analysis Agent.

Role: characterise the market - volatility & vol-of-vol of the primary asset with
a confidence interval, the BTC notional that must be hedged, the dependence of
every other instrument on BTC (Pearson/Spearman/Kendall/DCC-GARCH/cointegration),
the market regime, and a multi-criteria ranking selecting the best hedging
instruments.

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       2 / 11 — MarketAnalysisAgent.
НАЗНАЧЕНИЕ:  «характеризует» рынок: волатильность и vol-of-vol первичного актива
             с доверительным интервалом; объём BTC к хеджированию; зависимость
             остальных инструментов от BTC (5 мер связи); режим рынка; и строит
             многокритериальный РЕЙТИНГ лучших инструментов хеджа.
ВХОД (consumes):  DATA_READY (от агента 1).
ВЫХОД (produces): ANALYSIS_READY → агенту heston_calibration.
КЛАДЁТ НА ДОСКУ:  volatility, hedge_sizing, correlation_static, rankings,
                  rankings_df, hedge_universe, regime.
ИМПОРТИРУЕТ:
  - numpy/pandas                          : расчёты, агрегаты по барам.
  - services.correlation (как corr)       : меры связи, устойчивость, рейтинг.
  - services.volatility.*                 : оценка волатильности и объёма хеджа.
  - sklearn.cluster.KMeans (лениво)       : кластеризация режима рынка.
КОНФИГ:  config.market_analysis (окна, методы корреляции, веса рейтинга, режимы).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import List                                  # аннотация списков

import numpy as np                                       # числовые операции/кластеризация
import pandas as pd                                       # таблицы/группировки

from cryptohedge.core.agent import BaseAgent             # контракт агента
from cryptohedge.core.context import AgentContext        # контекст
from cryptohedge.core.message import Message, MessageType # сообщения
from cryptohedge.services import correlation as corr     # сервис корреляций (псевдоним corr)
from cryptohedge.services.volatility import estimate_volatility, size_primary_hedge  # волатильность и объём хеджа


class MarketAnalysisAgent(BaseAgent):
    name = "market_analysis"                             # имя агента / id этапа
    consumes = [MessageType.DATA_READY]                  # принимает DATA_READY
    produces = MessageType.ANALYSIS_READY                # выпускает ANALYSIS_READY
    checkpoint_keys = ["volatility", "hedge_sizing", "correlation_static", "rankings_df",  # ключи чекпойнта
                       "hedge_universe", "regime"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        cfg = context.config.market_analysis             # секция конфига анализа рынка
        primary = context.require("primary_symbol")      # первичный актив (обязателен на доске)
        spot_close: pd.DataFrame = context.require("spot_close")  # цены закрытия (обязательны)
        returns: pd.DataFrame = context.require("returns")       # доходности (обязательны)

        # ---- volatility & hedge sizing                 # --- волатильность и объём хеджа ---
        with log.timer("volatility"):                    # замеряем время оценки волатильности
            vol = estimate_volatility(                   # оцениваем волатильность первичного актива
                spot_close[primary].to_numpy(), window=cfg.vol_window,  #   ряд цен + окно
                vov_window=cfg.vol_of_vol_window, confidence_level=cfg.confidence_level,  #   vol-of-vol + дов. уровень
                horizon_days=context.config.horizons.forecast_days,    #   горизонт прогноза
                trading_days=context.config.horizons.trading_days_per_year,  #   дней в году
            )
        sizing = size_primary_hedge(                     # вычисляем требуемый объём первичного хеджа
            capital_usd=context.config.investment.capital_usd,  #   капитал
            spot=float(spot_close[primary].iloc[-1]),    #   текущая цена первичного актива
            vol=vol,                                     #   оценка волатильности
            risk_budget_pct=context.config.investment.risk_budget_pct,  #   бюджет риска
            confidence_level=cfg.confidence_level,       #   доверительный уровень
        )
        log.decision(                                    # лог-решение: объём хеджа и его обоснование
            "sized primary hedge",
            daily_vol=round(vol.daily_vol, 5), vol_of_vol=round(vol.vol_of_vol, 5),  #   волатильности
            ci=[round(vol.ci_low, 5), round(vol.ci_high, 5)],   #   доверительный интервал
            hedge_ratio=round(sizing.hedge_ratio, 4),    #   доля хеджирования
            quantity_to_hedge=round(sizing.quantity_to_hedge, 4),  #   объём к хеджу
        )

        # ---- static correlations + stability           # --- статические корреляции + устойчивость ---
        with log.timer("static_correlations"):           # замеряем время
            static = corr.static_correlations(returns, primary)  # Pearson/Spearman/Kendall к первичному
            stability = corr.rolling_stability(returns, primary, cfg.correlation.rolling_window)  # устойчивость связи

        # candidate pool for the heavier dynamic / cointegration analysis  # пул кандидатов для тяжёлого анализа
        candidates = static["pearson"].abs().sort_values(ascending=False)  # сортируем по |Pearson|
        pool = list(candidates.head(min(len(candidates), 3 * cfg.top_n_hedge_instruments)).index)  # берём топ-3N

        dcc = {}                                         # результаты DCC-GARCH (по умолчанию пусто)
        if "dcc_garch" in cfg.correlation.methods:       # если метод включён в конфиге…
            with log.timer("dcc_garch", n=len(pool)):    #   замеряем время
                dcc = corr.dcc_garch_correlations(       #   динамические корреляции по пулу
                    returns, primary, pool, a=cfg.correlation.dcc_a, b=cfg.correlation.dcc_b,
                    estimate=True, max_iter=cfg.correlation.dcc_max_iter,
                )
        cointegrated = {}                                # результаты коинтеграции (по умолчанию пусто)
        if "cointegration" in cfg.correlation.methods:   # если метод включён…
            with log.timer("cointegration", n=len(pool)):  #   замеряем время
                cointegrated = corr.cointegration(       #   тест коинтеграции по пулу
                    spot_close, primary, pool, method=cfg.correlation.cointegration_method,
                    pvalue=cfg.correlation.cointegration_pvalue,
                    det_order=cfg.correlation.johansen_det_order,
                    k_ar_diff=cfg.correlation.johansen_k_ar_diff,
                )

        # ---- liquidity & hedge-cost proxies from spot bars  # --- прокси ликвидности и стоимости хеджа ---
        liquidity, hedge_cost = self._liquidity_and_cost(context, static.index)  # считаем из баров

        rankings = corr.rank_instruments(                # строим многокритериальный рейтинг инструментов
            static.loc[pool], stability, dcc, cointegrated,
            liquidity.loc[pool] if set(pool).issubset(liquidity.index) else liquidity,    # ликвидность по пулу
            hedge_cost.loc[pool] if set(pool).issubset(hedge_cost.index) else hedge_cost,  # стоимость по пулу
            cfg.correlation, cfg.ranking_weights,        # настройки корреляций и веса критериев
        )
        rankings_df = pd.DataFrame([r.to_dict() for r in rankings])  # рейтинг → DataFrame (для дашборда/ноутбука)
        hedge_universe = [r.symbol for r in rankings[: cfg.top_n_hedge_instruments]]  # топ-N инструментов хеджа

        regime = self._detect_regime(returns[primary], cfg.regime_window, cfg.regime_n_states)  # режим рынка

        context.put("volatility", vol)                   # волатильность → на доску
        context.put("hedge_sizing", sizing)              # объём хеджа → на доску
        context.put("correlation_static", static)        # статические корреляции → на доску
        context.put("rankings", rankings)                # рейтинг (объекты) → на доску
        context.put("rankings_df", rankings_df)          # рейтинг (таблица) → на доску
        context.put("hedge_universe", hedge_universe)     # вселенная хеджа → на доску
        context.put("regime", regime)                    # режим рынка → на доску

        log.decision("selected hedge universe", instruments=hedge_universe,  # лог-решение: выбранные инструменты
                     top_scores=[round(r.score, 3) for r in rankings[: cfg.top_n_hedge_instruments]],
                     regime=regime["label"])

        payload = {                                      # краткая нагрузка для следующего агента
            "hedge_universe": hedge_universe,            #   выбранные инструменты
            "regime": regime["label"],                   #   метка режима
            "quantity_to_hedge": sizing.quantity_to_hedge,  #   объём хеджа
        }
        return Message(self.produces, self.name, "heston_calibration", payload=payload,  # ANALYSIS_READY дальше
                       correlation_id=message.correlation_id)

    def _liquidity_and_cost(self, context: AgentContext, symbols) -> tuple:
        bars: pd.DataFrame = context.require("spot_bars")  # полные OHLCV-бары
        grp = bars.groupby("symbol")                     # группируем по тикеру
        dollar_vol = (grp["close"].mean() * grp["volume"].mean())  # прокси ликвидности = ср.цена × ср.объём
        spread = ((grp["high"].mean() - grp["low"].mean()) / grp["close"].mean())  # прокси стоимости = относит. спред
        liquidity = dollar_vol.reindex([s for s in symbols]).fillna(dollar_vol.median())  # выравниваем по symbols
        hedge_cost = spread.reindex([s for s in symbols]).fillna(spread.median())          # пропуски → медиана
        liquidity.name, hedge_cost.name = "liquidity", "hedge_cost"  # называем серии
        return liquidity, hedge_cost                     # возвращаем обе серии

    def _detect_regime(self, primary_returns: pd.Series, window: int, n_states: int) -> dict:
        """Volatility-regime classification via K-means on (return, rolling vol)."""
        r = primary_returns.dropna()                     # доходности без пропусков
        roll_vol = r.rolling(window).std().bfill()       # скользящая волатильность (заполняем начало назад)
        feats = np.column_stack([r.to_numpy(), roll_vol.to_numpy()])  # признаки: (доходность, волатильность)
        try:
            from sklearn.cluster import KMeans           # ленивый импорт KMeans

            km = KMeans(n_clusters=n_states, n_init=10, random_state=0).fit(feats)  # кластеризуем на n_states
            labels = km.labels_                          # метки кластеров по точкам
            order = np.argsort(km.cluster_centers_[:, 1])  # by volatility   # сортируем кластеры по волатильности
            rank = {int(c): i for i, c in enumerate(order)}  # ранг кластера = его место по волатильности
            current = rank[int(labels[-1])]              # ранг текущего (последнего) состояния
            names = {0: "calm", 1: "normal", 2: "stressed"}  # человекочитаемые названия режимов
            label = names.get(current, f"state_{current}")   # метка текущего режима
        except Exception:                                # если sklearn недоступен/упал…
            current = int(roll_vol.iloc[-1] > roll_vol.median())  # …грубый бинарный режим по медиане волатильности
            label = "stressed" if current else "calm"    # метка по бинарному признаку
        return {"label": label, "current_vol": float(roll_vol.iloc[-1]),  # результат: метка + текущая/медианная вола
                "median_vol": float(roll_vol.median())}
