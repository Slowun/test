"""Bilingual (Russian / English) label catalogue for the dashboard.

Centralising every user-facing string lets the dashboard agent emit a fully
Russian and a fully English version from a single rendering pass.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (интернационализация, i18n).
НАЗНАЧЕНИЕ:  централизованный каталог всех пользовательских строк на русском и
             английском. Позволяет агенту dashboard сгенерировать полностью
             русскую и полностью английскую версии за один проход рендеринга.
ИМПОРТИРУЕТ: typing.Dict (без внешних зависимостей).
КОНСТАНТЫ:
  - LABELS         : словарь {язык → {ключ → подпись}} для всех элементов UI.
  - METRIC_LABELS  : отображаемые названия метрик доходности/риска по языкам.
ЭКСПОРТИРУЕТ: t (перевод ключа), metric_label (название метрики),
             relationship_label (название типа связи с BTC).
КЕМ ИСПОЛЬЗУЕТСЯ: агенты dashboard и explainability.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import Dict                                  # аннотации

# LABELS: каталог подписей интерфейса дашборда по языкам ("ru"/"en").
# Ключи одинаковы в обоих языках — это гарантирует зеркальность версий.
LABELS: Dict[str, Dict[str, str]] = {
    "ru": {
        # document / headers
        "doc_title": "CryptoHedge — Дашборд мониторинга хеджирования",
        "main_header": "CryptoHedge — мультиагентный мониторинг хеджирования валютного риска",
        "lang_switch": "English version",
        "explanation_header": "Объяснение решений (Explainability Agent)",
        "generated": "Сгенерировано",
        # section headings
        "sec_timeseries": "1. Синхронизированный мониторинг: цена, PnL, греки, хедж",
        "sec_greeks": "2. Панель греков с цветовой индикацией дисбаланса",
        "sec_portfolio_constituents": "3. Состав портфеля (инструменты и веса)",
        "sec_portfolio_equity": "4. Динамика стоимости портфеля и ребалансировки",
        "sec_portfolio_weights": "5. Эволюция весов портфеля (ребалансировка)",
        "sec_diversification": "6. Диверсификация портфеля и сравнение методов",
        "sec_heatmap": "7. Тепловая карта греков по страйкам",
        "sec_stress": "8. Стресс-тесты: хедж против голой позиции",
        "sec_metrics": "9. Ключевые метрики стратегии",
        "sec_costs": "10. Издержки и частота ребалансировок хеджа",
        "sec_rankings": "11. Рейтинг инструментов для хеджирования",
        # timeseries
        "ts_sub_spot": "Цена спота и трейлинг-стоп",
        "ts_sub_pnl": "PnL стратегии и комиссии",
        "ts_sub_thetavega": "Тета и Вега портфеля",
        "ts_sub_hedges": "Дельта-хедж и Вега-хедж",
        "ser_spot": "Спот",
        "ser_trailing_stop": "Трейлинг-стоп",
        "ser_pnl": "PnL",
        "ser_fees_cum": "Комиссии (накоп.)",
        "ser_theta": "Тета",
        "ser_vega": "Вега",
        "ser_delta_hedge": "Дельта-хедж",
        "ser_vega_hedge": "Вега-хедж",
        "axis_usd": "USD",
        # greeks indicators
        "ind_delta_ratio": "|Δ| / капитал",
        "ind_gamma": "Гамма",
        "ind_vega": "Вега портфеля",
        "ind_theta": "Тета",
        "zone_hint": "зелёная — баланс, жёлтая — внимание, красная — ребалансировать",
        # portfolio constituents
        "col_symbol": "Инструмент",
        "col_weight": "Вес",
        "col_exp_return": "Ожид. доходность (год)",
        "col_vol": "Волатильность (год)",
        "col_relationship": "Связь с BTC",
        # portfolio equity
        "ser_portfolio": "Портфель (оптимизированный)",
        "ser_benchmark_eqw": "Равновзвешенный бенчмарк",
        "ser_rebalance": "Ребалансировка",
        "axis_equity": "Стоимость (база 1.0)",
        "axis_weight": "Вес",
        # diversification indicators
        "ind_div_ratio": "Коэф. диверсификации",
        "ind_eff_n": "Эффективное число активов",
        "ind_max_weight": "Макс. вес актива",
        "ind_hhi": "Индекс концентрации (HHI)",
        "div_ret_axis": "Доходность бэктеста",
        "div_dr_axis": "Коэф. диверсификации",
        "div_methods_title": "Методы оптимизации: доходность и диверсификация",
        "div_bar_return": "Доходность",
        "div_bar_dr": "Коэф. диверсификации",
        # heatmap
        "heatmap_x": "Moneyness (K/S)",
        # stress
        "ser_unhedged": "Без хеджа (голый BTC)",
        "ser_hedged": "С хеджем (нетто)",
        "stress_yaxis": "PnL, $",
        # metrics table
        "col_metric": "Метрика",
        "col_value": "Значение",
        # costs
        "ser_rebalances": "Ребалансировки",
        # relationships
        "rel_positive": "прямая",
        "rel_inverse": "обратная",
        "rel_neutral": "нейтральная",
        "rel_weak": "слабая",
        "rel_cointegrated": "коинтеграция",
        "rel_primary": "первичный",
    },
    "en": {
        "doc_title": "CryptoHedge — Hedging Monitoring Dashboard",
        "main_header": "CryptoHedge — Multi-agent Monitoring of Currency (Volatility) Risk Hedging",
        "lang_switch": "Русская версия",
        "explanation_header": "Decision Explanation (Explainability Agent)",
        "generated": "Generated",
        "sec_timeseries": "1. Synchronized Monitoring: Price, PnL, Greeks, Hedge",
        "sec_greeks": "2. Greeks Panel with Imbalance Colour Indicators",
        "sec_portfolio_constituents": "3. Portfolio Constituents (Instruments & Weights)",
        "sec_portfolio_equity": "4. Portfolio Value Dynamics & Rebalancing",
        "sec_portfolio_weights": "5. Portfolio Weight Evolution (Rebalancing)",
        "sec_diversification": "6. Portfolio Diversification & Method Comparison",
        "sec_heatmap": "7. Greeks Heatmap Across Strikes",
        "sec_stress": "8. Stress Tests: Hedged vs Naked Position",
        "sec_metrics": "9. Key Strategy Metrics",
        "sec_costs": "10. Hedge Costs & Rebalancing Frequency",
        "sec_rankings": "11. Hedging Instrument Ranking",
        "ts_sub_spot": "Spot Price & Trailing Stop",
        "ts_sub_pnl": "Strategy PnL & Fees",
        "ts_sub_thetavega": "Portfolio Theta & Vega",
        "ts_sub_hedges": "Delta Hedge & Vega Hedge",
        "ser_spot": "Spot",
        "ser_trailing_stop": "Trailing stop",
        "ser_pnl": "PnL",
        "ser_fees_cum": "Fees (cumulative)",
        "ser_theta": "Theta",
        "ser_vega": "Vega",
        "ser_delta_hedge": "Delta hedge",
        "ser_vega_hedge": "Vega hedge",
        "axis_usd": "USD",
        "ind_delta_ratio": "|Δ| / capital",
        "ind_gamma": "Gamma",
        "ind_vega": "Portfolio Vega",
        "ind_theta": "Theta",
        "zone_hint": "green — balanced, amber — watch, red — rebalance",
        "col_symbol": "Instrument",
        "col_weight": "Weight",
        "col_exp_return": "Exp. return (ann.)",
        "col_vol": "Volatility (ann.)",
        "col_relationship": "Link to BTC",
        "ser_portfolio": "Portfolio (optimized)",
        "ser_benchmark_eqw": "Equal-weight benchmark",
        "ser_rebalance": "Rebalance",
        "axis_equity": "Value (base 1.0)",
        "axis_weight": "Weight",
        "ind_div_ratio": "Diversification ratio",
        "ind_eff_n": "Effective number of assets",
        "ind_max_weight": "Max asset weight",
        "ind_hhi": "Concentration index (HHI)",
        "div_ret_axis": "Backtest return",
        "div_dr_axis": "Diversification ratio",
        "div_methods_title": "Optimization methods: return and diversification",
        "div_bar_return": "Return",
        "div_bar_dr": "Diversification ratio",
        "heatmap_x": "Moneyness (K/S)",
        "ser_unhedged": "Unhedged (naked BTC)",
        "ser_hedged": "Hedged (net)",
        "stress_yaxis": "PnL, $",
        "col_metric": "Metric",
        "col_value": "Value",
        "ser_rebalances": "Rebalances",
        "rel_positive": "positive",
        "rel_inverse": "inverse",
        "rel_neutral": "neutral",
        "rel_weak": "weak",
        "rel_cointegrated": "cointegrated",
        "rel_primary": "primary",
    },
}

# METRIC_LABELS: отображаемые названия метрик доходности/риска по языкам.
# Performance-metric display names per language.
METRIC_LABELS: Dict[str, Dict[str, str]] = {
    "ru": {
        "roi": "ROI", "cagr": "CAGR", "sharpe": "Коэф. Шарпа", "sortino": "Коэф. Сортино",
        "calmar": "Коэф. Калмара", "max_drawdown": "Макс. просадка", "profit_factor": "Profit Factor",
        "win_rate": "Доля прибыльных", "var": "VaR", "cvar": "CVaR",
        "expected_shortfall": "Expected Shortfall", "beta": "Бета", "alpha": "Альфа",
        "information_ratio": "Information Ratio", "volatility": "Волатильность",
    },
    "en": {
        "roi": "ROI", "cagr": "CAGR", "sharpe": "Sharpe", "sortino": "Sortino",
        "calmar": "Calmar", "max_drawdown": "Max Drawdown", "profit_factor": "Profit Factor",
        "win_rate": "Win Rate", "var": "VaR", "cvar": "CVaR",
        "expected_shortfall": "Expected Shortfall", "beta": "Beta", "alpha": "Alpha",
        "information_ratio": "Information Ratio", "volatility": "Volatility",
    },
}


def t(lang: str, key: str) -> str:
    """Translate ``key`` into ``lang`` (falls back to English, then the key)."""
    lang = lang if lang in LABELS else "en"             # неизвестный язык → английский
    return LABELS[lang].get(key) or LABELS["en"].get(key, key)  # перевод → англ. → сам ключ


def metric_label(lang: str, key: str) -> str:
    lang = lang if lang in METRIC_LABELS else "en"      # неизвестный язык → английский
    return METRIC_LABELS[lang].get(key, METRIC_LABELS["en"].get(key, key))  # название метрики


def relationship_label(lang: str, value: str) -> str:
    return t(lang, f"rel_{value}") if f"rel_{value}" in LABELS.get(lang, {}) else value  # тип связи → подпись
