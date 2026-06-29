"""Strongly-typed, file-driven configuration.

All tunable parameters of the system live in ``config/*.yaml`` and are validated
here into immutable :class:`pydantic` models. No business parameter is hard-coded
in the source: agents read everything from :class:`SystemConfig`.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.core (каркас).
НАЗНАЧЕНИЕ:  ТИПИЗИРОВАННАЯ конфигурация. Превращает пять YAML-файлов из config/
             в одну валидированную (Pydantic) структуру SystemConfig. Ни один
             бизнес-параметр не зашит в код — агенты читают всё отсюда. Опечатка
             или неверный тип значения падают сразу при загрузке.
ИМПОРТИРУЕТ:
  - pathlib.Path : пути к каталогу/файлам конфигурации.
  - typing       : аннотации Any/Dict/List/Optional.
  - yaml         : чтение YAML-файлов (safe_load).
  - pydantic     : BaseModel/Field — валидация и значения по умолчанию.
ЭКСПОРТИРУЕТ:
  - SystemConfig : корневая модель, агрегирующая все подсистемы.
  - load_config(): чтение + глубокое слияние YAML + overrides → SystemConfig.
  - множество под-моделей (InvestmentConfig, HestonConfig, RiskConfig, …).
КЕМ ИСПОЛЬЗУЕТСЯ:
  - core/context.py хранит config; ВСЕ агенты читают context.config.<секция>.
СООТВЕТСТВИЕ YAML→МОДЕЛЬ:
  system.yaml→(seed,run_id,investment,horizons,paths,runtime),
  data.yaml→data, market.yaml→(market_analysis,heston,greeks,hedging),
  portfolio.yaml→(optimization,risk,backtest), diagnostics.yaml→(diagnostic,
  explainability,dashboard,logging).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from pathlib import Path                                 # пути к config/ и его файлам
from typing import Any, Dict, List, Optional             # аннотации

import yaml                                              # парсер YAML
from pydantic import BaseModel, Field                    # модели валидации и фабрики значений


class _Frozen(BaseModel):
    """Base model: validated on assignment and forbids unknown keys."""

    # extra="forbid" => неизвестные ключи в YAML вызовут ошибку (защита от опечаток);
    # validate_assignment=True => присваивание полю тоже валидируется.
    model_config = {"frozen": False, "extra": "forbid", "validate_assignment": True}


class InvestmentConfig(_Frozen):                         # секция investment (system.yaml)
    capital_usd: float = 10_000_000.0                    # капитал фонда, $10M
    risk_budget_pct: float = 0.02  # max fraction of capital allowed at risk (VaR)   # бюджет риска: 2% дн. VaR
    transaction_fee_pct: float = 0.0003  # spot/linear taker fee                      # комиссия спота (0.03%)
    option_fee_pct: float = 0.0003  # per-contract fee on the underlying notional     # комиссия опциона (0.03%)
    option_fee_cap_pct: float = 0.125  # fee capped at this fraction of option price  # кап комиссии: 12.5% цены


class HorizonsConfig(_Frozen):                           # секция horizons (system.yaml)
    analysis_days: int = 90  # historical look-back window     # окно истории для анализа (90 дн.)
    forecast_days: int = 1  # forecasting horizon              # горизонт прогноза (1 дн.)
    trading_days_per_year: int = 365  # crypto trades 24/7     # торговых дней в году (крипта 24/7)


class PathsConfig(_Frozen):                              # секция paths (system.yaml) — все каталоги проекта
    data_dir: str = "data"                               # корень данных
    raw_dir: str = "data/raw"                            # сырые данные
    processed_dir: str = "data/processed"                # обработанные данные
    artifacts_dir: str = "artifacts"                     # корень артефактов прогона
    checkpoint_dir: str = "artifacts/checkpoints"        # чекпойнты
    calibration_dir: str = "artifacts/calibration"       # данные калибровки
    results_dir: str = "artifacts/results"               # результаты (parquet, дашборды)
    log_dir: str = "artifacts/logs"                      # логи

    def ensure(self, root: Path) -> "PathsConfig":
        for field in self.model_fields:                  # перебираем все поля-пути модели…
            (root / getattr(self, field)).mkdir(parents=True, exist_ok=True)  # …создаём каталог
        return self                                      # возвращаем себя (для chaining)


class DataConfig(_Frozen):                               # секция data (data.yaml)
    provider: str = "bundled"  # one of: bundled | synthetic | binance   # источник данных
    primary_symbol: str = "BTCUSDT"                      # первичный (хеджируемый) актив
    quote_currency: str = "USDT"                         # котировочная валюта
    universe_size: int = 100                             # размер вселенной спот-инструментов
    symbols: List[str] = Field(default_factory=list)     # явный список (пусто → дефолтная вселенная)
    bar_interval: str = "1d"                             # интервал баров (дневные)
    option_expiry_days: int = 30                         # срок до экспирации опционов (30 дн.)
    options_universe: List[str] = Field(default_factory=lambda: ["BTCUSDT"])  # активы с опционами
    n_strikes_per_expiry: int = 11                       # число страйков в цепочке
    strike_width_pct: float = 0.6  # +-60% around forward   # ширина сетки страйков (±60%)
    binance_base_url: str = "https://api.binance.com"    # базовый URL Binance REST
    request_timeout_s: float = 15.0                      # таймаут HTTP-запроса
    cache_raw: bool = True                               # кэшировать ли сырые данные на диск


class CorrelationConfig(_Frozen):                        # вложенная: методы корреляции (market.yaml)
    methods: List[str] = Field(                          # какие методы корреляции считать
        default_factory=lambda: ["pearson", "spearman", "kendall", "dcc_garch", "cointegration"]
    )
    rolling_window: int = 30                             # окно скользящей корреляции
    positive_threshold: float = 0.5                      # порог «сильной положительной» связи
    negative_threshold: float = -0.3                     # порог «отрицательной» связи
    zero_band: float = 0.1                               # зона около нуля (связь незначима)
    dcc_a: float = 0.02                                  # параметр a модели DCC-GARCH
    dcc_b: float = 0.95                                  # параметр b модели DCC-GARCH
    dcc_max_iter: int = 50                               # макс. итераций оценки DCC
    cointegration_method: str = "both"  # engle_granger | johansen | both   # метод коинтеграции
    cointegration_pvalue: float = 0.05                   # порог p-value коинтеграции
    johansen_det_order: int = 0                          # порядок детерминированного тренда (Johansen)
    johansen_k_ar_diff: int = 1                          # лаги разностей (Johansen)


class RankingWeights(_Frozen):                           # веса критериев рейтинга инструментов хеджа
    correlation: float = 0.30                            # вклад корреляции
    stability: float = 0.25                              # вклад устойчивости связи
    liquidity: float = 0.15                              # вклад ликвидности
    hedge_cost: float = 0.15                             # вклад стоимости хеджа (меньше — лучше)
    risk_reduction: float = 0.15                         # вклад снижения риска


class MarketAnalysisConfig(_Frozen):                     # секция market_analysis (market.yaml)
    vol_window: int = 30                                 # окно оценки волатильности
    vol_of_vol_window: int = 30                          # окно оценки vol-of-vol
    confidence_level: float = 0.95                       # доверительный уровень
    regime_n_states: int = 3                             # число режимов рынка (кластеры)
    regime_window: int = 30                              # окно определения режима
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)  # под-конфиг корреляций
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)    # веса рейтинга
    top_n_hedge_instruments: int = 10                    # сколько лучших инструментов хеджа отобрать


class HestonConfig(_Frozen):                             # секция heston (market.yaml)
    calibration_method: str = "iv_surface"  # iv_surface | mle   # метод калибровки Хестона
    num_iter: int = 50                                   # макс. итераций оптимизатора калибровки
    tol: float = 1e-8                                    # допуск сходимости
    initial_params: List[float] = Field(default_factory=lambda: [0.04, 2.0, 0.04, 0.5, -0.5])  # v0,κ,θ,ε,ρ
    mle_n_steps: int = 90                                # число шагов для метода MLE
    stability_window: int = 10                           # окно контроля устойчивости параметров
    stability_max_rel_change: float = 0.5               # макс. относительное изменение (устойчивость)
    benchmarks: List[str] = Field(default_factory=lambda: ["black_scholes", "sabr"])  # модели-бенчмарки
    sabr_beta: float = 0.5                               # параметр β модели SABR (бенчмарк)
    flat_yield_fallback: float = 0.0                     # ставка по умолчанию, если нет кривой


class GreeksConfig(_Frozen):                             # секция greeks (market.yaml)
    engine: str = "analytical"  # analytical | mc        # движок расчёта греков
    spot_bump_pct: float = 0.01                          # сдвиг спота для конечных разностей (1%)
    vol_bump: float = 0.01                               # сдвиг волатильности
    time_bump_days: float = 1.0                          # сдвиг времени (для theta), дн.
    rate_bump: float = 0.0001                            # сдвиг ставки (для rho)
    mc_n_paths: int = 50_000                             # число траекторий Монте-Карло
    mc_max_dt: float = 0.0137  # ~5/365                  # макс. шаг по времени в МК
    mc_min_steps: int = 40                               # мин. число шагов в МК
    mc_minimum_var: float = 0.01                         # нижняя отсечка дисперсии в МК
    greeks_to_compute: List[str] = Field(               # список греков для расчёта
        default_factory=lambda: ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm"]
    )


class HedgingConfig(_Frozen):                            # секция hedging (market.yaml)
    delta_eps: float = 0.0                               # допуск по дельте (порог ребаланса)
    vega_eps: float = 0.0                                # допуск по веге (порог ребаланса)
    delta_green_zone: float = 0.05  # |delta|/capital fraction considered balanced   # «зелёная» зона дельты
    delta_red_zone: float = 0.15                         # «красная» зона дельты (нужен ребаланс)
    target_delta: float = 0.0                            # целевая дельта (нейтраль)
    target_vega: float = 0.0                             # целевая вега (нейтраль)
    hedge_instrument_strike_moneyness: float = 1.0  # ATM call by default   # moneyness хедж-опциона
    liability_put_moneyness: float = 0.95  # protective put strike as a fraction of spot   # страйк put-обязательства
    vega_call_moneyness: float = 1.0  # vega-hedge call strike as a fraction of spot        # страйк vega-call
    calibration_subsample: int = 1  # calibrate every k-th slice (1 = every slice)          # шаг калибровки


class OptimizationConfig(_Frozen):                       # секция optimization (portfolio.yaml)
    method: str = "max_diversification"  # fallback primary method   # метод по умолчанию (если авто-выбор off)
    methods: List[str] = Field(                          # пул методов для авто-выбора
        default_factory=lambda: [
            "mean_variance",                             #   среднее-дисперсия (Марковиц)
            "risk_parity",                               #   паритет риска
            "min_variance",                              #   минимум дисперсии
            "max_diversification",                       #   максимум диверсификации
            "cvar",                                      #   минимизация CVaR
        ]
    )
    rebalance_frequency_days: int = 5                    # частота ребалансировки (дн.)
    max_turnover: float = 0.5                            # макс. оборот при ребалансе
    transaction_cost_aversion: float = 1.0              # неприятие транзакционных издержек
    risk_aversion: float = 5.0                           # коэффициент неприятия риска (mean-variance)
    cvar_alpha: float = 0.95                             # уровень α для CVaR
    long_only: bool = True                               # только длинные позиции
    max_weight: float = 0.2  # cap concentration to enforce diversification   # макс. вес одного актива (20%)
    # investable portfolio construction / selection      # --- построение/выбор инвест-портфеля ---
    portfolio_universe_size: int = 15  # number of instruments held in the portfolio   # активов в портфеле
    lookback_days: int = 30  # trailing window for re-estimating weights                # окно переоценки весов
    min_expected_return: float = 0.0  # require a profitable portfolio when selecting    # порог доходности
    diversification_weight: float = 0.5  # weight of diversification vs Sharpe in selection  # вес диверсификации
    auto_select_method: bool = True  # pick the best method by the selection score           # авто-выбор метода


class StopLossConfig(_Frozen):                           # вложенная: стоп-лоссы (portfolio.yaml → risk.stop_loss)
    enabled: bool = True                                 # включены ли стоп-лоссы
    atr_window: int = 14                                 # окно ATR
    atr_multiplier: float = 3.0                          # множитель ATR для дистанции стопа
    var_multiplier: float = 1.5                          # множитель VaR для дистанции стопа
    trailing: bool = True                                # использовать ли трейлинг-стоп
    trailing_atr_multiplier: float = 2.5                 # множитель ATR для трейлинга
    min_stop_pct: float = 0.02                           # мин. дистанция стопа (2%)
    max_stop_pct: float = 0.25                           # макс. дистанция стопа (25%)
    recalibrate_window: int = 30                         # окно перерасчёта стопа


class RiskConfig(_Frozen):                               # секция risk (portfolio.yaml)
    var_method: str = "historical"  # historical | gaussian | cornish_fisher   # метод оценки VaR
    var_confidence: float = 0.95                         # доверительный уровень VaR
    cvar_confidence: float = 0.95                        # доверительный уровень CVaR
    var_limit_pct: float = 0.05                          # лимит VaR (5%)
    max_drawdown_limit_pct: float = 0.25                 # лимит макс. просадки (25%)
    leverage_limit: float = 3.0                          # лимит плеча (3×)
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)  # под-конфиг стоп-лоссов


class BacktestConfig(_Frozen):                           # секция backtest (portfolio.yaml)
    mode: str = "walk_forward"                           # режим бэктеста (walk-forward)
    train_window: int = 30                               # окно обучения (дн.)
    test_window: int = 5                                 # окно теста (дн.)
    step: int = 5                                        # шаг сдвига окна
    purge: int = 0                                       # «очистка» баров между train/test (purge)
    embargo: int = 0                                     # «эмбарго» после теста
    account_survivorship_bias: bool = True               # учитывать ли survivorship bias
    account_selection_bias: bool = True                  # учитывать ли selection bias
    account_transaction_cost_bias: bool = True           # учитывать ли transaction-cost bias
    stress_scenarios: List[Dict[str, Any]] = Field(      # набор стресс-сценариев (шок спота/волы)
        default_factory=lambda: [
            {"name": "crash_-10pct", "spot_shock": -0.10, "vol_shock": 0.50},  # обвал −10%, vol +50%
            {"name": "crash_-5pct", "spot_shock": -0.05, "vol_shock": 0.25},   # обвал −5%, vol +25%
            {"name": "rally_+5pct", "spot_shock": 0.05, "vol_shock": -0.10},   # рост +5%, vol −10%
            {"name": "rally_+10pct", "spot_shock": 0.10, "vol_shock": -0.20},  # рост +10%, vol −20%
            {"name": "vol_spike", "spot_shock": 0.0, "vol_shock": 1.0},        # скачок только волатильности
        ]
    )


class DiagnosticConfig(_Frozen):                         # секция diagnostic (diagnostics.yaml)
    drift_method: str = "psi"  # psi | ks                # метод оценки дрейфа данных
    drift_threshold: float = 0.2                         # порог дрейфа
    degradation_metric: str = "rmse"                     # метрика деградации прогноза
    degradation_window: int = 10                         # окно оценки деградации
    degradation_threshold: float = 2.0  # x baseline     # порог деградации (× базовой)
    confidence_weights: Dict[str, float] = Field(        # веса компонент индекса доверия
        default_factory=lambda: {
            "calibration": 0.25,                         #   качество калибровки
            "data_drift": 0.20,                          #   стабильность данных (дрейф)
            "forecast_error": 0.20,                      #   ошибка прогноза
            "hedge_quality": 0.20,                       #   качество хеджа
            "risk_compliance": 0.15,                     #   соблюдение риск-лимитов
        }
    )


class ExplainabilityConfig(_Frozen):                     # секция explainability (diagnostics.yaml)
    language: str = "ru"                                 # язык объяснений по умолчанию
    max_reasons: int = 6                                 # макс. число причин в объяснении
    decimals: int = 4                                    # знаков после запятой в числах


class DashboardConfig(_Frozen):                          # секция dashboard (diagnostics.yaml)
    output_html: str = "artifacts/results/dashboard.html"  # путь основного HTML-дашборда
    output_dir: str = "artifacts/results"  # per-language dashboards go here   # каталог дашбордов по языкам
    languages: List[str] = Field(default_factory=lambda: ["ru", "en"])        # языки дашборда
    delta_green_zone: float = 0.05                       # «зелёная» зона дельты (для индикатора)
    delta_red_zone: float = 0.15                         # «красная» зона дельты
    height: int = 2200                                   # высота дашборда, px
    width: int = 1300                                    # ширина дашборда, px
    theme: str = "plotly_dark"                           # тема Plotly


class LoggingConfig(_Frozen):                            # секция logging (diagnostics.yaml)
    level: str = "INFO"                                  # уровень логирования
    console: bool = True                                 # вывод в консоль
    jsonl: bool = True                                   # запись JSONL-файла
    file_name: str = "cryptohedge.jsonl"                 # имя файла лога
    timing: bool = True                                  # логировать тайминги операций


class RuntimeConfig(_Frozen):                            # секция runtime (system.yaml)
    parallel: bool = True                                # разрешить параллелизм
    n_jobs: int = -1                                     # число потоков (-1 = все ядра)
    checkpointing: bool = True                           # включить чекпойнты
    resume: bool = True                                  # возобновлять прогон из чекпойнтов


class SystemConfig(_Frozen):
    """Root configuration aggregating all sub-systems."""

    seed: int = 90909090                                 # ЕДИНЫЙ seed для всех RNG (воспроизводимость)
    run_id: str = "default"                              # идентификатор прогона (подкаталог чекпойнтов)
    investment: InvestmentConfig = Field(default_factory=InvestmentConfig)          # капитал/риск/комиссии
    horizons: HorizonsConfig = Field(default_factory=HorizonsConfig)                # горизонты
    paths: PathsConfig = Field(default_factory=PathsConfig)                         # каталоги
    data: DataConfig = Field(default_factory=DataConfig)                            # данные
    market_analysis: MarketAnalysisConfig = Field(default_factory=MarketAnalysisConfig)  # анализ рынка
    heston: HestonConfig = Field(default_factory=HestonConfig)                      # модель Хестона
    greeks: GreeksConfig = Field(default_factory=GreeksConfig)                      # греки
    hedging: HedgingConfig = Field(default_factory=HedgingConfig)                   # хеджирование
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)    # оптимизация портфеля
    risk: RiskConfig = Field(default_factory=RiskConfig)                            # риск-менеджмент
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)                # бэктест
    diagnostic: DiagnosticConfig = Field(default_factory=DiagnosticConfig)          # самодиагностика
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)  # объяснимость
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)             # дашборд
    logging: LoggingConfig = Field(default_factory=LoggingConfig)                   # логирование
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)                   # runtime-настройки


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)                                     # копия базового словаря
    for key, value in override.items():                 # перебираем ключи override…
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):  # оба значения — словари?
            out[key] = _deep_merge(out[key], value)     # …рекурсивно сливаем вложенные словари
        else:
            out[key] = value                            # иначе override побеждает (перетирает)
    return out                                          # результат слияния


def load_config(
    config_dir: str | Path = "config",                  # каталог с YAML-файлами конфигурации
    overrides: Optional[Dict[str, Any]] = None,         # переопределения поверх файлов (для тестов/ноутбука)
) -> SystemConfig:
    """Load and validate the configuration.

    All ``*.yaml`` files in ``config_dir`` are read in sorted order and deep-merged.
    A ``main.yaml`` (if present) is always merged last so it can override modules.
    The optional ``overrides`` mapping is applied on top (useful for notebooks/tests).
    """
    config_dir = Path(config_dir)                       # нормализуем путь к каталогу конфигов
    merged: Dict[str, Any] = {}                         # сюда сольём все YAML-файлы

    if config_dir.is_dir():                             # если каталог существует…
        files = sorted(p for p in config_dir.glob("*.yaml") if p.name != "main.yaml")  # все *.yaml, кроме main
        main = config_dir / "main.yaml"                 # путь к необязательному main.yaml
        if main.exists():                               # если main.yaml есть…
            files.append(main)                          # …добавляем его в КОНЕЦ (сливается последним)
        for path in files:                              # по каждому файлу…
            with open(path, "r", encoding="utf-8") as handle:  # открываем YAML
                data = yaml.safe_load(handle) or {}     # парсим (пустой файл → {})
            if not isinstance(data, dict):              # верхний уровень YAML обязан быть словарём…
                raise ValueError(f"Config file {path} must contain a top-level mapping")  # …иначе ошибка
            merged = _deep_merge(merged, data)          # сливаем файл в общий словарь

    if overrides:                                       # если заданы программные переопределения…
        merged = _deep_merge(merged, overrides)         # …применяем их поверх всего

    return SystemConfig(**merged)                       # валидируем итоговый словарь в типизированный конфиг
