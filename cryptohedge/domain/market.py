"""Market-domain value objects.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.domain (внутренний слой, без зависимостей).
НАЗНАЧЕНИЕ:  неизменяемые структуры данных РЫНКА: опционный контракт, оценка
             волатильности, параметры Хестона, рейтинг инструмента хеджа.
ИМПОРТИРУЕТ: dataclasses (frozen dataclass + asdict), typing.
ЭКСПОРТИРУЕТ: OptionContract, VolatilityEstimate, HestonParameters, InstrumentRanking.
КЕМ СОЗДАЁТСЯ:
  - OptionContract       : services/heston_pricing, агенты 4/5.
  - VolatilityEstimate   : services/volatility, агент 2 (MarketAnalysis).
  - HestonParameters     : services/calibration, агент 3 (HestonCalibration).
  - InstrumentRanking    : services/correlation, агент 2 (рейтинг хедж-инструментов).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from dataclasses import dataclass, asdict                # frozen dataclass + сериализация в словарь
from typing import Dict, Optional                        # аннотации


@dataclass(frozen=True)                                  # неизменяемая структура
class OptionContract:
    """A single vanilla option instrument."""

    underlying: str                                      # базовый актив (например, BTCUSDT)
    strike: float                                        # страйк (цена исполнения)
    expiry_ts: int  # nanoseconds                        # момент экспирации в наносекундах
    is_call: bool                                        # True=call, False=put
    notional: float = 1.0                                # номинал (объём контракта)

    @property
    def opt_type(self) -> str:
        return "call" if self.is_call else "put"         # строковый тип опциона


@dataclass(frozen=True)
class VolatilityEstimate:
    """Daily volatility, volatility-of-volatility and a confidence interval."""

    daily_vol: float                                     # дневная волатильность
    annualized_vol: float                                # годовая волатильность
    vol_of_vol: float                                    # волатильность волатильности
    ci_low: float                                        # нижняя граница доверительного интервала
    ci_high: float                                       # верхняя граница доверительного интервала
    confidence_level: float                              # доверительный уровень (например, 0.95)
    horizon_days: int                                    # горизонт оценки в днях

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)                              # сериализация всех полей в словарь


@dataclass(frozen=True)
class HestonParameters:
    """Calibrated Heston parameters (variance dynamics under the risk-neutral measure)."""

    v0: float                                            # начальная дисперсия
    kappa: float                                         # скорость возврата дисперсии к среднему (mean reversion)
    theta: float                                         # долгосрочная средняя дисперсия
    eps: float                                           # волатильность дисперсии (vol-of-vol)
    rho: float                                           # корреляция цены и дисперсии
    flat_yield: float = 0.0                              # плоская безрисковая ставка
    calibration_error: float = float("nan")             # ошибка калибровки (RMSE по IV)
    feller_satisfied: bool = False                       # выполнено ли условие Феллера (дисперсия > 0)

    def as_array(self) -> tuple:
        return (self.v0, self.kappa, self.theta, self.eps, self.rho)  # параметры как кортеж (для квант-функций)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)                              # сериализация в словарь

    @property
    def feller_condition(self) -> float:
        """``2*kappa*theta - eps^2`` (>= 0 keeps variance strictly positive)."""
        return 2.0 * self.kappa * self.theta - self.eps**2  # значение условия Феллера (>=0 — дисперсия положительна)


@dataclass(frozen=True)
class InstrumentRanking:
    """Multi-criteria ranking of a candidate hedging instrument vs the primary asset."""

    symbol: str                                          # тикер кандидата в инструменты хеджа
    pearson: float                                       # корреляция Пирсона с первичным активом
    spearman: float                                      # корреляция Спирмена
    kendall: float                                       # корреляция Кендалла
    dcc_mean: float                                      # средняя динамическая корреляция (DCC-GARCH)
    cointegrated: bool                                   # коинтегрирован ли с первичным активом
    stability: float                                     # устойчивость связи во времени
    liquidity: float                                     # ликвидность инструмента
    hedge_cost: float                                    # стоимость хеджа этим инструментом
    risk_reduction: float                                # ожидаемое снижение риска
    score: float                                         # итоговый интегральный рейтинг
    relationship: str  # 'positive' | 'inverse' | 'neutral'  # тип связи с первичным активом

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)                              # сериализация в словарь
