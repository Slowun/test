"""Greeks value objects (per-instrument and aggregated portfolio sensitivities).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.domain (внутренний слой).
НАЗНАЧЕНИЕ:  структуры «греков» — чувствительностей цены опциона к факторам.
             Greeks — по одному инструменту; PortfolioGreeks — агрегат по книге.
             Поддерживают масштабирование на объём (scaled) и сложение (__add__),
             что позволяет агрегировать греки книги простым суммированием.
ИМПОРТИРУЕТ: dataclasses (frozen dataclass, asdict, fields), typing.
ЭКСПОРТИРУЕТ: Greeks, PortfolioGreeks.
КЕМ ИСПОЛЬЗУЕТСЯ: services/greeks.py, агенты 4 (GreeksCalculation) и 5 (Hedging).
ГРЕКИ: delta(Δ), gamma(Γ), vega(ν), theta(Θ), rho(ρ), vanna, volga, charm.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from dataclasses import dataclass, asdict, fields        # frozen dataclass + перечисление полей
from typing import Dict                                  # аннотация


@dataclass(frozen=True)
class Greeks:
    """First- and second-order option sensitivities.

    All greeks are expressed for the held ``notional`` of the instrument.
    """

    premium: float = 0.0                                 # цена (премия) опциона
    delta: float = 0.0                                   # Δ — чувствительность к цене базового актива
    gamma: float = 0.0                                   # Γ — чувствительность дельты к цене
    vega: float = 0.0                                    # ν — чувствительность к волатильности
    theta: float = 0.0                                   # Θ — чувствительность ко времени (распад)
    rho: float = 0.0                                     # ρ — чувствительность к процентной ставке
    vanna: float = 0.0                                   # vanna — d²/dspot·dvol
    volga: float = 0.0                                   # volga — d²/dvol² (vomma)
    charm: float = 0.0                                   # charm — d²/dspot·dtime

    def scaled(self, qty: float) -> "Greeks":
        # Масштабирует ВСЕ поля на количество qty (позиция размера qty).
        return Greeks(**{f.name: getattr(self, f.name) * qty for f in fields(self)})

    def __add__(self, other: "Greeks") -> "Greeks":
        # Покомпонентное сложение греков (агрегация нескольких инструментов).
        return Greeks(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)                              # сериализация в словарь


@dataclass(frozen=True)
class PortfolioGreeks:
    """Aggregated greeks across the whole position book."""

    delta: float = 0.0                                   # суммарная Δ книги
    gamma: float = 0.0                                   # суммарная Γ книги
    vega: float = 0.0                                    # суммарная ν книги
    theta: float = 0.0                                   # суммарная Θ книги
    rho: float = 0.0                                     # суммарная ρ книги
    vanna: float = 0.0                                   # суммарная vanna
    volga: float = 0.0                                   # суммарная volga
    charm: float = 0.0                                   # суммарная charm
    premium: float = 0.0                                 # суммарная премия книги

    @classmethod
    def from_greeks(cls, greeks: Greeks) -> "PortfolioGreeks":
        # Конвертирует одиночные Greeks в агрегат портфеля (копирует поля).
        return cls(
            delta=greeks.delta,                          # переносим Δ
            gamma=greeks.gamma,                          # переносим Γ
            vega=greeks.vega,                            # переносим ν
            theta=greeks.theta,                          # переносим Θ
            rho=greeks.rho,                              # переносим ρ
            vanna=greeks.vanna,                          # переносим vanna
            volga=greeks.volga,                          # переносим volga
            charm=greeks.charm,                          # переносим charm
            premium=greeks.premium,                      # переносим премию
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)                              # сериализация в словарь
