"""pyquant: low-level quantitative library (Black-Scholes, Heston, vol surfaces).

This package bundles the numerically heavy primitives used by the multi-agent
hedging system. The numba-accelerated, NumPy-only modules (``common``,
``utils``, ``black_scholes``, ``vol_surface``, ``heston``) can be imported
without PyTorch. The Monte-Carlo modules (``heston_sim``, ``barrier``, ``lsm``,
``torch_spline``) require PyTorch and are imported lazily by the callers that
need them, so the core analytical pipeline runs even when torch is absent.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        корень пакета pyquant (низкоуровневое количественное ядро).
НАЗНАЧЕНИЕ:  собирает численно тяжёлые примитивы для системы хеджирования.
             numba/NumPy-модули (common, utils, black_scholes, vol_surface,
             heston) импортируются без PyTorch; Monte-Carlo модули (heston_sim,
             barrier, lsm, torch_spline) требуют torch и грузятся лениво —
             поэтому аналитический конвейер работает даже без torch.
ЭКСПОРТИРУЕТ: имена подмодулей через __all__ (utils, common, black_scholes,
             vol_surface, heston).
КЕМ ИСПОЛЬЗУЕТСЯ: cryptohedge.services.heston_pricing (фасад над pyquant).
=============================================================================
"""

__all__ = [                                              # публичные подмодули пакета:
    "utils",                                             #   базовые численные примитивы
    "common",                                            #   value-объекты и кривые
    "black_scholes",                                     #   модель Блэка-Шоулза
    "vol_surface",                                       #   поверхность волатильности
    "heston",                                            #   модель Хестона
]
