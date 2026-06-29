"""Data-drift detection and model-degradation monitoring.

Used by the self-diagnostic agent to (a) detect distribution drift between a
reference and a recent window (PSI or KS), (b) track forecasting-error
degradation against a baseline, and (c) blend signals into a single confidence
score in ``[0, 1]``.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (мониторинг дрейфа и деградации).
НАЗНАЧЕНИЕ:  (a) детектирует дрейф распределения между эталонным и недавним окном
             (PSI или тест Колмогорова-Смирнова); (b) отслеживает рост ошибки
             прогноза относительно базовой; (c) сводит сигналы качества в единый
             confidence score в диапазоне [0, 1].
ИМПОРТИРУЕТ: numpy; scipy.stats.
ЭКСПОРТИРУЕТ: population_stability_index, ks_drift, forecast_errors,
             degradation_ratio, confidence_score.
КЕМ ИСПОЛЬЗУЕТСЯ: агент self_diagnostic.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import Dict                                  # аннотации

import numpy as np                                       # вычисления
from scipy import stats                                  # тест Колмогорова-Смирнова


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI between two samples (>0.2 typically signals material drift)."""
    reference = np.asarray(reference, float)             # эталон → массив
    reference = reference[np.isfinite(reference)]        # только конечные
    current = np.asarray(current, float)                 # текущее → массив
    current = current[np.isfinite(current)]              # только конечные
    if len(reference) < 2 or len(current) < 2:           # данных мало…
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)              # квантильные точки
    edges = np.unique(np.quantile(reference, quantiles))  # границы бинов по эталону
    if len(edges) < 2:                                   # вырожденные бины…
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=edges)    # гистограмма эталона
    cur_hist, _ = np.histogram(current, bins=edges)      # гистограмма текущего
    ref_pct = np.clip(ref_hist / ref_hist.sum(), 1e-6, None)  # доли эталона (защита от 0)
    cur_pct = np.clip(cur_hist / cur_hist.sum(), 1e-6, None)  # доли текущего
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))  # PSI


def ks_drift(reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov drift test."""
    reference = np.asarray(reference, float)             # эталон → массив
    current = np.asarray(current, float)                 # текущее → массив
    reference = reference[np.isfinite(reference)]        # только конечные
    current = current[np.isfinite(current)]              # только конечные
    if len(reference) < 2 or len(current) < 2:           # данных мало…
        return {"statistic": 0.0, "pvalue": 1.0}
    res = stats.ks_2samp(reference, current)             # двухвыборочный тест КС
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}  # статистика и p-value


def forecast_errors(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, float)                  # факт → массив
    predicted = np.asarray(predicted, float)            # прогноз → массив
    m = min(len(actual), len(predicted))                # общая длина
    a, p = actual[-m:], predicted[-m:]                  # выравниваем по хвосту
    err = a - p                                         # ошибки прогноза
    return {                                            # метрики ошибок:
        "rmse": float(np.sqrt(np.mean(err**2))) if m else 0.0,  # СКО
        "mae": float(np.mean(np.abs(err))) if m else 0.0,       # средняя абс. ошибка
        "bias": float(np.mean(err)) if m else 0.0,              # систематическое смещение
    }


def degradation_ratio(recent_error: float, baseline_error: float) -> float:
    if baseline_error <= 1e-12:                          # базовая ошибка ~0…
        return 1.0
    return float(recent_error / baseline_error)         # во сколько раз выросла ошибка


def confidence_score(components: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted blend of normalised quality components, each in ``[0, 1]``."""
    total_w = sum(weights.get(k, 0.0) for k in components)  # сумма весов
    if total_w <= 0:                                    # весов нет…
        return float(np.mean(list(components.values()))) if components else 0.0  # → простое среднее
    score = sum(weights.get(k, 0.0) * float(np.clip(v, 0.0, 1.0)) for k, v in components.items())  # взвеш. сумма
    return float(np.clip(score / total_w, 0.0, 1.0))    # нормированный балл в [0,1]
