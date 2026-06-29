"""Unit tests for the critical computational services.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        tests (модульные тесты) — проверяет пакет cryptohedge.services.
НАЗНАЧЕНИЕ:  юнит-тесты ключевых вычислительных сервисов: волатильность и сайзинг
             хеджа, метрики (ROI/Sharpe/VaR/CVaR/MDD/beta), оптимизаторы портфеля
             и их ограничения, дрейф данных (PSI/confidence), walk-forward сплиты,
             ATR, статические корреляции и классификация связи.
ИМПОРТИРУЕТ: numpy, pandas, pytest; сервисы optimization/correlation/drift/metrics/
             stops/volatility/walkforward.
ПРОВЕРЯЕТ:   корректность численных свойств (знаки, монотонность, границы, инварианты).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import numpy as np                                       # генерация данных и числовые проверки
import pandas as pd                                      # данные для корреляций
import pytest                                            # фреймворк, approx, raises

from cryptohedge.services import optimization as opt     # оптимизаторы портфеля
from cryptohedge.services import correlation as corr     # корреляции/классификация связи
from cryptohedge.services import drift                   # дрейф данных, индекс доверия
from cryptohedge.services import metrics as mx           # метрики риска/доходности
from cryptohedge.services.stops import average_true_range  # ATR
from cryptohedge.services.volatility import estimate_volatility, log_returns, size_primary_hedge  # волатильность/сайзинг
from cryptohedge.services.walkforward import walk_forward_splits  # walk-forward сплиты


# ------------------------------------------------------------------- volatility
def test_log_returns_length():
    # log-доходности короче ряда цен на 1 и равны log(p_t/p_{t-1})
    p = np.array([100.0, 110.0, 121.0])
    r = log_returns(p)
    assert len(r) == 2
    assert np.allclose(r, np.log(p[1:] / p[:-1]))


def test_estimate_volatility_known_series():
    # на ряде с известной дневной волой ~2% оценка должна её восстанавливать
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.02, 500)
    prices = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rets])))
    vol = estimate_volatility(prices, window=30, trading_days=365)
    assert 0.015 < vol.daily_vol < 0.025          # восстановлена ~2% дневная вола
    assert vol.ci_low <= vol.daily_vol <= vol.ci_high  # оценка внутри доверительного интервала
    assert vol.annualized_vol == pytest.approx(vol.daily_vol * np.sqrt(365), rel=1e-6)  # годовая = дневная·√дней


def test_size_primary_hedge_monotonic():
    # меньший бюджет риска → больший коэффициент хеджа (монотонность)
    rng = np.random.default_rng(1)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.04, 200)))
    vol = estimate_volatility(prices)
    tight = size_primary_hedge(1e7, prices[-1], vol, risk_budget_pct=0.01)  # жёсткий бюджет
    loose = size_primary_hedge(1e7, prices[-1], vol, risk_budget_pct=0.04)  # мягкий бюджет
    assert 0.0 <= loose.hedge_ratio <= tight.hedge_ratio <= 1.0
    assert tight.quantity_to_hedge >= loose.quantity_to_hedge


# ----------------------------------------------------------------------- metrics
def test_metrics_basic_properties():
    # базовые инварианты метрик: ROI>-100%, вола>0, win_rate∈[0,1], MDD≤0, Sharpe конечен
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 365)
    m = mx.compute_metrics(r, periods_per_year=365)
    assert -1.0 < m.roi
    assert m.volatility > 0
    assert 0.0 <= m.win_rate <= 1.0
    assert m.max_drawdown <= 0.0
    assert np.isfinite(m.sharpe)


def test_max_drawdown_simple():
    # MDD на простой кривой капитала: провал 120→90 = -25%
    equity = np.array([100, 120, 90, 110.0])
    dd = mx.max_drawdown(equity)
    assert dd == pytest.approx((90 - 120) / 120)


def test_var_cvar_ordering():
    # CVaR (хвостовое среднее) не меньше VaR — фундаментальное неравенство
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.02, 5000)
    var = mx.value_at_risk(r, 0.95)
    cvar = mx.conditional_var(r, 0.95)
    assert cvar >= var > 0


def test_beta_against_self_is_one():
    # бета ряда относительно самого себя равна 1
    rng = np.random.default_rng(4)
    b = rng.normal(0, 0.01, 300)
    m = mx.compute_metrics(b.copy(), benchmark=b.copy(), periods_per_year=365)
    assert m.beta == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ optimization
def _spd_matrix(n, seed=0):
    # вспомогательная: сгенерировать симметричную положительно определённую матрицу (ковариация)
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T / n + np.eye(n) * 0.01            # A·Aᵀ + регуляризация диагонали


def test_optimizers_simplex_constraints():
    # все 5 оптимизаторов должны давать веса на симплексе (≥0, сумма=1) при long-only
    Sigma = _spd_matrix(5)
    mu = np.linspace(0.01, 0.05, 5)
    scen = np.random.default_rng(7).multivariate_normal(mu, Sigma, size=400)  # сценарии для CVaR
    for method in ["mean_variance", "min_variance", "risk_parity", "max_diversification", "cvar"]:
        w = opt.optimize(method, mu, Sigma, scenarios=scen, long_only=True, max_weight=1.0)
        assert w.shape == (5,)
        assert np.all(w >= -1e-6)                    # неотрицательность (long-only)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)  # полная инвестированность


def test_min_variance_beats_equal_weight():
    # портфель min-variance по построению имеет дисперсию не выше равновзвешенного
    Sigma = _spd_matrix(6, seed=11)
    w = opt.min_variance(Sigma)
    ew = np.ones(6) / 6
    assert w @ Sigma @ w <= ew @ Sigma @ ew + 1e-9


def test_transaction_cost_and_turnover():
    # оборот = сумма |Δвес|; издержки = оборот·ставка·капитал
    a = np.array([0.5, 0.5])
    b = np.array([0.2, 0.8])
    assert opt.turnover(a, b) == pytest.approx(0.6)
    assert opt.transaction_cost(a, b, 0.001, 1e6) == pytest.approx(0.6 * 0.001 * 1e6)


# ----------------------------------------------------------------------- drift
def test_psi_zero_for_identical():
    # PSI идентичных распределений ≈ 0 (нет дрейфа)
    rng = np.random.default_rng(5)
    x = rng.normal(size=1000)
    assert drift.population_stability_index(x, x.copy()) < 1e-6


def test_psi_detects_shift():
    # сдвиг среднего на 3σ должен давать большой PSI (>0.25 — существенный дрейф)
    rng = np.random.default_rng(6)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(3, 1, 2000)
    assert drift.population_stability_index(ref, cur) > 0.25


def test_confidence_score_weighting():
    # индекс доверия = взвешенное среднее компонент; результат обрезается в [0,1]
    comps = {"a": 1.0, "b": 0.0}
    w = {"a": 1.0, "b": 1.0}
    assert drift.confidence_score(comps, w) == pytest.approx(0.5)
    assert 0.0 <= drift.confidence_score({"a": 5.0}, {"a": 1.0}) <= 1.0  # значение >1 обрезается


# ------------------------------------------------------------------- walkforward
def test_walk_forward_no_leakage():
    # сплиты walk-forward не допускают заглядывания вперёд и соблюдают purge-зазор
    folds = walk_forward_splits(n=40, train_window=10, test_window=5, step=5, purge=1)
    assert len(folds) > 0
    for f in folds:
        assert f.train.max() < f.test.min()           # train строго раньше test
        assert f.test.min() - f.train.max() >= 1       # соблюдён purge-зазор
        assert len(f.test) == 5


def test_walk_forward_expanding_grows():
    # в расширяющемся режиме train-окно не убывает от фолда к фолду
    folds = walk_forward_splits(n=50, train_window=10, test_window=5, step=5, expanding=True)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes) and sizes[0] <= sizes[-1]


# -------------------------------------------------------------------------- ATR
def test_atr_positive_and_padded():
    # ATR: первые (window-1) значений = NaN (warm-up), далее строго положителен
    rng = np.random.default_rng(8)
    close = 100 + np.cumsum(rng.normal(0, 1, 100))
    high = close + np.abs(rng.normal(0, 1, 100))
    low = close - np.abs(rng.normal(0, 1, 100))
    atr = average_true_range(high, low, close, window=14)
    assert len(atr) == 100                            # длина сохранена
    assert np.isnan(atr[:13]).all()                   # warm-up период = NaN
    assert np.all(atr[14:] > 0)                       # далее положителен


# ------------------------------------------------------------------ correlation
def test_static_correlations_self_excluded():
    # статические корреляции к первичному символу: сам он исключён, знаки связи верны
    rng = np.random.default_rng(9)
    base = rng.normal(0, 1, 200)
    df = pd.DataFrame({
        "BTCUSDT": base,
        "POS": base * 0.9 + rng.normal(0, 0.1, 200),   # сильно положительно связан
        "NEG": -base * 0.8 + rng.normal(0, 0.1, 200),  # сильно отрицательно связан
    })
    res = corr.static_correlations(df, "BTCUSDT")
    assert "BTCUSDT" not in res.index                 # первичный символ исключён
    assert res.loc["POS", "pearson"] > 0.7            # положительная корреляция
    assert res.loc["NEG", "pearson"] < -0.6           # отрицательная корреляция


def test_classify_relationship():
    # классификация связи по корреляции: положительная/обратная/нейтральная
    assert corr.classify_relationship(0.8, 0.5, -0.3, 0.1) == "positive"
    assert corr.classify_relationship(-0.5, 0.5, -0.3, 0.1) == "inverse"
    assert corr.classify_relationship(0.02, 0.5, -0.3, 0.1) == "neutral"
