"""Integration tests: the eleven agents interacting through the orchestrator.

These run the full pipeline once (session fixture) and assert that every stage
succeeds, the expected artifacts are produced and exchanged across the blackboard,
and that the run is reproducible.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        tests (интеграционные тесты) — проверяет систему целиком (end-to-end).
НАЗНАЧЕНИЕ:  на одном прогоне всего пайплайна (фикстура pipeline_run на сессию)
             проверяет: все 11 агентов отработали по порядку; на blackboard есть
             все артефакты; сообщения трассируются; файлы результатов записаны;
             греки конечны; дельта захеджирована; стресс-таблица осмысленна;
             индекс доверия в [0,1]; портфель диверсифицирован/прибылен; вывод
             двуязычный; прогон воспроизводим при том же seed.
ИМПОРТИРУЕТ: pathlib, numpy, pandas, pytest; build_pipeline/конфиг/контекст;
             CONFIG_DIR и FAST_OVERRIDES из conftest.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from pathlib import Path                                 # проверка файлов-артефактов

import numpy as np                                       # числовые проверки
import pandas as pd                                      # работа с таблицами артефактов
import pytest                                            # фреймворк, approx, маркеры

from cryptohedge.agents import build_pipeline            # сборка пайплайна (для повторного прогона)
from cryptohedge.core.config import load_config          # загрузка конфига
from cryptohedge.core.context import AgentContext        # контекст прогона

from conftest import CONFIG_DIR, FAST_OVERRIDES          # путь конфига и быстрые override


EXPECTED_AGENTS = [                                       # эталонный порядок выполнения 11 агентов
    "data_acquisition", "market_analysis", "heston_calibration", "greeks_calculation",
    "hedging_decision", "portfolio_optimization", "risk_management", "backtesting",
    "self_diagnostic", "explainability", "dashboard",
]


def test_all_agents_run_successfully(pipeline_run):
    # все 11 агентов должны отработать успешно и в правильном порядке
    _, report, _ = pipeline_run
    assert report.success, f"failed stage: {report.failed_stage}"
    assert [r.agent for r in report.results] == EXPECTED_AGENTS
    assert all(r.success for r in report.results)


def test_blackboard_artifacts_present(pipeline_run):
    # на общей доске (blackboard) должны присутствовать все ключевые артефакты
    ctx, _, _ = pipeline_run
    for key in ["market_data", "spot_close", "returns", "volatility", "hedge_sizing",
                "rankings_df", "hedge_universe", "calibr_data", "heston_history",
                "portfolio_greeks_latest", "hedge_history", "opt_weights",
                "risk_assessment", "backtest_metrics", "stress_table",
                "confidence_score", "explanation_sections", "dashboard_path"]:
        assert ctx.has(key), f"missing blackboard artifact: {key}"


def test_message_routing_trace(pipeline_run):
    ctx, report, _ = pipeline_run
    # каждый агент произвёл сообщение для следующей стадии — проверяем края цепочки
    produced = [r.message.type.value for r in report.results]
    assert "data_ready" in produced                      # начало цепочки (данные готовы)
    assert "dashboard_ready" in produced                 # конец цепочки (дашборд готов)


def test_result_files_written(pipeline_run):
    # ключевые файлы результатов должны быть записаны на диск
    _, _, root = pipeline_run
    results = Path(root) / "artifacts" / "results"
    for fname in ["performance_metrics.json", "hedging_history.parquet",
                  "stress_test.parquet", "explanation.md", "dashboard.html"]:
        assert (results / fname).exists(), f"missing result file: {fname}"
    # артефакты калибровки также сохранены
    calib = Path(root) / "artifacts" / "calibration" / "calibr_data.parquet"
    assert calib.exists()


def test_greeks_are_finite(pipeline_run):
    # все греки портфеля должны быть конечными числами
    ctx, _, _ = pipeline_run
    g = ctx.get("portfolio_greeks_latest")
    for key in ["delta", "gamma", "vega", "theta", "rho"]:
        assert key in g and np.isfinite(g[key])


def test_hedge_neutralises_delta(pipeline_run):
    # дельта-хедж: остаточная дельта на каждом шаге ≈ 0
    ctx, _, _ = pipeline_run
    hist = ctx.get("hedge_history")
    residual = (hist["delta"] - hist["delta_hedge"]).abs()
    assert residual.max() < 1e-6        # дельта полностью захеджирована на каждом шаге


def test_stress_table_decomposition(pipeline_run):
    # стресс-таблица: нужные колонки есть; хедж лучше «голой» позиции в худшем сценарии
    ctx, _, _ = pipeline_run
    stress = ctx.get("stress_table")
    for col in ["scenario", "net_hedged_pnl", "unhedged_pnl", "hedge_effectiveness"]:
        assert col in stress.columns
    # захеджированная книга существенно безопаснее голой экспозиции при худшем шоке
    worst = stress.loc[stress["unhedged_pnl"].idxmin()]
    assert abs(worst["net_hedged_pnl"]) < abs(worst["unhedged_pnl"])


def test_confidence_score_in_unit_interval(pipeline_run):
    # индекс доверия должен лежать в [0,1]
    ctx, _, _ = pipeline_run
    cs = float(ctx.get("confidence_score"))
    assert 0.0 <= cs <= 1.0


def test_portfolio_constituents_and_diversification(pipeline_run):
    ctx, _, _ = pipeline_run
    constituents = ctx.get("portfolio_constituents")
    div = ctx.get("diversification")
    # портфель — реальная корзина инструментов с корректными весами
    assert constituents is not None and not constituents.empty
    assert {"symbol", "weight", "exp_return_annual", "vol_annual", "relationship"} <= set(constituents.columns)
    assert constituents["weight"].sum() == pytest.approx(1.0, abs=1e-6)  # веса суммируются в 1
    # диверсификация измерима и осмысленна
    for key in ["diversification_ratio", "effective_n", "max_weight", "hhi", "n_assets"]:
        assert key in div
    assert div["diversification_ratio"] >= 1.0 - 1e-9       # DR никогда не меньше 1
    assert 1.0 <= div["effective_n"] <= div["n_assets"] + 1e-9
    assert div["effective_n"] > 1.0                          # действительно диверсифицирован, не одна ставка


def test_portfolio_is_profitable_and_rebalanced(pipeline_run):
    ctx, _, _ = pipeline_run
    methods = ctx.get("method_comparison")
    reb = ctx.get("rebalance_decision")
    equity = ctx.get("portfolio_equity")
    chosen = methods.loc[methods["chosen"]].iloc[0]      # выбранный оптимизатор
    assert chosen["method"] == reb["method"]             # согласованность выбора метода
    assert chosen["total_return"] > 0.0                  # выбранный портфель прибылен
    # построены кривая капитала и траектория ребалансировок
    assert equity is not None and len(equity) > 1
    assert float(equity["equity"].iloc[-1]) > float(equity["equity"].iloc[0])  # капитал вырос
    assert len(ctx.get("portfolio_rebalances")) >= 1     # была хотя бы одна ребалансировка


def test_bilingual_outputs_written(pipeline_run):
    ctx, _, root = pipeline_run
    results = Path(root) / "artifacts" / "results"
    for fname in ["dashboard_ru.html", "dashboard_en.html", "explanation.md", "explanation.en.md"]:
        assert (results / fname).exists(), f"missing localized file: {fname}"  # оба языка записаны
    paths = ctx.get("dashboard_paths")
    assert set(paths) == {"ru", "en"}                    # доступны обе локали
    # у каждого языка свои непустые секции
    assert ctx.get("explanation_sections_ru") and ctx.get("explanation_sections_en")
    ru_html = (results / "dashboard_ru.html").read_text(encoding="utf-8")
    en_html = (results / "dashboard_en.html").read_text(encoding="utf-8")
    assert "Состав портфеля" in ru_html and "Portfolio Constituents" in en_html  # верный язык в каждом файле
    assert "Portfolio Constituents" not in ru_html and "Состав портфеля" not in en_html  # без смешения языков


@pytest.mark.slow
def test_pipeline_reproducible(pipeline_run, tmp_path):
    """A second independent run with the same seed reproduces key numbers."""
    ctx1, _, _ = pipeline_run                            # первый прогон (из фикстуры)
    cfg = load_config(CONFIG_DIR, overrides=FAST_OVERRIDES)
    ctx2 = AgentContext(cfg, root=tmp_path)
    build_pipeline(ctx2, fail_fast=True).run()           # второй независимый прогон с тем же seed

    h1 = ctx1.get("hedge_history")["pnl"].to_numpy()
    h2 = ctx2.get("hedge_history")["pnl"].to_numpy()
    assert np.allclose(h1, h2, rtol=1e-8, atol=1e-8)     # PnL воспроизводится бит-в-бит (в пределах допуска)

    s1 = ctx1.get("hedge_sizing").quantity_to_hedge
    s2 = ctx2.get("hedge_sizing").quantity_to_hedge
    assert s1 == pytest.approx(s2, rel=1e-10)            # размер хеджа воспроизводим
