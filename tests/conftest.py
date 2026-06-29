"""Shared pytest fixtures.

The integration fixture runs the full multi-agent pipeline exactly once per test
session in an isolated temporary root, using a small but representative
configuration so the suite stays fast while still exercising every agent.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        tests (инфраструктура тестирования) — общие фикстуры pytest.
НАЗНАЧЕНИЕ:  предоставить переиспользуемые фикстуры: «быстрый» конфиг,
             единый прогон всего пайплайна на сессию (в изолированной temp-папке)
             и детерминированный генератор случайных чисел. Это ускоряет набор
             тестов: пайплайн запускается один раз, а проверяют его много тестов.
ИМПОРТИРУЕТ: pathlib, numpy, pytest; внутри фикстур — cryptohedge.core/agents.
ЭКСПОРТИРУЕТ (фикстуры): fast_config, pipeline_run, rng; константы FAST_OVERRIDES, CONFIG_DIR.
КЕМ ИСПОЛЬЗУЕТСЯ: tests/test_integration.py, test_core.py, test_quant_services.py.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from pathlib import Path                                 # путь к каталогу конфигурации

import numpy as np                                       # детерминированный RNG для фикстуры
import pytest                                            # фреймворк тестирования

# Маленький, но полный конфиг: затрагивает все ветки кода, оставаясь быстрым.
FAST_OVERRIDES = {
    "seed": 12345,                                       # фиксированный seed (воспроизводимость тестов)
    "data": {"universe_size": 6},                        # маленькая вселенная (6 активов)
    "horizons": {"analysis_days": 24},                   # короткий горизонт анализа
    "market_analysis": {
        "top_n_hedge_instruments": 3,                    # меньше инструментов хеджа
        "regime_n_states": 3,                            # число режимов рынка
        "correlation": {"methods": ["pearson", "spearman", "kendall", "dcc_garch", "cointegration"]},  # все методы корреляции
    },
    "hedging": {"calibration_subsample": 4},             # подвыборка для калибровки (ускорение)
    "backtest": {"train_window": 10, "test_window": 4, "step": 4},  # маленькие окна walk-forward
    "runtime": {"resume": False, "checkpointing": True},  # без resume, но с чекпойнтами
}

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"  # путь к каталогу config/ проекта


@pytest.fixture(scope="session")
def fast_config():
    # фикстура: загруженный «быстрый» конфиг (один раз на сессию)
    from cryptohedge.core.config import load_config

    return load_config(CONFIG_DIR, overrides=FAST_OVERRIDES)


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory):
    """Run the whole pipeline once in an isolated root; reused by integration tests."""
    from cryptohedge.agents import build_pipeline        # сборка пайплайна
    from cryptohedge.core.config import load_config      # загрузка конфига
    from cryptohedge.core.context import AgentContext    # контекст прогона

    root = tmp_path_factory.mktemp("run")                # изолированный временный корень для артефактов
    config = load_config(CONFIG_DIR, overrides=FAST_OVERRIDES)  # быстрый конфиг
    context = AgentContext(config, root=root)            # контекст
    orchestrator = build_pipeline(context, fail_fast=True)  # оркестратор со всеми агентами
    report = orchestrator.run()                          # ОДИН прогон пайплайна на сессию
    return context, report, root                         # отдаём контекст, отчёт и путь


@pytest.fixture
def rng():
    # фикстура: детерминированный генератор случайных чисел (seed=0)
    return np.random.default_rng(0)
