"""Unit tests for the core framework: config, seeding, checkpoint, messaging.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        tests (модульные тесты) — проверяет пакет cryptohedge.core.
НАЗНАЧЕНИЕ:  юнит-тесты ядра: загрузка/валидация конфига, воспроизводимость
             сидинга, чекпойнты (roundtrip + manifest), сообщения (immutability,
             correlation_id), шина и оркестратор (маршрутизация, resume).
ИМПОРТИРУЕТ: numpy, pandas, pytest; компоненты cryptohedge.core; CONFIG_DIR из conftest.
ПРОВЕРЯЕТ:   config.py, seeding.py, checkpoint.py, message.py, bus.py, orchestrator.py.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import numpy as np                                       # проверки RNG/массивов
import pandas as pd                                      # проверка roundtrip DataFrame
import pytest                                            # фреймворк и assert-исключения

from cryptohedge.core.agent import BaseAgent             # базовый класс агента (для тест-агентов)
from cryptohedge.core.bus import MessageBus              # шина сообщений
from cryptohedge.core.checkpoint import CheckpointManager  # менеджер чекпойнтов
from cryptohedge.core.config import load_config          # загрузка конфига
from cryptohedge.core.context import AgentContext        # контекст прогона
from cryptohedge.core.message import Message, MessageType  # сообщения и их типы
from cryptohedge.core.orchestrator import Orchestrator   # оркестратор
from cryptohedge.core.seeding import set_global_seed, spawn_rng  # сидинг и независимые RNG-потоки

from conftest import CONFIG_DIR                           # путь к каталогу config/


# -------------------------------------------------------------------- config
def test_config_loads_and_validates():
    # конфиг должен корректно загружаться и проходить валидацию (ключевые поля)
    cfg = load_config(CONFIG_DIR)
    assert cfg.investment.capital_usd == 10_000_000.0    # капитал по умолчанию
    assert cfg.data.universe_size == 100                 # размер вселенной
    assert len(cfg.optimization.methods) == 5            # 5 методов оптимизации
    assert set(["pearson", "spearman", "kendall", "dcc_garch", "cointegration"]).issubset(
        set(cfg.market_analysis.correlation.methods))    # все методы корреляции присутствуют


def test_config_override_applies():
    # программные override должны применяться поверх YAML
    cfg = load_config(CONFIG_DIR, overrides={"data": {"universe_size": 7}})
    assert cfg.data.universe_size == 7


def test_config_rejects_unknown_keys():
    # неизвестные ключи конфига должны приводить к ошибке валидации
    with pytest.raises(Exception):
        load_config(CONFIG_DIR, overrides={"investment": {"nonexistent_param": 1}})


# -------------------------------------------------------------------- seeding
def test_seeding_is_reproducible():
    # один seed → одинаковые последовательности глобального RNG
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_spawn_rng_streams_independent_but_deterministic():
    # spawn_rng: детерминирован по (seed, stream), но потоки независимы
    r1 = spawn_rng(7, 1).random(4)
    r1b = spawn_rng(7, 1).random(4)
    r2 = spawn_rng(7, 2).random(4)
    assert np.allclose(r1, r1b)        # одинаковый поток — воспроизводимо
    assert not np.allclose(r1, r2)     # разные потоки — независимы


# ------------------------------------------------------------------ checkpoint
def test_checkpoint_roundtrip(tmp_path):
    # сохранение и загрузка артефактов разных типов должны давать исходные данные
    cm = CheckpointManager(tmp_path, run_id="t", enabled=True)
    df = pd.DataFrame({"a": [1, 2, 3]})
    cm.save("frame", df)                                  # DataFrame → parquet
    cm.save("dict", {"x": 1, "y": [1, 2]})               # dict → json
    cm.save("obj", {"n": np.int64(3)})  # несериализуемое json → fallback на default=str
    assert cm.exists("frame") and cm.exists("dict")
    pd.testing.assert_frame_equal(cm.load("frame"), df)  # DataFrame восстановлен точно
    assert cm.load("dict")["y"] == [1, 2]


def test_checkpoint_manifest_tracks_stages(tmp_path):
    # манифест должен отмечать завершённые стадии и переживать перезагрузку с диска
    cm = CheckpointManager(tmp_path, run_id="t", enabled=True)
    assert not cm.is_completed("stage1")
    cm.mark_completed("stage1", {"k": 1})
    assert cm.is_completed("stage1")
    cm2 = CheckpointManager(tmp_path, run_id="t", enabled=True)  # новый менеджер читает манифест с диска
    assert cm2.is_completed("stage1")


# --------------------------------------------------------------------- message
def test_message_reply_preserves_correlation():
    # reply() должен сохранять correlation_id и менять отправителя/получателя местами
    m = Message(MessageType.DATA_READY, "a", "b", {"k": 1})
    r = m.reply("b", MessageType.ANALYSIS_READY, {"v": 2})
    assert r.correlation_id == m.correlation_id          # цепочка трассировки сохранена
    assert r.recipient == "a" and r.sender == "b"        # ответ направлен исходному отправителю


def test_message_is_immutable():
    # сообщения неизменяемы (frozen) — попытка записи поля даёт ошибку
    m = Message(MessageType.START, "a", "b")
    with pytest.raises(Exception):
        m.sender = "c"  # type: ignore[misc]


# -------------------------------------------------------------- bus + orchestrator
class _ProducerAgent(BaseAgent):                          # тест-агент: кладёт значение и шлёт DATA_READY
    name = "producer"                                    # имя агента
    consumes = [MessageType.START]                       # реагирует на START
    produces = MessageType.DATA_READY                    # производит DATA_READY
    checkpoint_keys = ["produced_value"]                 # ключи для чекпойнта

    def execute(self, context, message):
        context.put("produced_value", 123)              # пишем артефакт на blackboard
        return Message(self.produces, self.name, "consumer", {"value": 123},
                       correlation_id=message.correlation_id)  # сообщение потребителю


class _ConsumerAgent(BaseAgent):                          # тест-агент: удваивает значение продюсера
    name = "consumer"                                    # имя агента
    consumes = [MessageType.DATA_READY]                  # реагирует на DATA_READY
    produces = MessageType.COMPLETED                     # производит COMPLETED
    checkpoint_keys = ["consumed_value"]                 # ключи для чекпойнта

    def execute(self, context, message):
        context.put("consumed_value", message.payload["value"] * 2)  # 123*2 = 246
        return Message(self.produces, self.name, "orchestrator",
                       {"value": context.get("consumed_value")},
                       correlation_id=message.correlation_id)


def test_bus_routes_by_subscription():
    # шина маршрутизирует сообщение только подписанным на его тип агентам
    bus = MessageBus()
    bus.register(_ProducerAgent())
    bus.register(_ConsumerAgent())
    msg = Message(MessageType.DATA_READY, "x", "")
    assert bus.recipients(msg) == ["consumer"]           # DATA_READY → только consumer


def test_orchestrator_runs_two_agents(tmp_path):
    # оркестратор должен выполнить два связанных агента и протрассировать сообщения
    cfg = load_config(CONFIG_DIR, overrides={"runtime": {"resume": False}})
    ctx = AgentContext(cfg, root=tmp_path)
    orch = Orchestrator(ctx, fail_fast=True)
    orch.register(_ProducerAgent()).register(_ConsumerAgent())
    report = orch.run()
    assert report.success                                # прогон успешен
    assert ctx.get("consumed_value") == 246              # данные прошли через цепочку
    # лог сообщений сохранён (маршрутизация аудируема)
    assert len(orch.bus.trace) > 0


def test_orchestrator_checkpoint_resume(tmp_path):
    """Second run with resume=True should skip the already-completed stages."""
    base = {"runtime": {"resume": False}}
    ctx = AgentContext(load_config(CONFIG_DIR, overrides=base), root=tmp_path)
    Orchestrator(ctx).register(_ProducerAgent()).register(_ConsumerAgent()).run()  # первый прогон (создаёт чекпойнты)

    ctx2 = AgentContext(load_config(CONFIG_DIR, overrides={"runtime": {"resume": True}}), root=tmp_path)  # resume=True
    report2 = Orchestrator(ctx2).register(_ProducerAgent()).register(_ConsumerAgent()).run()  # второй прогон
    assert report2.success
    assert all(r.skipped for r in report2.results)        # все стадии восстановлены из чекпойнта
    assert ctx2.get("consumed_value") == 246              # состояние восстановлено
