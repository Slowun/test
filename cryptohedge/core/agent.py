"""The unified agent contract.

Every agent is an independent module exposing the same interface: it ``consumes``
one or more :class:`MessageType` values, ``produces`` exactly one, and implements
:meth:`BaseAgent.execute`. The :meth:`BaseAgent.run` template method wraps every
execution with structured logging, timing, error capture and checkpointing, so
agents only contain domain logic (Single Responsibility / Open-Closed).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.core (каркас) — БАЗОВЫЙ КЛАСС всех агентов.
НАЗНАЧЕНИЕ:  задаёт ЕДИНЫЙ контракт агента и «обвес» вокруг его логики.
             Шаблонный метод run() делает за всех: логирование, тайминги,
             перехват ошибок, чекпойнт/восстановление. Конкретный агент пишет
             только execute() — чистую доменную логику. Это ключевой файл: поняв
             его, понимаешь устройство ВСЕХ 11 агентов.
ИМПОРТИРУЕТ:
  - time, traceback : измерение длительности и текст трассировки исключения.
  - abc             : ABC/abstractmethod — абстрактный базовый класс.
  - dataclasses     : @dataclass для AgentResult.
  - typing          : аннотации Any/Dict/List.
  - core.context    : AgentContext (доска, логгеры, чекпойнты).
  - core.message    : Message/MessageType (вход и выход агента).
ЭКСПОРТИРУЕТ:
  - AgentResult : результат запуска агента (успех/ошибка/тайминг/сообщение).
  - BaseAgent   : абстрактный базовый класс (наследуют все agents/*).
КЕМ ИСПОЛЬЗУЕТСЯ:
  - всеми agents/*; orchestrator.py вызывает agent.run(); bus.py регистрирует.
КОНСТАНТЫ (атрибуты класса по умолчанию):
  - name="base", consumes=[], produces=RESPONSE, checkpoint_keys=[]
    (каждый агент переопределяет их под себя).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import time                                              # измерение длительности execute()
import traceback                                         # формирование текста трассировки при сбое
from abc import ABC, abstractmethod                      # абстрактный класс и абстрактный метод
from dataclasses import dataclass, field                 # @dataclass + field для AgentResult
from typing import Any, Dict, List                       # аннотации полей и контракта

from cryptohedge.core.context import AgentContext        # контекст (доска/логгеры/чекпойнты)
from cryptohedge.core.message import Message, MessageType # вход/выход агента — сообщения


@dataclass
class AgentResult:
    """Outcome of an agent execution."""

    agent: str                                           # имя агента, к которому относится результат
    message: Message                                     # выпущенное агентом сообщение (выход)
    success: bool = True                                 # успешно ли отработал
    error: str = ""                                      # текст ошибки (если success=False)
    duration_s: float = 0.0                              # длительность работы в секундах
    skipped: bool = False                                # был ли этап пропущен (восстановлен из чекпойнта)
    artifacts: Dict[str, Any] = field(default_factory=dict)  # доп. артефакты результата (если нужны)


class BaseAgent(ABC):
    """Abstract base class defining the unified agent interface."""

    #: Human-readable, unique agent name (also used as the checkpoint stage id).
    name: str = "base"                                   # уникальное имя агента (и id этапа чекпойнта)
    #: Message types this agent is able to handle.
    consumes: List[MessageType] = []                     # типы сообщений, которые агент принимает
    #: Message type emitted on success.
    produces: MessageType = MessageType.RESPONSE         # тип сообщения, выпускаемого при успехе
    #: Blackboard keys this agent writes (persisted for checkpoint/resume).
    checkpoint_keys: List[str] = []                      # ключи доски, которые сохранять в чекпойнт

    def __init__(self) -> None:
        self._validate_contract()                        # при создании проверяем корректность контракта

    def _validate_contract(self) -> None:
        if not self.name or self.name == "base":         # имя обязано быть переопределено и уникально
            raise ValueError(f"{type(self).__name__} must define a unique 'name'")
        if not self.consumes:                            # агент обязан объявить, что он потребляет
            raise ValueError(f"Agent '{self.name}' must declare the messages it consumes")

    # ------------------------------------------------------------------ contract
    @abstractmethod                                      # абстрактный: КАЖДЫЙ агент обязан реализовать
    def execute(self, context: AgentContext, message: Message) -> Message:
        """Perform the agent's work and return the output message.

        Implementations read inputs from ``context``/``message``, write their
        artifacts onto ``context`` (blackboard) and return a message describing
        the result. They must not catch their own fatal errors: the template
        method :meth:`run` handles logging and recovery uniformly.
        """

    def can_handle(self, message: Message) -> bool:
        return message.type in self.consumes             # умеет ли агент обработать данное сообщение

    # ------------------------------------------------------------- template method
    def run(self, context: AgentContext, message: Message) -> AgentResult:
        log = context.logger(self.name)                  # получаем логгер для этого агента
        resume = context.config.runtime.resume           # включён ли режим возобновления (resume)

        # Если resume включён, этап уже завершён ранее и его артефакты на диске —
        # пропускаем выполнение и восстанавливаем результат из чекпойнта.
        if resume and context.checkpoints.is_completed(self.name) and self._can_restore(context):
            self._restore(context)                       # подгружаем сохранённые ключи на доску
            log.info("restored from checkpoint", stage=self.name)  # логируем восстановление
            out = Message(self.produces, self.name, "orchestrator", {"restored": True},  # формируем выход
                          correlation_id=message.correlation_id)   # сохраняем correlation_id
            return AgentResult(self.name, out, success=True, skipped=True)  # помечаем как пропущенный

        log.info("started", consumes=message.type.value)  # логируем старт (что приняли на вход)
        start = time.perf_counter()                       # засекаем время начала
        try:
            with log.timer(f"{self.name}.execute"):       # измеряем длительность самой работы агента
                out_message = self.execute(context, message)  # ВЫЗОВ доменной логики агента
            duration = time.perf_counter() - start        # полная длительность с учётом обвеса
            self._persist(context)                        # сохраняем checkpoint_keys на диск
            context.checkpoints.mark_completed(self.name, {"duration_s": round(duration, 3)})  # отмечаем этап
            log.info("completed", duration_s=round(duration, 3), produces=out_message.type.value)  # лог успеха
            return AgentResult(self.name, out_message, success=True, duration_s=duration)  # успешный результат
        except Exception as exc:  # noqa: BLE001 - top-level agent guard   # верхнеуровневый перехват ошибок
            duration = time.perf_counter() - start        # длительность до момента сбоя
            tb = traceback.format_exc()                   # полный текст трассировки
            log.error("failed", error=str(exc), traceback=tb, duration_s=round(duration, 3))  # лог ошибки
            fail = Message(                               # формируем сообщение об ошибке для оркестратора
                MessageType.FAILED,                       #   тип FAILED
                self.name,                                #   отправитель — этот агент
                "orchestrator",                           #   получатель — оркестратор
                {"error": str(exc), "agent": self.name},  #   полезная нагрузка с описанием ошибки
                correlation_id=message.correlation_id,    #   сохраняем correlation_id
            )
            return AgentResult(self.name, fail, success=False, error=str(exc), duration_s=duration)  # провал

    # ----------------------------------------------------------------- checkpoint
    def _can_restore(self, context: AgentContext) -> bool:
        # Восстановление возможно, только если ВСЕ checkpoint_keys присутствуют на диске.
        return all(context.checkpoints.exists(self._ckpt_key(k)) for k in self.checkpoint_keys)

    def _restore(self, context: AgentContext) -> None:
        for key in self.checkpoint_keys:                  # для каждого сохраняемого ключа…
            context.put(key, context.checkpoints.load(self._ckpt_key(key)))  # …загрузить значение на доску

    def _persist(self, context: AgentContext) -> None:
        for key in self.checkpoint_keys:                  # для каждого сохраняемого ключа…
            if context.has(key):                          # …если он есть на доске…
                context.checkpoints.save(self._ckpt_key(key), context.get(key))  # …сохранить на диск

    def _ckpt_key(self, blackboard_key: str) -> str:
        return f"{self.name}__{blackboard_key}"           # имя файла чекпойнта = "<агент>__<ключ>"
