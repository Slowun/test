"""Message protocol used for inter-agent communication.

Agents never call each other directly; they exchange immutable :class:`Message`
objects through the :class:`cryptohedge.core.bus.MessageBus`. Each message carries
a type, a sender, a recipient, a free-form payload and a correlation id linking a
request to its response, which makes routing and tracing explicit.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.core (каркас).
НАЗНАЧЕНИЕ:  определяет «язык» общения агентов — неизменяемое сообщение Message
             и перечисление его типов MessageType. Агенты не вызывают друг друга
             напрямую: они шлют Message через шину (bus.py), а оркестратор
             (orchestrator.py) гоняет эти сообщения по пайплайну.
ИМПОРТИРУЕТ:
  - uuid                : генерация уникальных идентификаторов сообщения/корреляции.
  - dataclasses        : декоратор @dataclass(frozen=True) для неизменяемой структуры.
  - datetime/timezone  : временна́я метка создания сообщения (UTC).
  - enum.Enum          : базовый класс для перечисления типов сообщений.
  - typing             : аннотации Any/Dict/Optional.
ЭКСПОРТИРУЕТ:
  - MessageType (Enum) : семантические типы сообщений (DATA_READY, HEDGE_DECISION…).
  - Message (dataclass): сам объект сообщения + методы reply()/describe().
КЕМ ИСПОЛЬЗУЕТСЯ:
  - core/bus.py, core/agent.py, core/orchestrator.py и ВСЕ агенты (agents/*).
КОНСТАНТЫ:   строковые значения членов MessageType — это «контракт» маршрутизации.
=============================================================================
"""

from __future__ import annotations            # отложенные аннотации типов

import uuid                                    # уникальные id (message_id, correlation_id)
from dataclasses import dataclass, field       # @dataclass + field(default_factory=...)
from datetime import datetime, timezone        # метка времени создания сообщения в UTC
from enum import Enum                          # перечисление типов сообщений
from typing import Any, Dict, Optional         # аннотации произвольного payload


class MessageType(str, Enum):
    """Semantic type of a message used by the router to dispatch work."""
    # Наследуемся от str и Enum: значение можно сравнивать как строку и сериализовать в JSON.

    # control                                  # --- управляющие сообщения пайплайна ---
    START = "start"                            # старт пайплайна (отправляет оркестратор)
    COMPLETED = "completed"                    # этап успешно завершён
    FAILED = "failed"                          # этап упал с ошибкой
    # data / analysis flow                     # --- доменный поток данных/анализа ---
    DATA_READY = "data_ready"                  # выпускает агент 1 (DataAcquisition)
    ANALYSIS_READY = "analysis_ready"          # выпускает агент 2 (MarketAnalysis)
    CALIBRATION_READY = "calibration_ready"    # выпускает агент 3 (HestonCalibration)
    GREEKS_READY = "greeks_ready"              # выпускает агент 4 (GreeksCalculation)
    HEDGE_DECISION = "hedge_decision"          # выпускает агент 5 (HedgingDecision)
    PORTFOLIO_READY = "portfolio_ready"        # выпускает агент 6 (PortfolioOptimization)
    RISK_ASSESSMENT = "risk_assessment"        # выпускает агент 7 (RiskManagement)
    BACKTEST_READY = "backtest_ready"          # выпускает агент 8 (Backtesting)
    DIAGNOSTIC_READY = "diagnostic_ready"      # выпускает агент 9 (SelfDiagnostic)
    EXPLANATION_READY = "explanation_ready"    # выпускает агент 10 (Explainability)
    DASHBOARD_READY = "dashboard_ready"        # выпускает агент 11 (Dashboard)
    # generic request/response                 # --- обобщённые запрос/ответ ---
    REQUEST = "request"                        # произвольный запрос
    RESPONSE = "response"                      # произвольный ответ (тип по умолчанию)


@dataclass(frozen=True)                        # frozen=True => экземпляры неизменяемы (immutable)
class Message:
    """An immutable unit of communication between agents."""

    type: MessageType                          # семантический тип (см. MessageType выше)
    sender: str                                # имя агента-отправителя
    recipient: str                             # имя агента-получателя ("" => маршрут по типу)
    payload: Dict[str, Any] = field(default_factory=dict)        # полезная нагрузка (произвольные данные)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)  # связывает запрос и ответ
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)      # уникальный id самого сообщения
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # момент создания (UTC)

    def reply(
        self,
        sender: str,                           # кто отвечает (имя агента)
        type: MessageType,                     # тип ответного сообщения
        payload: Optional[Dict[str, Any]] = None,   # данные ответа (по умолчанию пусто)
        recipient: Optional[str] = None,       # кому ответ (по умолчанию — исходному отправителю)
    ) -> "Message":
        """Create a response message preserving the correlation id."""
        return Message(                        # строим НОВОЕ сообщение (исходное неизменяемо)
            type=type,                         # тип ответа
            sender=sender,                     # отправитель ответа
            recipient=recipient or self.sender,  # получатель = заданный или исходный sender
            payload=payload or {},             # данные ответа или пустой словарь
            correlation_id=self.correlation_id,  # СОХРАНЯЕМ correlation_id => связь запрос↔ответ
        )

    def describe(self) -> str:
        # Краткое человекочитаемое представление для логов/трассировки шины.
        return f"{self.sender} -> {self.recipient} [{self.type.value}] keys={list(self.payload)}"
