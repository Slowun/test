"""In-process message bus and router.

The bus decouples senders from receivers. Agents register the message types they
consume; when a message is published the bus routes it to the matching agents and
records every hop in an auditable message log. This realises the "message routing
between agents" requirement without agents holding references to one another.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.core (каркас).
НАЗНАЧЕНИЕ:  внутрипроцессная ШИНА сообщений. Разрывает прямую связь
             «отправитель→получатель»: агенты подписываются на типы сообщений,
             а шина сама находит получателей и пишет полную трассу (для аудита).
ИМПОРТИРУЕТ:
  - collections.defaultdict : словарь подписок {тип сообщения: [имена агентов]}.
  - dataclasses             : @dataclass для RoutedMessage.
  - typing                  : аннотации Callable/DefaultDict/Dict/List.
  - core.agent.BaseAgent    : тип регистрируемого агента.
  - core.message.*          : Message и MessageType — то, что маршрутизируется.
ЭКСПОРТИРУЕТ:
  - RoutedMessage : запись трассы (сообщение + кто его обработал).
  - MessageBus    : сам маршрутизатор (register/publish/recipients/трасса).
КЕМ ИСПОЛЬЗУЕТСЯ:
  - core/orchestrator.py : создаёт шину, регистрирует агентов, публикует сообщения.
КОНСТАНТЫ:   своих нет.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from collections import defaultdict                       # словарь подписок с пустым списком по умолчанию
from dataclasses import dataclass, field                  # @dataclass + field для RoutedMessage
from typing import Callable, DefaultDict, Dict, List      # аннотации структур шины

from cryptohedge.core.agent import BaseAgent              # тип агента (для регистрации)
from cryptohedge.core.message import Message, MessageType # сообщения и их типы


@dataclass
class RoutedMessage:                                       # одна запись в трассе шины
    message: Message                                       # само опубликованное сообщение
    handled_by: List[str] = field(default_factory=list)    # имена агентов-получателей этого сообщения


class MessageBus:
    """Routes messages to subscribed agents and keeps a full message trace."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}            # реестр агентов по имени
        self._subscriptions: DefaultDict[MessageType, List[str]] = defaultdict(list)  # тип → подписчики
        self._listeners: List[Callable[[Message], None]] = []  # наблюдатели (вызываются на каждое сообщение)
        self.trace: List[RoutedMessage] = []               # полная трасса всех опубликованных сообщений

    def register(self, agent: BaseAgent) -> None:          # зарегистрировать агента в шине
        if agent.name in self._agents:                     # имя должно быть уникальным
            raise ValueError(f"Agent '{agent.name}' already registered")  # иначе — ошибка
        self._agents[agent.name] = agent                   # кладём агента в реестр
        for message_type in agent.consumes:                # для каждого типа, который агент потребляет…
            self._subscriptions[message_type].append(agent.name)  # …добавляем его в подписчики этого типа

    def add_listener(self, listener: Callable[[Message], None]) -> None:
        """Register an observer invoked for every published message (e.g. tracing)."""
        self._listeners.append(listener)                   # добавить наблюдателя (например, для логов)

    def agent(self, name: str) -> BaseAgent:               # получить агента по имени
        return self._agents[name]                          # вернуть зарегистрированный экземпляр

    def recipients(self, message: Message) -> List[str]:
        """Resolve the target agents for a message.

        Explicit ``recipient`` wins; otherwise subscriptions to the message type
        are used (topic routing).
        """
        if message.recipient and message.recipient in self._agents:  # если получатель указан явно и известен…
            return [message.recipient]                     # …маршрут — только ему
        return list(self._subscriptions.get(message.type, []))  # иначе — всем подписчикам этого типа

    def publish(self, message: Message) -> RoutedMessage:  # опубликовать сообщение в шину
        routed = RoutedMessage(message=message, handled_by=self.recipients(message))  # вычисляем получателей
        self.trace.append(routed)                          # пишем запись в трассу (для аудита)
        for listener in self._listeners:                   # уведомляем всех наблюдателей…
            listener(message)                              # …передавая им сообщение
        return routed                                      # возвращаем запись маршрутизации

    def registered_agents(self) -> List[str]:              # список имён всех зарегистрированных агентов
        return list(self._agents)                          # ключи реестра
