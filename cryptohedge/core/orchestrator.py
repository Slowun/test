"""Pipeline orchestrator.

The orchestrator wires the registered agents into a directed flow and drives the
message routing between them. The canonical hedging pipeline is mostly linear
(data -> analysis -> calibration -> greeks -> hedging -> optimization -> risk ->
backtest -> diagnostics -> explanation -> dashboard), but the orchestrator routes
each produced message through the :class:`MessageBus`, supports per-stage
checkpoint/resume and fails fast (or continues) according to configuration.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.core (каркас).
НАЗНАЧЕНИЕ:  ОРКЕСТРАТОР — соединяет агентов в этапы (PipelineStage) и гоняет
             сообщение по цепочке: выход предыдущего агента адаптируется во вход
             следующего. Публикует всё через шину, поддерживает fail-fast и
             собирает отчёт о прогоне (RunReport: успех/ошибки/тайминги).
ИМПОРТИРУЕТ:
  - dataclasses     : @dataclass для PipelineStage и RunReport.
  - typing          : аннотации Dict/List/Optional.
  - core.agent      : AgentResult/BaseAgent.
  - core.bus        : MessageBus (маршрутизация).
  - core.context    : AgentContext (прокидывается в каждый agent.run()).
  - core.message    : Message/MessageType.
ЭКСПОРТИРУЕТ:
  - PipelineStage : этап = (имя агента, ожидаемый тип сообщения).
  - RunReport     : отчёт о прогоне (результаты, упавший этап, суммарное время).
  - Orchestrator  : register()/run() — сборка и запуск пайплайна.
КЕМ ИСПОЛЬЗУЕТСЯ:
  - agents/__init__.py::build_pipeline() создаёт и наполняет Orchestrator;
  - cli.py::run_pipeline() и solution.ipynb запускают orchestrator.run().
КОНСТАНТЫ:   своих нет.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from dataclasses import dataclass, field                 # @dataclass + field для отчётов
from typing import Dict, List, Optional                  # аннотации структур

from cryptohedge.core.agent import AgentResult, BaseAgent  # результат и базовый класс агента
from cryptohedge.core.bus import MessageBus              # шина для публикации сообщений
from cryptohedge.core.context import AgentContext        # контекст прогона
from cryptohedge.core.message import Message, MessageType  # сообщения и их типы


@dataclass
class PipelineStage:
    """A node in the pipeline: an agent and the message type it expects."""

    agent: str                                           # имя агента-этапа
    expects: MessageType                                 # тип сообщения, который этот этап ожидает на входе


@dataclass
class RunReport:
    success: bool                                        # общий итог прогона (все этапы успешны?)
    results: List[AgentResult] = field(default_factory=list)  # результаты по каждому этапу
    failed_stage: Optional[str] = None                   # имя первого упавшего этапа (если был)

    def by_agent(self) -> Dict[str, AgentResult]:
        return {r.agent: r for r in self.results}        # индекс результатов по имени агента

    def total_seconds(self) -> float:
        return sum(r.duration_s for r in self.results)   # суммарное время всех этапов


class Orchestrator:
    def __init__(self, context: AgentContext, fail_fast: bool = True) -> None:
        self.context = context                           # общий контекст прогона
        self.bus = MessageBus()                          # собственная шина сообщений
        self.fail_fast = fail_fast                       # останавливаться ли на первой ошибке
        self.pipeline: List[PipelineStage] = []          # упорядоченный список этапов

    def register(self, agent: BaseAgent, expects: Optional[MessageType] = None) -> "Orchestrator":
        self.bus.register(agent)                         # регистрируем агента в шине
        # ожидаемый вход: явный expects, иначе первый тип из consumes, иначе START
        expects = expects if expects is not None else (agent.consumes[0] if agent.consumes else MessageType.START)
        self.pipeline.append(PipelineStage(agent.name, expects))  # добавляем этап в пайплайн
        return self                                      # возвращаем self для chaining (.register().register())

    def run(self) -> RunReport:
        log = self.context.logger("orchestrator")        # логгер оркестратора
        report = RunReport(success=True)                 # стартовый отчёт (пока успешен)
        # стартовое сообщение START, адресованное первому агенту пайплайна
        current = Message(MessageType.START, "orchestrator", self.pipeline[0].agent if self.pipeline else "")
        self.bus.publish(current)                        # публикуем START в шину (попадёт в трассу)

        for stage in self.pipeline:                      # идём по этапам строго по порядку
            agent = self.bus.agent(stage.agent)          # достаём экземпляр агента по имени
            input_message = self._adapt(current, stage.expects, agent.name)  # адаптируем выход→вход
            self.bus.publish(input_message)              # публикуем входное сообщение этапа
            result = agent.run(self.context, input_message)  # ЗАПУСКАЕМ агента (через шаблонный run())
            report.results.append(result)                # сохраняем результат этапа в отчёт
            self.bus.publish(result.message)             # публикуем выход агента в шину

            if not result.success:                       # если этап упал…
                report.success = False                   # общий итог — провал
                report.failed_stage = stage.agent        # запоминаем упавший этап
                log.error("pipeline halted", stage=stage.agent, error=result.error)  # лог остановки
                if self.fail_fast:                       # при fail-fast…
                    break                                # …прерываем пайплайн немедленно
            current = result.message                     # выход текущего этапа становится «текущим» для следующего

        log.info(                                        # финальный лог о завершении пайплайна
            "pipeline finished",
            success=report.success,                      #   общий успех
            stages=len(report.results),                  #   сколько этапов отработало
            total_seconds=round(report.total_seconds(), 2),  #   суммарное время
        )
        return report                                    # возвращаем отчёт о прогоне

    def _adapt(self, previous: Message, expected: MessageType, recipient: str) -> Message:
        """Bridge the previous output to the next stage, preserving the payload."""
        return Message(                                  # строим входное сообщение следующего этапа
            type=expected,                               #   тип = тот, что ожидает следующий агент
            sender=previous.sender,                      #   отправитель = из предыдущего сообщения
            recipient=recipient,                         #   получатель = следующий агент
            payload=previous.payload,                    #   СОХРАНЯЕМ полезную нагрузку без изменений
            correlation_id=previous.correlation_id,      #   и correlation_id (сквозная трассировка)
        )
