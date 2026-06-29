"""Structured logging for agents.

Provides a :class:`StructuredLogger` that records actions, decisions, errors and
operation timings both to the console (human readable) and to a JSONL file
(machine readable, one event per line). A :meth:`StructuredLogger.timer` context
manager measures wall-clock duration of any operation.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.core (каркас).
НАЗНАЧЕНИЕ:  СТРУКТУРИРОВАННОЕ логирование. Каждое событие пишется и в консоль
             (человекочитаемо), и в JSONL-файл (по одному JSON-объекту на строку —
             удобно парсить). Особые виды событий: decision() (решение агента с
             числовым обоснованием) и timer() (замер длительности операции).
ИМПОРТИРУЕТ:
  - json, logging, time      : сериализация, стандартный logging, таймеры.
  - contextlib.contextmanager: для timer() как context manager.
  - dataclasses              : @dataclass для LoggerFactory.
  - datetime/timezone        : метка времени события (UTC).
  - pathlib.Path             : путь к файлу лога.
  - typing                   : аннотации.
ЭКСПОРТИРУЕТ:
  - LoggerFactory   : создаёт per-agent логгеры с общим JSONL-приёмником.
  - StructuredLogger: info/debug/warning/error/decision/timer.
ВНУТРЕННИЕ:
  - _JsonlHandler   : handler, пишущий по одному JSON на строку.
  - _AgentDefaultFilter: подставляет поле agent, если оно не задано.
КЕМ ИСПОЛЬЗУЕТСЯ:
  - core/context.py создаёт LoggerFactory; agents/* зовут context.logger(name).
КОНСТАНТЫ:
  - _LEVELS : строковое имя уровня → числовой уровень модуля logging.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import json                                              # сериализация события в JSON-строку
import logging                                           # стандартная подсистема логирования
import time                                              # измерение длительности операций
from contextlib import contextmanager                    # декоратор для timer() как with-блока
from dataclasses import dataclass, field                 # @dataclass + поле _root
from datetime import datetime, timezone                  # метка времени события (UTC)
from pathlib import Path                                 # путь к каталогу/файлу логов
from typing import Any, Dict, Iterator, Optional         # аннотации

_LEVELS = {                                              # маппинг строковых уровней → числовых logging.*
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class _JsonlHandler(logging.Handler):
    """Logging handler that appends a JSON object per record to a file."""

    def __init__(self, path: Path) -> None:
        super().__init__()                               # инициализация базового Handler
        self.path = path                                 # путь к JSONL-файлу
        self.path.parent.mkdir(parents=True, exist_ok=True)  # создаём каталог под файл
        self._fh = open(self.path, "a", encoding="utf-8")    # открываем файл на дозапись (append)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - io
        payload: Dict[str, Any] = {                      # собираем структуру одного события
            "ts": datetime.now(timezone.utc).isoformat(),  #   метка времени UTC
            "level": record.levelname,                   #   уровень (INFO/ERROR…)
            "agent": getattr(record, "agent", record.name),  #   имя агента (или имя логгера)
            "event": record.getMessage(),                #   текст события
        }
        extra = getattr(record, "structured", None)      # доп. структурированные поля (если есть)
        if isinstance(extra, dict):                      # если это словарь…
            payload.update(extra)                        # …добавляем его поля в событие
        try:
            self._fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")  # пишем JSON-строку
            self._fh.flush()                             # сбрасываем буфер на диск немедленно
        except Exception:                                # ошибки записи лога не должны ронять программу
            pass                                         # тихо игнорируем

    def close(self) -> None:  # pragma: no cover - io
        try:
            self._fh.close()                             # закрываем файловый дескриптор
        finally:
            super().close()                              # и закрываем сам handler


@dataclass
class LoggerFactory:
    """Creates per-agent :class:`StructuredLogger` instances sharing one JSONL sink."""

    log_dir: Path                                        # каталог для файла лога
    level: str = "INFO"                                  # уровень логирования по умолчанию
    console: bool = True                                 # выводить ли в консоль
    jsonl: bool = True                                   # писать ли JSONL-файл
    file_name: str = "cryptohedge.jsonl"                 # имя JSONL-файла
    timing: bool = True                                  # включать ли логирование таймингов
    _root: logging.Logger = field(init=False)            # корневой логгер "cryptohedge" (создаётся в __post_init__)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)                # нормализуем путь к каталогу логов
        self.log_dir.mkdir(parents=True, exist_ok=True)  # создаём каталог логов
        self._root = logging.getLogger("cryptohedge")    # берём именованный корневой логгер проекта
        self._root.setLevel(_LEVELS.get(self.level.upper(), logging.INFO))  # выставляем уровень
        # close existing handlers before dropping them so file handles are released
        # (prevents leaked handles / Windows file locks when contexts are recreated)
        for handler in list(self._root.handlers):        # перебираем уже навешанные обработчики…
            try:
                handler.close()                          # …закрываем (освобождаем файловые дескрипторы)
            finally:
                self._root.removeHandler(handler)        # …и снимаем с логгера
        self._root.propagate = False                     # не дублировать сообщения в корневой root-логгер

        if self.console:                                 # если нужен вывод в консоль…
            stream = logging.StreamHandler()             #   создаём консольный обработчик
            stream.setFormatter(                         #   задаём формат строки лога
                logging.Formatter("%(asctime)s | %(levelname)-7s | %(agent)-22s | %(message)s")
            )
            stream.addFilter(_AgentDefaultFilter())      #   фильтр гарантирует наличие поля agent
            self._root.addHandler(stream)                #   навешиваем обработчик на логгер

        if self.jsonl:                                   # если нужен JSONL-файл…
            self._root.addHandler(_JsonlHandler(self.log_dir / self.file_name))  # добавляем файловый handler

    def get(self, agent_name: str) -> "StructuredLogger":
        return StructuredLogger(agent_name, self._root, timing=self.timing)  # создать логгер для агента


class _AgentDefaultFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "agent"):                 # если в записи нет поля agent…
            record.agent = record.name                   # …подставляем имя логгера, чтобы формат не падал
        return True                                      # запись всегда пропускаем дальше


class StructuredLogger:
    """Thin wrapper that attaches the agent name and structured fields to records."""

    def __init__(self, agent_name: str, logger: logging.Logger, timing: bool = True) -> None:
        self.agent_name = agent_name                     # имя агента-владельца логгера
        self._logger = logger                            # ссылка на общий корневой логгер
        self._timing = timing                            # включено ли логирование таймингов
        self.timings: Dict[str, float] = {}              # накопленные длительности операций {имя: секунды}

    def _log(self, level: int, event: str, **fields: Any) -> None:
        # Прокидываем имя агента и произвольные структурированные поля через extra.
        self._logger.log(level, event, extra={"agent": self.agent_name, "structured": fields})

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)        # событие уровня DEBUG

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)         # событие уровня INFO

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)      # событие уровня WARNING

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)        # событие уровня ERROR

    def decision(self, what: str, **fields: Any) -> None:
        """Log a decision taken by the agent with its quantitative justification."""
        # Особый вид события: РЕШЕНИЕ агента (помечается kind="decision" для фильтрации).
        self._log(logging.INFO, f"DECISION: {what}", kind="decision", **fields)

    @contextmanager
    def timer(self, operation: str, **fields: Any) -> Iterator[None]:
        """Context manager measuring and logging the duration of ``operation``."""
        start = time.perf_counter()                      # засекаем начало операции
        try:
            yield                                        # выполняется тело with-блока
        finally:
            elapsed = time.perf_counter() - start        # вычисляем длительность
            self.timings[operation] = self.timings.get(operation, 0.0) + elapsed  # накапливаем по имени операции
            if self._timing:                             # если логирование таймингов включено…
                self._log(                               # …пишем событие тайминга
                    logging.INFO,
                    f"timing: {operation}",
                    kind="timing",                       #   помечаем kind="timing"
                    operation=operation,                 #   имя операции
                    seconds=round(elapsed, 4),           #   длительность в секундах
                    **fields,                            #   доп. поля
                )
