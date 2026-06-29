"""Command-line entry point and programmatic runner for the hedging pipeline.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge (верхний уровень) — ТОЧКА ВХОДА.
НАЗНАЧЕНИЕ:  запуск всего пайплайна двумя способами:
             1) программно: run_pipeline(...) — используется в ноутбуке и тестах;
             2) из терминала: main() с argparse (python -m cryptohedge.cli).
             Собирает конфиг → контекст → оркестратор → запускает и печатает итог.
ИМПОРТИРУЕТ:
  - argparse                 : разбор аргументов командной строки.
  - pathlib.Path, typing     : типы путей/аргументов.
  - agents.build_pipeline    : сборка оркестратора из 11 агентов.
  - core.config.load_config  : загрузка конфигурации.
  - core.context.AgentContext: контекст прогона.
  - core.orchestrator.RunReport: тип отчёта о прогоне.
ЭКСПОРТИРУЕТ:
  - run_pipeline() -> (AgentContext, RunReport) — программный запуск.
  - main(argv) -> int — CLI-точка входа (код возврата 0/1).
КЕМ ИСПОЛЬЗУЕТСЯ:
  - solution.ipynb, tests/test_integration.py, ручной запуск из консоли.
АРГУМЕНТЫ CLI: --config, --root, --provider, --reset, --no-fail-fast.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import argparse                                          # разбор аргументов командной строки
from pathlib import Path                                 # тип пути
from typing import Any, Dict, Optional, Tuple            # аннотации

from cryptohedge.agents import build_pipeline            # сборка пайплайна (11 агентов)
from cryptohedge.core.config import load_config          # загрузка конфигурации
from cryptohedge.core.context import AgentContext        # контекст прогона
from cryptohedge.core.orchestrator import RunReport      # тип отчёта о прогоне


def run_pipeline(
    config_dir: str | Path = "config",                  # каталог конфигурации
    root: str | Path = ".",                              # корень проекта (куда писать артефакты)
    overrides: Optional[Dict[str, Any]] = None,         # программные переопределения конфига
    fail_fast: bool = True,                              # останавливаться ли на первой ошибке
) -> Tuple[AgentContext, RunReport]:
    """Build the context, wire the agents and run the full pipeline.

    Returns the populated :class:`AgentContext` (with all blackboard artifacts)
    and the :class:`RunReport`. Designed to be called from the notebook too.
    """
    config = load_config(config_dir, overrides=overrides)  # читаем и валидируем конфиг
    context = AgentContext(config, root=root)            # создаём контекст (сидинг, логгеры, чекпойнты)
    orchestrator = build_pipeline(context, fail_fast=fail_fast)  # собираем оркестратор из агентов
    report = orchestrator.run()                          # ЗАПУСКАЕМ весь пайплайн
    return context, report                               # отдаём контекст (с артефактами) и отчёт


def _print_summary(report: RunReport) -> None:
    print("\n================ PIPELINE SUMMARY ================")  # шапка сводки
    for r in report.results:                             # по каждому этапу…
        flag = "OK " if r.success else "FAIL"            # метка успеха/провала
        skip = " (restored)" if r.skipped else ""        # пометка восстановления из чекпойнта
        print(f"  [{flag}] {r.agent:<24} {r.duration_s:6.2f}s{skip}"   # имя агента + длительность
              + (f"  error={r.error}" if not r.success else ""))        # текст ошибки при провале
    print(f"  total: {report.total_seconds():.2f}s | success={report.success}")  # суммарное время и итог
    print("=================================================\n")  # подвал сводки


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CryptoHedge multi-agent hedging pipeline")  # парсер CLI
    parser.add_argument("--config", default="config", help="configuration directory")        # каталог конфига
    parser.add_argument("--root", default=".", help="project root for artifacts")            # корень артефактов
    parser.add_argument("--provider", default=None, help="override data provider")           # переопределить провайдер
    parser.add_argument("--reset", action="store_true", help="ignore checkpoints and rerun all stages")  # сброс чекпойнтов
    parser.add_argument("--no-fail-fast", action="store_true", help="continue after a stage fails")      # не падать на ошибке
    args = parser.parse_args(argv)                       # разбираем аргументы

    overrides: Dict[str, Any] = {}                       # копим переопределения конфига из флагов
    if args.provider:                                    # если задан --provider…
        overrides["data"] = {"provider": args.provider}  # …переопределяем провайдер данных
    if args.reset:                                       # если задан --reset…
        overrides.setdefault("runtime", {})["resume"] = False  # …отключаем resume (прогон заново)

    context, report = run_pipeline(args.config, args.root, overrides, fail_fast=not args.no_fail_fast)  # запуск
    _print_summary(report)                               # печатаем сводку прогона
    if context.has("dashboard_path"):                    # если дашборд был создан…
        print(f"Dashboard: {context.get('dashboard_path')}")  # …выводим путь к нему
    return 0 if report.success else 1                    # код возврата: 0 — успех, 1 — провал


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())                             # запуск из консоли: код возврата → код выхода процесса
