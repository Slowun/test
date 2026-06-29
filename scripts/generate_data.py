"""Generate (or regenerate) the reproducible bundled dataset under ``data/raw``.

Run once before the first pipeline execution (the bundled provider also does this
lazily). Everything is a deterministic function of the configured ``seed``.

Usage:
    python scripts/generate_data.py [--config config] [--root .]

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        scripts (вспомогательные утилиты) — ТОЧКА ВХОДА (CLI-скрипт).
НАЗНАЧЕНИЕ:  заранее сгенерировать «вшитый» (bundled) набор данных в data/raw,
             чтобы пайплайн стартовал без обращения к сети. Полностью
             детерминирован по seed из конфига (воспроизводимость).
ИМПОРТИРУЕТ:
  - argparse, pathlib.Path        : CLI и пути.
  - core.config.load_config       : загрузка конфигурации.
  - services.providers.bundled.BundledProvider: материализация датасета.
ЭКСПОРТИРУЕТ: main() -> int (код возврата процесса).
ЗАПУСК: python scripts/generate_data.py [--config config] [--root .]
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import argparse                                          # разбор аргументов командной строки
from pathlib import Path                                 # тип пути

from cryptohedge.core.config import load_config          # загрузка конфигурации
from cryptohedge.services.providers.bundled import BundledProvider  # провайдер вшитых данных


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the bundled hedging dataset")  # парсер CLI
    parser.add_argument("--config", default="config")    # каталог конфигурации
    parser.add_argument("--root", default=".")           # корень проекта (куда писать data/raw)
    args = parser.parse_args()                            # разбираем аргументы

    cfg = load_config(args.config)                       # читаем и валидируем конфиг
    provider = BundledProvider(cfg.data, root=Path(args.root), seed=cfg.seed,  # провайдер с параметрами из конфига
                               n_steps=cfg.horizons.analysis_days)  # число шагов = горизонт анализа
    print(f"Generating dataset: provider=bundled, universe={cfg.data.universe_size}, "  # информируем о параметрах
          f"seed={cfg.seed}, samples={cfg.horizons.analysis_days + 1} ...")
    bundle = provider.materialize()                      # ГЕНЕРАЦИЯ и запись датасета на диск
    print(f"  symbols:            {len(bundle.symbols)}")          # число символов
    print(f"  spot_close shape:   {bundle.spot_close.shape}")     # форма матрицы цен закрытия
    print(f"  option rows:        {bundle.option_market_data.shape[0]}")  # число строк опционных котировок
    print(f"  written to:         {provider.raw_dir}")            # куда записан датасет
    return 0                                             # успешное завершение


if __name__ == "__main__":
    raise SystemExit(main())                             # запуск из консоли: код возврата → код выхода процесса
