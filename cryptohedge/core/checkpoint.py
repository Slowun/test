"""Crash-recovery via checkpointing.

The :class:`CheckpointManager` persists arbitrary artifacts to disk and tracks
which pipeline stages have completed in a JSON manifest. On restart the
orchestrator can skip already-completed stages and reload their outputs, so a run
interrupted by a failure resumes where it left off.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.core (каркас).
НАЗНАЧЕНИЕ:  ЧЕКПОЙНТЫ — сохраняет артефакты этапов на диск и ведёт манифест
             выполненных этапов. При resume оркестратор пропускает готовые этапы
             и подгружает их результаты (продолжение прогона после сбоя).
ИМПОРТИРУЕТ:
  - json, pickle        : сериализация (JSON для простых структур, pickle для прочего).
  - datetime/timezone   : временны́е метки в манифесте.
  - pathlib.Path        : пути к файлам чекпойнтов.
  - typing              : аннотации.
  - pandas              : сохранение DataFrame в parquet.
ЭКСПОРТИРУЕТ:
  - CheckpointManager : save/load/exists + is_completed/mark_completed/reset.
ВНУТРЕННИЕ ХЕЛПЕРЫ:
  - _is_jsonable / _load_json / _load_pickle.
КЕМ ИСПОЛЬЗУЕТСЯ:
  - core/context.py создаёт менеджер; core/agent.py зовёт save/load/exists/
    mark_completed/is_completed в шаблонном run().
ФОРМАТЫ ХРАНЕНИЯ:  DataFrame→.parquet, json-совместимое→.json, остальное→.pkl.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import json                                              # сериализация манифеста и json-артефактов
import pickle                                            # сериализация произвольных объектов
from datetime import datetime, timezone                  # метки времени (UTC) в манифесте
from pathlib import Path                                 # пути к каталогу/файлам чекпойнтов
from typing import Any, Dict, List, Optional             # аннотации

import pandas as pd                                      # сохранение/чтение DataFrame в parquet


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path, run_id: str = "default", enabled: bool = True) -> None:
        self.enabled = enabled                           # включены ли чекпойнты в принципе
        self.run_dir = Path(checkpoint_dir) / run_id     # подкаталог конкретного прогона (run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)  # создаём каталог прогона
        self.manifest_path = self.run_dir / "manifest.json"  # путь к манифесту выполненных этапов
        self._manifest: Dict[str, Any] = self._load_manifest()  # загружаем (или создаём) манифест

    # ------------------------------------------------------------------ manifest
    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():                  # если манифест уже есть на диске…
            with open(self.manifest_path, "r", encoding="utf-8") as fh:  # …открываем его…
                return json.load(fh)                     # …и читаем
        return {"stages": {}, "created": datetime.now(timezone.utc).isoformat()}  # иначе — новый пустой манифест

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as fh:  # открываем манифест на запись
            json.dump(self._manifest, fh, indent=2, ensure_ascii=False, default=str)  # сериализуем в JSON

    def is_completed(self, stage: str) -> bool:
        return self.enabled and stage in self._manifest["stages"]  # этап завершён, если чекпойнты вкл. и он в манифесте

    def completed_stages(self) -> List[str]:
        return list(self._manifest["stages"])            # список имён завершённых этапов

    def mark_completed(self, stage: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self._manifest["stages"][stage] = {              # отмечаем этап как завершённый…
            "ts": datetime.now(timezone.utc).isoformat(),  #   …с меткой времени…
            "meta": meta or {},                          #   …и доп. метаданными (например, длительность)
        }
        self._save_manifest()                            # сразу сохраняем манифест на диск

    def reset(self) -> None:
        self._manifest = {"stages": {}, "created": datetime.now(timezone.utc).isoformat()}  # обнуляем манифест
        self._save_manifest()                            # и записываем пустой манифест

    # ----------------------------------------------------------------- artifacts
    def save(self, key: str, obj: Any) -> Path:
        """Persist ``obj`` choosing a format based on its type."""
        if isinstance(obj, pd.DataFrame):                # DataFrame → колоночный parquet (эффективно)
            path = self.run_dir / f"{key}.parquet"       #   путь .parquet
            obj.to_parquet(path)                         #   сохраняем
        elif isinstance(obj, (dict, list)) and _is_jsonable(obj):  # простые json-совместимые структуры → JSON
            path = self.run_dir / f"{key}.json"          #   путь .json
            with open(path, "w", encoding="utf-8") as fh:  #   открываем на запись
                json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)  #   сериализуем
        else:                                            # всё остальное (объекты, numpy и т.п.) → pickle
            path = self.run_dir / f"{key}.pkl"           #   путь .pkl
            with open(path, "wb") as fh:                 #   открываем в бинарном режиме
                pickle.dump(obj, fh)                     #   сериализуем pickle'ом
        return path                                      # возвращаем путь сохранённого файла

    def load(self, key: str) -> Any:
        # Пробуем форматы по очереди: parquet → json → pickle (что найдём — то и загрузим).
        for suffix, loader in ((".parquet", pd.read_parquet), (".json", _load_json), (".pkl", _load_pickle)):
            path = self.run_dir / f"{key}{suffix}"       # кандидат-путь для данного формата
            if path.exists():                            # если файл такого формата существует…
                return loader(path)                      # …грузим подходящим загрузчиком
        raise FileNotFoundError(f"No checkpoint artifact for key '{key}' in {self.run_dir}")  # ничего не нашли

    def exists(self, key: str) -> bool:
        # Артефакт существует, если есть файл в ЛЮБОМ из поддерживаемых форматов.
        return any((self.run_dir / f"{key}{s}").exists() for s in (".parquet", ".json", ".pkl"))


def _is_jsonable(obj: Any) -> bool:
    try:
        json.dumps(obj, default=str)                     # пробуем сериализовать в JSON…
        return True                                      # …получилось — объект json-совместим
    except (TypeError, ValueError):                      # не сериализуется стандартно…
        return False                                     # …значит, не json-совместим (пойдёт в pickle)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:        # открыть json-файл
        return json.load(fh)                             # прочитать и вернуть объект


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as fh:                         # открыть pkl-файл в бинарном режиме
        return pickle.load(fh)                           # десериализовать и вернуть объект
