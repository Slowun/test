"""Data Acquisition Agent.

Role: load and validate market data (spot universe + primary-asset option chain)
from the configured provider, compute returns and publish a clean dataset.
Responsibility boundary: it is the *only* agent that touches external data
sources; everyone else consumes its validated output.

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       1 / 11 — DataAcquisitionAgent (первый этап пайплайна).
НАЗНАЧЕНИЕ:  ЕДИНСТВЕННЫЙ агент, который трогает внешние источники данных.
             Загружает спот-вселенную и опционную цепочку BTC через выбранный
             провайдер, валидирует, считает лог-доходности и публикует чистый
             датасет на blackboard.
ВХОД (consumes):  MessageType.START.
ВЫХОД (produces): MessageType.DATA_READY → агенту market_analysis.
КЛАДЁТ НА ДОСКУ:  spot_close, spot_bars, returns, market_data, symbols,
                  primary_symbol, data_meta (они же checkpoint_keys).
ИМПОРТИРУЕТ:
  - numpy/pandas                 : расчёт доходностей и валидация.
  - core.agent/context/message   : контракт агента и инфраструктура.
  - services.providers.build_provider : фабрика провайдера данных по конфигу.
КОНФИГ:  config.data (провайдер/вселенная/опционы), config.horizons.analysis_days.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import numpy as np                                       # лог-доходности (np.log/diff)
import pandas as pd                                       # таблицы спота/опционов

from cryptohedge.core.agent import BaseAgent             # базовый контракт агента
from cryptohedge.core.context import AgentContext        # контекст (доска/логгер/конфиг)
from cryptohedge.core.message import Message, MessageType # сообщения и их типы
from cryptohedge.services.providers import build_provider  # фабрика провайдера данных


class DataAcquisitionAgent(BaseAgent):
    name = "data_acquisition"                            # уникальное имя агента / id этапа
    consumes = [MessageType.START]                       # запускается по сигналу START
    produces = MessageType.DATA_READY                    # выпускает DATA_READY
    checkpoint_keys = ["spot_close", "returns", "market_data", "spot_bars", "symbols",  # ключи для чекпойнта
                       "primary_symbol", "data_meta"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        cfg = context.config                             # полный конфиг
        n_steps = cfg.horizons.analysis_days             # сколько дней истории грузить

        provider = build_provider(cfg.data, root=context.root, seed=cfg.seed, n_steps=n_steps)  # создаём провайдер
        with log.timer("load_data", provider=cfg.data.provider):  # замеряем время загрузки
            bundle = provider.load()                     # загружаем «пакет» данных (спот + опционы)

        self._validate(bundle, log)                      # валидируем загруженные данные

        returns = np.log(bundle.spot_close).diff().dropna(how="all")  # лог-доходности (по строкам)
        returns = returns.dropna(axis=1, how="any")      # убираем колонки с пропусками (неполные ряды)

        context.put("spot_close", bundle.spot_close)     # цены закрытия спота → на доску
        context.put("spot_bars", bundle.spot_bars)       # полные OHLCV-бары → на доску
        context.put("returns", returns)                  # матрица доходностей → на доску
        context.put("market_data", bundle.option_market_data)  # опционные данные → на доску
        context.put("symbols", bundle.symbols)           # список тикеров → на доску
        context.put("primary_symbol", bundle.primary_symbol)   # первичный актив → на доску

        meta = {                                         # сводные метаданные загрузки
            "provider": cfg.data.provider,               #   какой провайдер
            "n_symbols": len(bundle.symbols),            #   число тикеров
            "n_samples": int(bundle.spot_close.shape[0]),  #   число временны́х точек
            "n_option_rows": int(bundle.option_market_data.shape[0]),  #   число строк опционов
            "primary_symbol": bundle.primary_symbol,     #   первичный актив
            "date_start": str(bundle.spot_close.index[0]),   #   начало периода
            "date_end": str(bundle.spot_close.index[-1]),    #   конец периода
        }
        context.put("data_meta", meta)                   # метаданные → на доску
        log.decision("loaded and validated market data", **meta)  # лог-решение с метаданными

        return Message(self.produces, self.name, "market_analysis", payload=meta,  # DATA_READY следующему агенту
                       correlation_id=message.correlation_id)

    def _validate(self, bundle, log) -> None:
        close = bundle.spot_close                        # цены закрытия для проверок
        if close.isna().any().any():                     # есть ли пропуски (NaN)?
            n = int(close.isna().sum().sum())            #   сколько всего пропусков
            log.warning("forward-filling NaNs in spot_close", n_missing=n)  #   предупреждаем
            bundle.spot_close = close.ffill().bfill()    #   заполняем вперёд и назад
        if (bundle.spot_close <= 0).any().any():         # цены должны быть строго положительны…
            raise ValueError("Non-positive spot prices detected")  #   иначе ошибка
        if bundle.primary_symbol not in bundle.spot_close.columns:  # первичный актив обязан присутствовать…
            raise ValueError("Primary symbol missing from spot data")  #   иначе ошибка
        opt = bundle.option_market_data                  # опционные данные
        if (opt["price"] < 0).any():                     # цены опционов не могут быть отрицательны…
            raise ValueError("Negative option prices detected")  #   иначе ошибка
        log.info("validation passed", n_symbols=len(bundle.symbols),  # лог успешной валидации
                 n_samples=int(bundle.spot_close.shape[0]))
