"""Greeks Calculation Agent.

Role: build the liability portfolio (protective put) and the vega-hedge option,
then compute per-instrument and aggregated portfolio greeks - delta, gamma, vega,
theta, rho, vanna, volga, charm - at the latest slice, across the full history
(for monitoring) and across the strike grid (for the dashboard heatmap).

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       4 / 11 — GreeksCalculationAgent.
НАЗНАЧЕНИЕ:  строит книгу обязательств (защитный put) и vega-хедж опцион, затем
             считает греки (Δ Γ ν Θ ρ vanna volga charm): по каждому инструменту,
             агрегированно по портфелю, по всей истории (мониторинг) и по сетке
             страйков (тепловая карта дашборда). Определяет зону баланса дельты.
ВХОД (consumes):  CALIBRATION_READY (от агента 3).
ВЫХОД (produces): GREEKS_READY → агенту hedging_decision.
КЛАДЁТ НА ДОСКУ:  hedge_setup, hedge_contracts, portfolio_greeks_latest,
                  greeks_per_instrument, greeks_timeseries, chain_greeks, hedge_status.
ИМПОРТИРУЕТ:
  - domain.greeks.Greeks, domain.market.HestonParameters/OptionContract.
  - services.greeks.HestonGreeksEngine/aggregate : движок расчёта греков.
  - services.hedging_engine.YEAR_NANOS           : наносекунд в году (для TTM).
  - services.portfolio_spec.*                    : сетка страйков и сборка книги.
КОНФИГ:  config.greeks (движок/сдвиги), config.hedging (moneyness, зоны дельты),
         config.investment.capital_usd.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import Dict, List                            # аннотации

import numpy as np                                       # числовые операции
import pandas as pd                                       # таблицы временны́х рядов/цепочки

from cryptohedge.core.agent import BaseAgent             # контракт агента
from cryptohedge.core.context import AgentContext        # контекст
from cryptohedge.core.message import Message, MessageType # сообщения
from cryptohedge.domain.greeks import Greeks             # структура греков
from cryptohedge.domain.market import HestonParameters, OptionContract  # параметры Хестона и опцион
from cryptohedge.services.greeks import HestonGreeksEngine, aggregate   # движок и агрегация греков
from cryptohedge.services.hedging_engine import YEAR_NANOS  # константа: наносекунд в году
from cryptohedge.services.portfolio_spec import available_strikes, build_hedge_setup  # страйки и сборка книги
from cryptohedge.services.providers.base import INSTR_ASSET  # константа типа «базовый актив»


class GreeksCalculationAgent(BaseAgent):
    name = "greeks_calculation"                          # имя агента / id этапа
    consumes = [MessageType.CALIBRATION_READY]           # принимает CALIBRATION_READY
    produces = MessageType.GREEKS_READY                  # выпускает GREEKS_READY
    checkpoint_keys = ["hedge_setup", "portfolio_greeks_latest", "greeks_per_instrument",  # ключи чекпойнта
                       "greeks_timeseries", "chain_greeks", "hedge_status", "hedge_contracts"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        engine = HestonGreeksEngine(context.config.greeks)  # движок греков (настройки из конфига)
        md: pd.DataFrame = context.require("market_data")   # опционные данные
        calibr: pd.DataFrame = context.require("calibr_data")  # параметры калибровки по срезам
        primary = context.require("primary_symbol")      # первичный актив
        sizing = context.require("hedge_sizing")         # объём хеджа (от агента 2)

        spot0 = float(md[md["instrument_type"] == INSTR_ASSET].sort_values("sample_idx")["price"].iloc[0])  # стартовый спот
        liability, vega_option = build_hedge_setup(      # строим книгу обязательств + vega-хедж опцион
            md, spot0, sizing.quantity_to_hedge, primary,
            put_moneyness=context.config.hedging.liability_put_moneyness,  # страйк защитного put
            call_moneyness=context.config.hedging.vega_call_moneyness,     # страйк vega-call
        )

        # ---- full-history portfolio greeks (monitoring time series)  # --- греки по всей истории (мониторинг) ---
        with log.timer("greeks_timeseries", n=len(calibr)):  # замеряем время
            ts_rows = self._timeseries(engine, md, calibr, liability, vega_option)  # строки греков по срезам
        greeks_timeseries = pd.DataFrame(ts_rows)        # временно́й ряд греков

        # ---- latest-slice detailed greeks               # --- детальные греки на ПОСЛЕДНЕМ срезе ---
        last = calibr.sort_values("sample_idx").iloc[-1]  # последний срез калибровки
        sidx = int(last["sample_idx"])                   # его индекс
        grp = md[md["sample_idx"] == sidx]               # данные последнего среза
        spot = float(grp[grp["instrument_type"] == INSTR_ASSET]["price"].iloc[0])  # спот на последнем срезе
        params = self._params(last)                      # параметры Хестона последнего среза
        ts_ns = int(pd.Timestamp(last["timestamp"]).value)  # метка времени в наносекундах

        per_instrument: Dict[str, Greeks] = {}           # греки по каждому инструменту книги
        for i, c in enumerate(liability):                # по каждому put-обязательству…
            ttm = (c.expiry_ts - ts_ns) / YEAR_NANOS     #   срок до экспирации в годах
            per_instrument[f"liability_put_{c.strike:.0f}"] = engine.compute(spot, params, c, ttm)  # считаем греки
        ttm_v = (vega_option.expiry_ts - ts_ns) / YEAR_NANOS  # TTM vega-опциона
        per_instrument["vega_hedge_call"] = engine.compute(spot, params, vega_option, ttm_v)  # греки vega-хеджа

        portfolio = aggregate([g for name, g in per_instrument.items() if name.startswith("liability")])  # агрегат книги

        # ---- chain greeks for the heatmap               # --- греки по сетке страйков (тепловая карта) ---
        chain = self._chain_greeks(engine, spot, params, md, ttm_v)  # профиль греков по страйкам

        # ---- delta/vega balance status (green/red zone)  # --- статус баланса дельты (зелёная/красная зона) ---
        delta_usd = portfolio.delta * spot               # дельта в долларах
        frac = abs(delta_usd) / context.config.investment.capital_usd  # доля дельты от капитала
        zone = ("green" if frac <= context.config.hedging.delta_green_zone else  # классификация зоны:
                "red" if frac >= context.config.hedging.delta_red_zone else "amber")  # green/amber/red
        status = {"delta_usd": delta_usd, "delta_fraction": frac, "zone": zone,  # сводный статус баланса
                  "portfolio_vega": portfolio.vega, "portfolio_gamma": portfolio.gamma}

        context.put("hedge_setup", (liability, vega_option))  # книга + vega-опцион → на доску
        context.put("hedge_contracts", {                 # сериализуемое описание контрактов → на доску
            "liability": [c.__dict__ for c in liability],
            "vega_option": vega_option.__dict__,
        })
        context.put("portfolio_greeks_latest", portfolio.to_dict())  # последние греки портфеля → на доску
        context.put("greeks_per_instrument", {k: v.to_dict() for k, v in per_instrument.items()})  # по инструментам
        context.put("greeks_timeseries", greeks_timeseries)  # временно́й ряд греков → на доску
        context.put("chain_greeks", chain)               # греки по страйкам → на доску
        context.put("hedge_status", status)              # статус баланса → на доску

        log.decision("computed portfolio greeks", **{k: round(v, 4) for k, v in portfolio.to_dict().items()})  # лог греков
        log.decision("delta balance", zone=zone, delta_fraction=round(frac, 4))  # лог баланса дельты

        return Message(self.produces, self.name, "hedging_decision",  # GREEKS_READY следующему агенту
                       payload={"zone": zone, "portfolio_delta": portfolio.delta,
                                "portfolio_vega": portfolio.vega},
                       correlation_id=message.correlation_id)

    @staticmethod
    def _params(row: pd.Series) -> HestonParameters:
        # Собираем HestonParameters из строки таблицы калибровки.
        return HestonParameters(v0=float(row["v0"]), kappa=float(row["kappa"]), theta=float(row["theta"]),
                                eps=float(row["eps"]), rho=float(row["rho"]),
                                flat_yield=float(row.get("flat_yield", 0.0)))

    def _timeseries(self, engine, md, calibr, liability, vega_option) -> List[dict]:
        rows = []                                        # строки временно́го ряда греков
        for _, row in calibr.sort_values("sample_idx").iterrows():  # по каждому срезу калибровки…
            sidx = int(row["sample_idx"])                #   индекс среза
            grp = md[md["sample_idx"] == sidx]           #   данные среза
            asset = grp[grp["instrument_type"] == INSTR_ASSET]  #   строки базового актива
            if asset.empty:                              #   нет спота на срезе…
                continue                                 #     → пропускаем
            spot = float(asset["price"].iloc[0])         #   спот на срезе
            ts_ns = int(pd.Timestamp(row["timestamp"]).value)  # метка времени, нс
            params = self._params(row)                   #   параметры Хестона среза
            total = Greeks()                             #   аккумулятор греков книги
            for c in liability:                          #   по каждому put-обязательству…
                ttm = (c.expiry_ts - ts_ns) / YEAR_NANOS #     TTM в годах
                if ttm > 0:                              #     если опцион ещё жив…
                    total = total + engine.compute(spot, params, c, ttm)  # складываем греки
            d = total.to_dict()                          #   греки в словарь
            d.update({"ts": pd.to_datetime(ts_ns), "sample_idx": sidx, "spot": spot})  # добавляем время/срез/спот
            rows.append(d)                               #   копим строку
        return rows                                      # список строк временно́го ряда

    def _chain_greeks(self, engine, spot, params, md, ttm) -> pd.DataFrame:
        strikes = available_strikes(md, True)            # доступные страйки (call) на сетке
        rows = []                                        # строки профиля по страйкам
        for k in strikes:                                # по каждому страйку…
            g = engine.compute(spot, params, OptionContract("primary", float(k), 0, True), ttm)  # греки call(K)
            d = g.to_dict()                              #   греки в словарь
            d["strike"] = float(k)                       #   добавляем страйк
            d["moneyness"] = float(k / spot)             #   и moneyness (K/spot)
            rows.append(d)                               #   копим строку
        return pd.DataFrame(rows)                        # таблица профиля греков по страйкам
