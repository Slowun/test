"""Heston delta-vega hedging engine.

A faithful, config-driven refactor of ``heston_greeks_hedging.ipynb``: it walks a
time series of spot/option quotes and per-slice Heston parameters, computes the
greeks of the liability option portfolio, and dynamically neutralises its delta
(with spot) and vega (with a hedging option), accounting for transaction fees.

It returns the full set of required outputs - spot, strategy PnL, fees paid,
delta-hedge, vega-hedge, spot/option positions, option-portfolio premium and the
portfolio delta/vega (plus gamma, theta, rho, charm for monitoring).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (движок динамического хеджирования).
НАЗНАЧЕНИЕ:  идёт по временному ряду котировок спота/опционов и параметров Хестона
             на каждом срезе, считает греки портфеля обязательств и динамически
             нейтрализует дельту (спотом) и вегу (хедж-опционом) с учётом комиссий.
             Возвращает полную историю: спот, PnL, комиссии, Δ/ν-хедж, позиции,
             премию портфеля и греки (Δ Γ ν Θ ρ charm) для мониторинга.
ИМПОРТИРУЕТ: math, dataclass; numpy, pandas; domain.decisions.HedgeDecision;
             domain.greeks.Greeks; domain.market.{HestonParameters,OptionContract};
             services.greeks.HestonGreeksEngine; base.{INSTR_ASSET,INSTR_CALL,
             INSTR_PUT}.
КОНСТАНТЫ:   YEAR_NANOS — наносекунд в году (для перевода срока в годы).
ЭКСПОРТИРУЕТ: FeeModel, StrategyConfig, HedgingResult, HedgingEngine.
КЕМ ИСПОЛЬЗУЕТСЯ: агенты hedging_decision и backtesting.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import math                                              # математика (резерв)
from dataclasses import dataclass, field                 # для конфигов/результата
from typing import Dict, List, Optional, Tuple           # аннотации

import numpy as np                                       # вычисления
import pandas as pd                                       # таблицы

from cryptohedge.domain.decisions import HedgeDecision   # решение о хедже
from cryptohedge.domain.greeks import Greeks             # греки инструмента/портфеля
from cryptohedge.domain.market import HestonParameters, OptionContract  # параметры и контракт
from cryptohedge.services.greeks import HestonGreeksEngine  # движок расчёта греков
from cryptohedge.services.providers.base import INSTR_ASSET, INSTR_CALL, INSTR_PUT  # коды инструментов

YEAR_NANOS = 31_536_000_000_000_000                      # наносекунд в году (365 дней)


@dataclass
class FeeModel:                                           # модель комиссий
    spot_fee_pct: float = 0.0003                         # комиссия по споту (доля)
    option_fee_pct: float = 0.0003                       # комиссия по опциону (доля)
    option_fee_cap_pct: float = 0.125                    # потолок комиссии опциона (доля цены)


@dataclass
class StrategyConfig:                                     # допуски стратегии хеджа
    delta_eps: float = 0.0                               # порог нейтрализации дельты
    vega_eps: float = 0.0                                # порог нейтрализации веги


@dataclass
class HedgingResult:                                      # результат прогона движка
    history: pd.DataFrame                                # история по срезам
    trades: List[dict] = field(default_factory=list)     # список сделок
    decisions: List[HedgeDecision] = field(default_factory=list)  # список решений


class HedgingEngine:
    def __init__(self, greeks_engine: HestonGreeksEngine, fees: FeeModel, strategy: StrategyConfig) -> None:
        self.greeks = greeks_engine                      # движок греков
        self.fees = fees                                 # модель комиссий
        self.strategy = strategy                         # допуски стратегии

    # -------------------------------------------------------------- data access
    @staticmethod
    def _asset_row(md_slice: pd.DataFrame) -> pd.Series:
        rows = md_slice[md_slice["instrument_type"] == INSTR_ASSET]  # строки базового актива
        if len(rows) != 1:                              # должна быть ровно одна…
            raise ValueError("Expected exactly one ASSET row per sample")
        return rows.iloc[0]                             # строка спота

    @staticmethod
    def _option_row(md_slice: pd.DataFrame, contract: OptionContract) -> Optional[pd.Series]:
        instr = INSTR_CALL if contract.is_call else INSTR_PUT  # тип опциона
        rows = md_slice[                                # ищем строку по типу/страйку/экспирации:
            (md_slice["instrument_type"] == instr)
            & (np.isclose(md_slice["strike"], contract.strike))
            & (md_slice["expiry_ts"].astype("int64") == contract.expiry_ts)
        ]
        return rows.iloc[0] if len(rows) else None      # котировка опциона или None

    @staticmethod
    def _params(cal_row: pd.Series) -> HestonParameters:
        return HestonParameters(                        # параметры Хестона из строки калибровки:
            v0=float(cal_row["v0"]), kappa=float(cal_row["kappa"]), theta=float(cal_row["theta"]),
            eps=float(cal_row["eps"]), rho=float(cal_row["rho"]),
            flat_yield=float(cal_row.get("flat_yield", 0.0)),
        )

    # ------------------------------------------------------------------- greeks
    def _portfolio_greeks(self, spot, params, contracts, ts) -> Greeks:
        total = Greeks()                                # нулевые греки
        for c in contracts:                             # по каждому контракту обязательств…
            ttm = (c.expiry_ts - ts) / YEAR_NANOS       #   срок до экспирации (годы)
            if ttm <= 0:                                #   истёк…
                continue
            total = total + self.greeks.compute(spot, params, c, ttm)  #   суммируем греки
        return total                                    # греки портфеля обязательств

    # --------------------------------------------------------------------- run
    def run(
        self,
        market_data: pd.DataFrame,                      # рыночные данные (спот + опционы)
        calibr_data: pd.DataFrame,                      # калиброванные параметры по срезам
        portfolio_contracts: List[OptionContract],      # портфель обязательств (путы)
        vega_option: OptionContract,                    # опцион для вега-хеджа
        sample_indices: Optional[List[int]] = None,     # подмножество срезов (опц.)
    ) -> HedgingResult:
        calibr_data = calibr_data.sort_values("sample_idx").reset_index(drop=True)  # сортируем по срезу
        if sample_indices is not None:                  # если задано подмножество…
            calibr_data = calibr_data[calibr_data["sample_idx"].isin(sample_indices)].reset_index(drop=True)  # фильтр

        md_by_sample = {idx: grp for idx, grp in market_data.groupby("sample_idx")}  # данные по срезам

        pos = {"spot": 0.0, "vega_opt": 0.0}            # позиции: спот и хедж-опцион
        pos_quote = 0.0                                 # денежный счёт (котируемая валюта)
        fee_paid = 0.0                                  # накопленные комиссии
        hedge_delta = 0.0                               # дельта хеджа
        hedge_vega = 0.0                                # вега хеджа
        portf_init_premium: Optional[float] = None      # начальная премия портфеля обязательств

        rows: List[dict] = []                           # строки истории
        trades: List[dict] = []                         # сделки
        decisions: List[HedgeDecision] = []             # решения

        for _, cal_row in calibr_data.iterrows():       # по каждому срезу времени…
            sample_idx = int(cal_row["sample_idx"])     #   индекс среза
            if sample_idx not in md_by_sample:          #   нет рыночных данных…
                continue
            md_slice = md_by_sample[sample_idx]         #   данные среза
            asset = self._asset_row(md_slice)           #   строка спота
            ts = int(pd.Timestamp(cal_row["timestamp"]).value)  #   метка времени (нс)
            spot = float(asset["price"])                #   спот (mid)
            spot_bid = float(asset["best_bid_price"])   #   bid спота
            spot_ask = float(asset["best_ask_price"])   #   ask спота
            params = self._params(cal_row)              #   параметры Хестона среза

            portf = self._portfolio_greeks(spot, params, portfolio_contracts, ts)  # греки обязательств
            if portf_init_premium is None:              #   при первом срезе…
                portf_init_premium = portf.premium      #     запоминаем начальную премию

            vo_row = self._option_row(md_slice, vega_option)  # котировка хедж-опциона
            vo_ttm = (vega_option.expiry_ts - ts) / YEAR_NANOS  # его срок (годы)
            if vo_row is None or vo_ttm <= 0:           #   нет котировки/истёк…
                vo_greeks = Greeks()                    #     нулевые греки
            else:
                vo_greeks = self.greeks.compute(spot, params, vega_option, vo_ttm)  # греки хедж-опциона

            # ---- hedge vega with the option, then delta with spot  # сначала вега, затем дельта:
            self._recompute_hedge(pos, vo_greeks)       #   (зарезервировано для расширения)
            hedge_vega = pos["vega_opt"] * vo_greeks.vega  #   текущая вега хеджа
            hedge_delta = pos["spot"] + pos["vega_opt"] * vo_greeks.delta  #   текущая дельта хеджа

            diff_vega = portf.vega - hedge_vega         #   рассогласование по веге
            if vo_row is not None and abs(vo_greeks.vega) > 1e-12 and abs(diff_vega) > self.strategy.vega_eps:  # нужна корректировка веги
                amount = diff_vega / vo_greeks.vega     #     требуемое число опционов
                side = "buy" if amount > 0 else "sell"  #     сторона сделки
                t = self._trade("vega_opt", side, abs(amount), spot, spot_bid, spot_ask, vo_row, pos)  # исполняем
                pos_quote += t["cash"]                  #     обновляем кэш
                fee_paid += t["fee"]                    #     накапливаем комиссию
                trades.append(t)                        #     лог сделки
                decisions.append(HedgeDecision(         #     лог решения (вега):
                    timestamp=ts, instrument="vega_option", side=side, quantity=abs(amount),
                    target_greek="vega", pre_hedge_value=hedge_vega, post_hedge_value=portf.vega,
                    rationale="Neutralise portfolio vega via hedging option",
                    metrics={"portfolio_vega": portf.vega, "option_vega": vo_greeks.vega},
                ))
                self._recompute_hedge(pos, vo_greeks)   #     пересчёт хеджа
                hedge_vega = pos["vega_opt"] * vo_greeks.vega  #     новая вега хеджа
                hedge_delta = pos["spot"] + pos["vega_opt"] * vo_greeks.delta  #     новая дельта хеджа

            diff_delta = portf.delta - hedge_delta      #   рассогласование по дельте
            if abs(diff_delta) > self.strategy.delta_eps:  # нужна корректировка дельты
                side = "buy" if diff_delta > 0 else "sell"  #     сторона сделки
                t = self._trade("spot", side, abs(diff_delta), spot, spot_bid, spot_ask, None, pos)  # торгуем спотом
                pos_quote += t["cash"]                  #     обновляем кэш
                fee_paid += t["fee"]                    #     накапливаем комиссию
                trades.append(t)                        #     лог сделки
                decisions.append(HedgeDecision(         #     лог решения (дельта):
                    timestamp=ts, instrument="spot", side=side, quantity=abs(diff_delta),
                    target_greek="delta", pre_hedge_value=hedge_delta, post_hedge_value=portf.delta,
                    rationale="Neutralise portfolio delta via spot",
                    metrics={"portfolio_delta": portf.delta},
                ))
                self._recompute_hedge(pos, vo_greeks)   #     пересчёт хеджа
                hedge_delta = pos["spot"] + pos["vega_opt"] * vo_greeks.delta  #     новая дельта хеджа

            net_worth = self._mark_to_market(           #   переоценка позиции (mark-to-market):
                portf_init_premium, portf.premium, pos, pos_quote, spot, spot_bid, spot_ask, vo_row
            )

            rows.append({                               #   строка истории по срезу:
                "ts": pd.to_datetime(ts), "sample_idx": sample_idx, "spot": spot,
                "premium": portf.premium, "delta": portf.delta, "gamma": portf.gamma,
                "vega": portf.vega, "theta": portf.theta, "rho": portf.rho, "charm": portf.charm,
                "delta_hedge": hedge_delta, "vega_hedge": hedge_vega,
                "vega_option_premium": vo_greeks.premium, "vega_option_delta": vo_greeks.delta,
                "vega_option_vega": vo_greeks.vega,
                "pos_spot": pos["spot"], "pos_vega_option": pos["vega_opt"], "pos_usd": pos_quote,
                "fee": fee_paid, "pnl": net_worth,
            })

        history = pd.DataFrame(rows)                    # история → DataFrame
        return HedgingResult(history=history, trades=trades, decisions=decisions)  # результат прогона

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _recompute_hedge(pos: Dict[str, float], vo_greeks: Greeks) -> None:
        # kept for symmetry / future extension (greeks recomputed by caller)
        return None                                     # заглушка (греки пересчитывает вызывающий код)

    def _trade(self, instr, side, amount, spot, spot_bid, spot_ask, opt_row, pos) -> dict:
        sign = 1.0 if side == "buy" else -1.0           # знак сделки (+покупка / -продажа)
        fee = 0.0                                       # комиссия
        if instr == "spot":                             # сделка по споту:
            price = spot_ask if sign > 0 else spot_bid  #   исполнение по ask/bid
            fee = self.fees.spot_fee_pct * amount * price  #   комиссия спота
        else:                                           # сделка по опциону:
            spot_mid = 0.5 * (spot_bid + spot_ask)      #   mid спота
            coin_price = float(opt_row["best_ask_price"] if sign > 0 else opt_row["best_bid_price"])  # цена в монетах
            price = coin_price * spot_mid               #   цена в котируемой валюте
            fee = min(amount * spot_mid * self.fees.option_fee_pct, self.fees.option_fee_cap_pct * price)  # комиссия с потолком
        key = "spot" if instr == "spot" else "vega_opt"  # ключ позиции
        pos[key] += sign * amount                       # обновляем позицию
        cash = -sign * amount * price - fee             # денежный поток (− стоимость − комиссия)
        return {"instrument": instr, "side": side, "quantity": amount, "price": price, "fee": fee, "cash": cash}  # лог сделки

    def _mark_to_market(self, init_premium, premium, pos, pos_quote, spot, spot_bid, spot_ask, opt_row) -> float:
        net = init_premium - premium + pos_quote        # P&L обязательств + денежный счёт
        # liquidate spot                                 # ликвидация спот-позиции:
        if pos["spot"] != 0:
            liq = spot_bid if pos["spot"] > 0 else spot_ask  #   продаём по bid / покупаем по ask
            net += pos["spot"] * liq
        # liquidate hedging option                       # ликвидация позиции в хедж-опционе:
        if pos["vega_opt"] != 0 and opt_row is not None:
            spot_mid = 0.5 * (spot_bid + spot_ask)
            coin = float(opt_row["best_bid_price"] if pos["vega_opt"] > 0 else opt_row["best_ask_price"])
            net += pos["vega_opt"] * coin * spot_mid
        return float(net)                               # чистая стоимость позиции
