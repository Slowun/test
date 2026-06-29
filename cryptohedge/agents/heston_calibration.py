"""Heston Calibration Agent.

Role: calibrate the Heston parameters at every time slice (implied-volatility
surface least squares), maintain a maximum-likelihood calibration on the spot
time series, monitor parameter stability over time and benchmark Heston against
Black-Scholes and SABR. All intermediate calibration artefacts are persisted.

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       3 / 11 — HestonCalibrationAgent.
НАЗНАЧЕНИЕ:  калибрует параметры Хестона (v0,κ,θ,ε,ρ) на КАЖДОМ временно́м срезе
             по поверхности implied-volatility (МНК), плюс ведёт MLE-калибровку по
             ряду спота, контролирует устойчивость параметров и сравнивает Хестон
             с бенчмарками Black-Scholes и SABR. Все промежуточные артефакты пишет на диск.
ВХОД (consumes):  ANALYSIS_READY (от агента 2).
ВЫХОД (produces): CALIBRATION_READY → агенту greeks_calculation.
КЛАДЁТ НА ДОСКУ:  calibr_data, heston_history, heston_stability, heston_benchmarks, heston_mle.
ИМПОРТИРУЕТ:
  - domain.market.HestonParameters       : структура параметров.
  - services.calibration (как cal)        : MLE, устойчивость, BS/SABR-бенчмарки.
  - services.heston_pricing.*             : implied vol, калибровка по IV, цены Хестона.
  - services.providers.base.INSTR_*       : константы типов инструментов.
КОНФИГ:  config.heston (метод/итерации/нач.параметры/устойчивость/бенчмарки),
         config.hedging.calibration_subsample (шаг калибровки).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import List, Optional                        # аннотации

import numpy as np                                       # числовые операции (RMSE, медиана)
import pandas as pd                                       # таблицы срезов/калибровки

from cryptohedge.core.agent import BaseAgent             # контракт агента
from cryptohedge.core.context import AgentContext        # контекст
from cryptohedge.core.message import Message, MessageType # сообщения
from cryptohedge.domain.market import HestonParameters   # структура параметров Хестона
from cryptohedge.services import calibration as cal      # сервис калибровки (псевдоним cal)
from cryptohedge.services.heston_pricing import bs_implied_vol, calibrate_iv_surface, heston_premiums  # ценообразование
from cryptohedge.services.providers.base import INSTR_ASSET, INSTR_CALL, INSTR_PUT  # константы типов инструментов


class HestonCalibrationAgent(BaseAgent):
    name = "heston_calibration"                          # имя агента / id этапа
    consumes = [MessageType.ANALYSIS_READY]              # принимает ANALYSIS_READY
    produces = MessageType.CALIBRATION_READY             # выпускает CALIBRATION_READY
    checkpoint_keys = ["calibr_data", "heston_history", "heston_stability",  # ключи чекпойнта
                       "heston_benchmarks", "heston_mle"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        hcfg = context.config.heston                     # секция конфига Хестона
        md: pd.DataFrame = context.require("market_data")  # опционные рыночные данные
        spot_close = context.require("spot_close")       # цены закрытия спота
        primary = context.require("primary_symbol")      # первичный актив

        # ---- MLE on the spot time series (filtering + maximum likelihood)  # --- MLE по ряду спота ---
        with log.timer("mle"):                           # замеряем время MLE
            mle = cal.calibrate_mle(                      # калибровка параметров методом макс. правдоподобия
                spot_close[primary].to_numpy(),          #   ряд цен первичного актива
                dt=1.0 / context.config.horizons.trading_days_per_year,  #   шаг времени (1 день в годах)
                flat_yield=hcfg.flat_yield_fallback,     #   безрисковая ставка
                trading_days=context.config.horizons.trading_days_per_year,  #   дней в году
            )
        log.decision("MLE calibration", v0=round(mle.v0, 5), kappa=round(mle.kappa, 3),  # лог результатов MLE
                     theta=round(mle.theta, 5), eps=round(mle.eps, 3), rho=round(mle.rho, 3))

        # ---- per-slice IV-surface calibration           # --- калибровка по IV на каждом срезе ---
        samples = sorted(md["sample_idx"].unique())      # упорядоченные индексы временны́х срезов
        subsample = max(1, context.config.hedging.calibration_subsample)  # шаг калибровки (>=1)
        init = list(hcfg.initial_params)                 # стартовые параметры (обновляются по ходу)
        history: List[HestonParameters] = []             # история параметров по срезам
        records = []                                     # строки для таблицы calibr_data
        last: Optional[HestonParameters] = None          # параметры предыдущего успешного среза
        n_failed = 0                                     # счётчик неудачных калибровок

        with log.timer("per_slice_calibration", n=len(samples)):  # замеряем общее время
            for k, sidx in enumerate(samples):           # по каждому срезу…
                grp = md[md["sample_idx"] == sidx]       #   данные этого среза
                ts = grp["timestamp"].iloc[0]            #   временна́я метка среза
                spot = float(grp[grp["instrument_type"] == INSTR_ASSET]["price"].iloc[0])  # спот на срезе

                if hcfg.calibration_method == "mle":     # режим MLE: используем единые MLE-параметры…
                    params = HestonParameters(v0=mle.v0, kappa=mle.kappa, theta=mle.theta,
                                              eps=mle.eps, rho=mle.rho, flat_yield=mle.flat_yield)
                elif k % subsample != 0 and last is not None:  # пропускаемый срез (subsample): берём прошлые…
                    params = last
                else:                                    # иначе калибруем срез по IV-поверхности
                    params = self._calibrate_slice(grp, spot, init, hcfg, last, mle)

                if params is None:                       # если калибровка не удалась…
                    n_failed += 1                        #   считаем неудачу
                    params = last or mle                 #   откатываемся к прошлым или MLE
                last = params                            # запоминаем как последние успешные
                init = list(params.as_array())           # следующий срез стартует с текущих параметров
                history.append(params)                   # сохраняем в историю
                records.append({                         # строка таблицы calibr_data
                    "sample_idx": int(sidx), "timestamp": ts,
                    "v0": params.v0, "kappa": params.kappa, "theta": params.theta,
                    "eps": params.eps, "rho": params.rho, "flat_yield": params.flat_yield,
                    "calibration_error": params.calibration_error,
                })

        calibr_data = pd.DataFrame(records)              # таблица параметров по срезам

        stability = cal.parameter_stability(history, hcfg.stability_max_rel_change)  # анализ устойчивости параметров
        benchmarks = self._benchmarks(md, samples, history, hcfg)  # сравнение Heston vs BS vs SABR

        # ---- persist intermediate calibration artefacts  # --- сохраняем промежуточные артефакты на диск ---
        calibr_data.to_parquet(context.calibration_path("calibr_data.parquet"))  # таблица параметров → parquet
        pd.Series(stability).to_json(context.calibration_path("heston_stability.json"))  # устойчивость → json
        pd.Series({k: str(v) for k, v in benchmarks.items()}).to_json(  # бенчмарки → json
            context.calibration_path("heston_benchmarks.json"))

        context.put("calibr_data", calibr_data)          # таблица калибровки → на доску
        context.put("heston_history", history)           # история параметров → на доску
        context.put("heston_stability", stability)       # устойчивость → на доску
        context.put("heston_benchmarks", benchmarks)     # бенчмарки → на доску
        context.put("heston_mle", mle)                   # MLE-параметры → на доску

        log.decision("calibration complete", n_slices=len(samples), n_failed=n_failed,  # лог итога калибровки
                     stable=stability["stable"], max_rel_change=round(stability["max_rel_change"], 3),
                     heston_iv_rmse=round(benchmarks.get("heston", {}).get("iv_rmse", float("nan")), 5),
                     bs_iv_rmse=round(benchmarks.get("black_scholes", {}).get("iv_rmse", float("nan")), 5),
                     sabr_iv_rmse=round(benchmarks.get("sabr", {}).get("rmse", float("nan")), 5))

        return Message(self.produces, self.name, "greeks_calculation",  # CALIBRATION_READY следующему агенту
                       payload={"n_slices": len(samples), "stable": stability["stable"]},
                       correlation_id=message.correlation_id)

    # ------------------------------------------------------------------ helpers
    def _slice_chain(self, grp: pd.DataFrame, spot: float):
        opts = grp[grp["instrument_type"].isin([INSTR_CALL, INSTR_PUT])]  # только опционы (call/put) среза
        strikes = opts["strike"].to_numpy(float)         # массив страйков
        ttm = opts["time_to_maturity"].to_numpy(float)   # массив сроков до экспирации
        is_call = (opts["instrument_type"] == INSTR_CALL).to_numpy()  # флаги call/put
        premiums_usd = opts["price"].to_numpy(float) * spot  # coin-quoted -> USD  # цены опционов в USD
        return strikes, ttm, is_call, premiums_usd       # компоненты опционной цепочки среза

    def _calibrate_slice(self, grp, spot, init, hcfg, last, mle) -> Optional[HestonParameters]:
        strikes, ttm, is_call, premiums_usd = self._slice_chain(grp, spot)  # извлекаем цепочку среза
        if len(strikes) < 6:                             # слишком мало точек для калибровки…
            return None                                  #   → неудача
        try:
            return calibrate_iv_surface(                 # МНК-калибровка по поверхности implied vol
                spot, strikes, ttm, is_call, premiums_usd, flat_yield=hcfg.flat_yield_fallback,
                init_params=init, num_iter=hcfg.num_iter, tol=hcfg.tol,
            )
        except Exception:                                # любая ошибка калибровки…
            return None                                  #   → неудача (откат выше по коду)

    def _benchmarks(self, md, samples, history, hcfg) -> dict:
        """Compare Heston, Black-Scholes and SABR on a representative mid slice."""
        mid = samples[len(samples) // 2]                 # «средний» репрезентативный срез
        grp = md[md["sample_idx"] == mid]                # данные этого среза
        spot = float(grp[grp["instrument_type"] == INSTR_ASSET]["price"].iloc[0])  # спот на срезе
        strikes, ttm, is_call, premiums_usd = self._slice_chain(grp, spot)  # цепочка среза
        if len(strikes) < 6:                             # мало точек → пустые бенчмарки
            return {"heston": {}, "black_scholes": {}, "sabr": {}}

        T = float(np.median(ttm))                        # репрезентативный срок (медиана TTM)
        market_iv = np.array([bs_implied_vol(spot, k, t, hcfg.flat_yield_fallback, p, bool(c))  # рыночная IV из цен
                              for k, t, p, c in zip(strikes, ttm, premiums_usd, is_call)])

        result = {}                                      # копим результаты бенчмарков
        params = history[len(history) // 2]              # параметры Хестона на среднем срезе
        heston_prices = heston_premiums(spot, strikes, ttm, is_call, params)  # цены опционов по Хестону
        heston_iv = np.array([bs_implied_vol(spot, k, t, hcfg.flat_yield_fallback, p, bool(c))  # IV из цен Хестона
                              for k, t, p, c in zip(strikes, ttm, heston_prices, is_call)])
        mask = np.isfinite(market_iv) & np.isfinite(heston_iv)  # валидные точки (обе IV конечны)
        result["heston"] = {                             # метрика качества Хестона
            "iv_rmse": float(np.sqrt(np.nanmean((heston_iv[mask] - market_iv[mask]) ** 2))) if mask.any() else float("nan"),  # RMSE по IV
            "params": params.to_dict(),                  #   сами параметры
        }
        if "black_scholes" in hcfg.benchmarks:           # если включён бенчмарк BS…
            result["black_scholes"] = cal.black_scholes_benchmark(  # считаем BS-бенчмарк
                spot, strikes, T, premiums_usd, is_call, hcfg.flat_yield_fallback)
        if "sabr" in hcfg.benchmarks:                    # если включён бенчмарк SABR…
            forward = spot * np.exp(hcfg.flat_yield_fallback * T)  # форвардная цена
            sabr = cal.sabr_calibrate(forward, strikes, T, market_iv, beta=hcfg.sabr_beta)  # калибровка SABR по IV
            result["sabr"] = sabr.to_dict()              # результат SABR
        return result                                    # словарь бенчмарков
