"""Self-Diagnostic Agent.

Role: assess the system's own adequacy. Monitors data drift (PSI/KS), model
degradation (calibration error & volatility-forecast error), hedge quality
(residual delta/vega) and risk compliance, and condenses them into a single
Confidence Score in [0, 1].

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       9 / 11 — SelfDiagnosticAgent.
НАЗНАЧЕНИЕ:  система «проверяет сама себя». Считает дрейф данных (PSI/KS),
             деградацию модели (ошибка калибровки и прогноза волатильности),
             качество хеджа (остаточные греки) и соблюдение лимитов, и сводит всё
             в единый ИНДЕКС ДОВЕРИЯ [0,1].
ВХОД (consumes):  BACKTEST_READY (от агента 8).
ВЫХОД (produces): DIAGNOSTIC_READY → агенту explainability.
КЛАДЁТ НА ДОСКУ:  diagnostic, confidence_score.
ИМПОРТИРУЕТ:
  - services.drift (как dr) : PSI, KS, ошибки прогноза, индекс доверия.
КОНФИГ:  config.diagnostic (метод/порог дрейфа, веса компонент доверия),
         config.hedging.delta_red_zone, config.investment.capital_usd.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import numpy as np                                       # числовые операции
import pandas as pd                                       # таблицы данных

from cryptohedge.core.agent import BaseAgent             # контракт агента
from cryptohedge.core.context import AgentContext        # контекст
from cryptohedge.core.message import Message, MessageType # сообщения
from cryptohedge.services import drift as dr             # сервис дрейфа/доверия (псевдоним dr)


class SelfDiagnosticAgent(BaseAgent):
    name = "self_diagnostic"                             # имя агента / id этапа
    consumes = [MessageType.BACKTEST_READY]              # принимает BACKTEST_READY
    produces = MessageType.DIAGNOSTIC_READY              # выпускает DIAGNOSTIC_READY
    checkpoint_keys = ["diagnostic", "confidence_score"]  # ключи чекпойнта

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        dcfg = context.config.diagnostic                 # секция конфига диагностики
        returns: pd.DataFrame = context.require("returns")  # доходности (для дрейфа)
        primary = context.require("primary_symbol")      # первичный актив
        calibr: pd.DataFrame = context.require("calibr_data")  # калибровка (ошибка/v0)
        history: pd.DataFrame = context.require("hedge_history")  # история хеджа (остаточные греки)
        stability = context.require("heston_stability")  # устойчивость параметров Хестона
        risk = context.require("risk_assessment")        # оценка риска (соблюдение лимитов)

        r = returns[primary].to_numpy()                  # ряд доходностей первичного актива
        half = len(r) // 2                               # середина ряда (для сравнения распределений)
        psi = dr.population_stability_index(r[:half], r[half:], bins=10)  # PSI между половинами ряда
        ks = dr.ks_drift(r[:half], r[half:])             # тест Колмогорова-Смирнова на дрейф

        # volatility forecast error: Heston daily vol vs realised |return|  # --- ошибка прогноза волатильности ---
        v0 = calibr.sort_values("sample_idx")["v0"].to_numpy()  # ряд начальной дисперсии v0
        ppy = context.config.horizons.trading_days_per_year  # дней в году
        pred_vol = np.sqrt(np.maximum(v0, 0) / ppy)      # предсказанная дневная волатильность (Хестон)
        realised = np.abs(np.concatenate([[0.0], np.diff(np.log(history["spot"].to_numpy()))]))  # реализованная |доходность|
        m = min(len(pred_vol), len(realised))            # длина для сопоставления
        fe = dr.forecast_errors(realised[-m:], pred_vol[-m:])  # ошибки прогноза (RMSE и т.п.)
        mean_realised = float(np.mean(realised[-m:])) or 1e-6  # средняя реализованная волатильность (защита от 0)

        # hedge quality: residual greeks                  # --- качество хеджа: остаточные греки ---
        resid_delta = np.abs(history["delta"] - history["delta_hedge"]).to_numpy()  # |остаточная дельта|
        resid_delta_usd = float(np.mean(resid_delta * history["spot"].to_numpy()))  # средняя остаточная дельта в USD
        resid_frac = resid_delta_usd / context.config.investment.capital_usd  # доля от капитала

        mean_cal_err = float(np.nanmean(calibr.get("calibration_error", pd.Series([np.nan]))))  # средняя ошибка калибровки
        components = {                                    # компоненты индекса доверия (каждая в [0,1])
            "calibration": float(np.clip(1.0 / (1.0 + (0.0 if np.isnan(mean_cal_err) else mean_cal_err))  # качество калибровки
                                         * (0.5 if not stability.get("stable", True) else 1.0), 0, 1)),
            "data_drift": float(np.clip(1.0 - psi / max(dcfg.drift_threshold, 1e-6), 0, 1)),  # стабильность данных
            "forecast_error": float(np.clip(1.0 - fe["rmse"] / mean_realised, 0, 1)),  # точность прогноза
            "hedge_quality": float(np.clip(1.0 - resid_frac / max(context.config.hedging.delta_red_zone, 1e-6), 0, 1)),  # качество хеджа
            "risk_compliance": 1.0 if risk.get("within_limits", False) else 0.3,  # соблюдение лимитов
        }
        confidence = dr.confidence_score(components, dcfg.confidence_weights)  # взвешенный индекс доверия

        diagnostic = {                                   # сводка диагностики
            "psi": psi, "ks": ks, "drift_detected": bool(psi > dcfg.drift_threshold or ks["pvalue"] < 0.05),  # дрейф?
            "forecast_error": fe, "stability": stability,  # ошибки прогноза и устойчивость
            "residual_delta_usd": resid_delta_usd, "residual_delta_fraction": resid_frac,  # остаточная дельта
            "components": components, "confidence_score": confidence,  # компоненты и итоговый индекс
            "self_assessment": self._label(confidence),  # текстовая метка уверенности
        }

        pd.Series({k: str(v) for k, v in diagnostic.items()}).to_json(  # сохраняем диагностику на диск
            context.results_path("diagnostic.json"))
        context.put("diagnostic", diagnostic)            # диагностика → на доску
        context.put("confidence_score", confidence)      # индекс доверия → на доску

        log.decision("self-diagnostic", confidence=round(confidence, 3),  # лог-итог диагностики
                     drift_detected=diagnostic["drift_detected"], assessment=diagnostic["self_assessment"],
                     components={k: round(v, 3) for k, v in components.items()})

        return Message(self.produces, self.name, "explainability",  # DIAGNOSTIC_READY следующему агенту
                       payload={"confidence": confidence, "assessment": diagnostic["self_assessment"]},
                       correlation_id=message.correlation_id)

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.75:                                # высокий уровень доверия
            return "high_confidence"
        if score >= 0.5:                                 # умеренный
            return "moderate_confidence"
        if score >= 0.3:                                 # низкий
            return "low_confidence"
        return "unreliable"                              # ненадёжно
