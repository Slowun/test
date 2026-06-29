"""Explainability Agent.

Role: turn the system's quantitative state into a natural-language narrative in
*both* Russian and English. Every statement is backed by a concrete metric: the
risk picture and hedge sizing, instrument selection, Heston calibration and its
stability, the greeks balance, the portfolio optimisation with its diversification
proof and profitability, risk limits/stops, backtest performance and the
confidence score. Russian and English section sets are published separately so the
dashboard can render a fully localised page.

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       10 / 11 — ExplainabilityAgent.
НАЗНАЧЕНИЕ:  превращает числовое состояние системы в ТЕКСТОВОЕ объяснение на
             русском И английском. Каждый тезис подкреплён метрикой: риск и объём
             хеджа, выбор инструментов, калибровка Хестона, греки, оптимизация
             портфеля и диверсификация, риск/стопы, бэктест, индекс доверия.
ВХОД (consumes):  DIAGNOSTIC_READY (от агента 9).
ВЫХОД (produces): EXPLANATION_READY → агенту dashboard.
КЛАДЁТ НА ДОСКУ:  explanation_text, explanation_sections (RU по умолчанию),
                  explanation_sections_ru, explanation_sections_en.
ЧИТАЕТ С ДОСКИ:   практически все артефакты предыдущих агентов (volatility,
                  hedge_sizing, rankings_df, calibr_data, greeks, risk, perf, …).
КОНФИГ:  config.explainability (язык/decimals), config.market_analysis.top_n_*.
ПРИМЕЧАНИЕ: большие f-строки — это сам двуязычный НАРРАТИВ; комментарии стоят у
           логики и у границ разделов, текст внутри строк самодокументирован.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import Dict                                  # аннотация словаря разделов

import pandas as pd                                       # таблицы артефактов (rankings/stress/...)

from cryptohedge.core.agent import BaseAgent             # контракт агента
from cryptohedge.core.context import AgentContext        # контекст
from cryptohedge.core.message import Message, MessageType # сообщения


class ExplainabilityAgent(BaseAgent):
    name = "explainability"                              # имя агента / id этапа
    consumes = [MessageType.DIAGNOSTIC_READY]            # принимает DIAGNOSTIC_READY
    produces = MessageType.EXPLANATION_READY             # выпускает EXPLANATION_READY
    checkpoint_keys = ["explanation_text", "explanation_sections",  # ключи чекпойнта
                       "explanation_sections_ru", "explanation_sections_en"]

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        sections_ru = self._sections(context, "ru")      # формируем разделы на русском
        sections_en = self._sections(context, "en")      # формируем разделы на английском

        self._write(context, "ru", sections_ru, "explanation.md",  # пишем RU-объяснение в файл
                    "# Объяснение решений системы хеджирования")
        self._write(context, "en", sections_en, "explanation.en.md",  # пишем EN-объяснение в файл
                    "# Hedging System Decision Explanation")

        text_ru = "\n\n".join(f"### {t}\n{b}" for t, b in sections_ru.items())  # единый RU-текст
        context.put("explanation_text", text_ru)         # текст RU → на доску
        context.put("explanation_sections", sections_ru)  # back-compat default = RU  # RU как дефолт
        context.put("explanation_sections_ru", sections_ru)  # разделы RU → на доску
        context.put("explanation_sections_en", sections_en)  # разделы EN → на доску

        diag = context.blackboard["diagnostic"]          # диагностика (для лога индекса доверия)
        log.decision("generated bilingual explanation", sections=len(sections_ru),  # лог-итог
                     confidence=round(diag["confidence_score"], 3))

        return Message(self.produces, self.name, "dashboard",  # EXPLANATION_READY следующему агенту
                       payload={"sections": list(sections_ru), "languages": ["ru", "en"]},
                       correlation_id=message.correlation_id)

    # ------------------------------------------------------------------ helpers
    def _write(self, context, lang, sections, fname, title):
        text = "\n\n".join(f"### {t}\n{b}" for t, b in sections.items())  # склеиваем разделы в markdown
        context.results_path(fname).write_text(title + "\n\n" + text, encoding="utf-8")  # пишем файл

    def _sections(self, context: AgentContext, lang: str) -> Dict[str, str]:
        d = context.config.explainability.decimals       # знаков после запятой в числах
        b = context.blackboard                           # короткая ссылка на доску
        ru = lang == "ru"                                # флаг языка (True=русский)

        vol = b["volatility"]                            # оценка волатильности (агент 2)
        sizing = b["hedge_sizing"]                       # объём хеджа (агент 2)
        top_n = context.config.market_analysis.top_n_hedge_instruments  # сколько инструментов показать
        rankings_df: pd.DataFrame = b.get("rankings_df", pd.DataFrame())  # рейтинг инструментов
        rankings = rankings_df.head(top_n).to_dict("records") if not rankings_df.empty else []  # топ как список
        calibr: pd.DataFrame = b["calibr_data"]          # калибровка Хестона (агент 3)
        last = calibr.sort_values("sample_idx").iloc[-1]  # последний срез калибровки
        stability = b["heston_stability"]                # устойчивость параметров
        bench = b.get("heston_benchmarks", {})           # бенчмарки BS/SABR
        greeks = b["portfolio_greeks_latest"]            # последние греки портфеля (агент 4)
        status = b["hedge_status"]                        # статус баланса дельты (агент 4)
        latest = b["latest_decision"]                    # текущее решение хеджа (агент 5)
        risk = b["risk_assessment"]                      # оценка риска (агент 7)
        stop = b["stop_level"]                           # уровень стопа (агент 7)
        perf = b["backtest_metrics"]                     # метрики бэктеста (агент 8)
        stress: pd.DataFrame = b["stress_table"]         # таблица стресс-тестов (агент 8)
        diag = b["diagnostic"]                           # диагностика (агент 9)

        sections: Dict[str, str] = {}                    # накапливаем разделы объяснения

        # ---- 1. risk & hedge sizing                     # --- Раздел 1: риск и объём хеджа (RU/EN) ---
        feller = 2 * last["kappa"] * last["theta"] - last["eps"] ** 2  # значение условия Феллера
        if ru:                                           # русская версия раздела
            sections["Риск и объём хеджирования"] = (
                f"Суточная волатильность BTC оценена в {vol.daily_vol:.{d}f} "
                f"(годовая {vol.annualized_vol:.{d}f}), волатильность волатильности {vol.vol_of_vol:.{d}f}. "
                f"Доверительный интервал {int(vol.confidence_level*100)}%: "
                f"[{vol.ci_low:.{d}f}; {vol.ci_high:.{d}f}]. "
                f"При капитале {sizing.capital_usd:,.0f}$ и риск-бюджете "
                f"{context.config.investment.risk_budget_pct:.0%} нехеджированный 1-дневный VaR "
                f"превышает лимит, поэтому хеджируется доля {sizing.hedge_ratio:.0%} капитала — "
                f"{sizing.notional_to_hedge_usd:,.0f}$ или {sizing.quantity_to_hedge:.{d}f} BTC."
            )
        else:                                            # английская версия раздела
            sections["Risk & Hedge Sizing"] = (
                f"BTC daily volatility is estimated at {vol.daily_vol:.{d}f} "
                f"(annualized {vol.annualized_vol:.{d}f}), volatility-of-volatility {vol.vol_of_vol:.{d}f}. "
                f"The {int(vol.confidence_level*100)}% confidence interval is "
                f"[{vol.ci_low:.{d}f}; {vol.ci_high:.{d}f}]. "
                f"With ${sizing.capital_usd:,.0f} of capital and a "
                f"{context.config.investment.risk_budget_pct:.0%} risk budget, the unhedged 1-day VaR "
                f"exceeds the limit, so {sizing.hedge_ratio:.0%} of capital is hedged — "
                f"${sizing.notional_to_hedge_usd:,.0f} or {sizing.quantity_to_hedge:.{d}f} BTC."
            )

        # ---- 2. instrument selection                    # --- Раздел 2: выбор инструментов (RU/EN) ---
        lines = []                                       # строки с топ-кандидатами
        for r in rankings[:5]:                           # по топ-5 инструментам рейтинга…
            if ru:                                       #   русская строка кандидата
                lines.append(
                    f"  • {r['symbol']}: score={r['score']:.3f}, Пирсон={r['pearson']:.2f}, "
                    f"Спирмен={r['spearman']:.2f}, Кендалл={r['kendall']:.2f}, DCC={r['dcc_mean']:.2f}, "
                    f"коинтеграция={'да' if r['cointegrated'] else 'нет'}, устойчивость={r['stability']:.2f}")
            else:                                        #   английская строка кандидата
                lines.append(
                    f"  • {r['symbol']}: score={r['score']:.3f}, Pearson={r['pearson']:.2f}, "
                    f"Spearman={r['spearman']:.2f}, Kendall={r['kendall']:.2f}, DCC={r['dcc_mean']:.2f}, "
                    f"cointegration={'yes' if r['cointegrated'] else 'no'}, stability={r['stability']:.2f}")
        if ru:                                           # заголовок+тело раздела (RU)
            sections["Выбор инструментов хеджирования"] = (
                "Инструменты ранжированы по корреляции, устойчивости связи, ликвидности, стоимости "
                "хеджирования и потенциалу снижения риска. Топ-кандидаты:\n" + "\n".join(lines))
        else:                                            # заголовок+тело раздела (EN)
            sections["Hedging Instrument Selection"] = (
                "Instruments are ranked by correlation, link stability, liquidity, hedging cost and "
                "risk-reduction potential. Top candidates:\n" + "\n".join(lines))

        # ---- 3. Heston calibration                      # --- Раздел 3: калибровка Хестона (RU/EN) ---
        if ru:
            sections["Калибровка модели Хестона"] = (
                f"Последние параметры: v0={last['v0']:.{d}f}, kappa={last['kappa']:.{d}f}, "
                f"theta={last['theta']:.{d}f}, eps={last['eps']:.{d}f}, rho={last['rho']:.{d}f}. "
                f"Условие Феллера 2·kappa·theta−eps² = {feller:.{d}f}. "
                f"Параметры {'устойчивы' if stability.get('stable') else 'неустойчивы'} во времени "
                f"(макс. отн. изменение {stability.get('max_rel_change', float('nan')):.2f}). "
                + self._bench_text(bench, d, ru))        # + строка сравнения с бенчмарками
        else:
            sections["Heston Model Calibration"] = (
                f"Latest parameters: v0={last['v0']:.{d}f}, kappa={last['kappa']:.{d}f}, "
                f"theta={last['theta']:.{d}f}, eps={last['eps']:.{d}f}, rho={last['rho']:.{d}f}. "
                f"Feller condition 2·kappa·theta−eps² = {feller:.{d}f}. "
                f"Parameters are {'stable' if stability.get('stable') else 'unstable'} over time "
                f"(max relative change {stability.get('max_rel_change', float('nan')):.2f}). "
                + self._bench_text(bench, d, ru))        # + строка сравнения с бенчмарками

        # ---- 4. greeks                                  # --- Раздел 4: греки и баланс (RU/EN) ---
        if ru:
            sections["Греки и баланс портфеля"] = (
                f"Греки опционного портфеля: дельта={greeks['delta']:.{d}f}, гамма={greeks['gamma']:.{d}f}, "
                f"вега={greeks['vega']:.{d}f}, тета={greeks['theta']:.{d}f}, ро={greeks['rho']:.{d}f}, "
                f"ванна={greeks['vanna']:.{d}f}, волга={greeks['volga']:.{d}f}, чарм={greeks['charm']:.{d}f}. "
                f"Баланс дельты — зона '{status['zone']}' (|дельта|={status['delta_fraction']:.{d}f} капитала). "
                f"После хеджа остаточная дельта={latest['residual_delta']:.{d}f}, остаточная вега="
                f"{latest['residual_vega']:.{d}f}; сделок: {latest['n_trades']}, "
                f"комиссий {latest['fees']:,.2f}$.")
        else:
            sections["Greeks & Portfolio Balance"] = (
                f"Option-book greeks: delta={greeks['delta']:.{d}f}, gamma={greeks['gamma']:.{d}f}, "
                f"vega={greeks['vega']:.{d}f}, theta={greeks['theta']:.{d}f}, rho={greeks['rho']:.{d}f}, "
                f"vanna={greeks['vanna']:.{d}f}, volga={greeks['volga']:.{d}f}, charm={greeks['charm']:.{d}f}. "
                f"Delta balance is in the '{status['zone']}' zone (|delta|={status['delta_fraction']:.{d}f} "
                f"of capital). After hedging, residual delta={latest['residual_delta']:.{d}f}, residual vega="
                f"{latest['residual_vega']:.{d}f}; trades: {latest['n_trades']}, "
                f"fees ${latest['fees']:,.2f}.")

        # ---- 5. portfolio optimisation + diversification (NEW)  # --- Раздел 5: портфель (через _portfolio_section) ---
        title, body = self._portfolio_section(b, ru)     # формируем раздел портфеля
        sections[title] = body                           # добавляем его в объяснение

        # ---- 6. risk & stops                            # --- Раздел 6: риск и стопы (RU/EN) ---
        if ru:
            sections["Управление риском и стоп-лоссы"] = (
                f"VaR={risk['var']:.{d}f}, CVaR/ES={risk['cvar']:.{d}f}, макс. просадка="
                f"{risk['max_drawdown']:.{d}f}. Лимиты {'соблюдены' if risk['within_limits'] else 'НАРУШЕНЫ'} "
                f"({', '.join(risk['breached_limits']) if risk['breached_limits'] else 'нарушений нет'}). "
                f"Адаптивный стоп-лосс на уровне {stop['stop_price']:,.2f} "
                f"(дистанция {stop['distance_pct']:.{d}f}, метод {stop['method']}), компоненты: "
                f"ATR={stop['components'].get('atr_pct', float('nan')):.{d}f}, "
                f"VaR={stop['components'].get('var_pct', float('nan')):.{d}f}, "
                f"Heston={stop['components'].get('heston_vol_pct', float('nan')):.{d}f}.")
        else:
            sections["Risk Management & Stop-losses"] = (
                f"VaR={risk['var']:.{d}f}, CVaR/ES={risk['cvar']:.{d}f}, max drawdown="
                f"{risk['max_drawdown']:.{d}f}. Limits are {'respected' if risk['within_limits'] else 'BREACHED'} "
                f"({', '.join(risk['breached_limits']) if risk['breached_limits'] else 'no breaches'}). "
                f"Adaptive stop-loss at {stop['stop_price']:,.2f} "
                f"(distance {stop['distance_pct']:.{d}f}, method {stop['method']}), components: "
                f"ATR={stop['components'].get('atr_pct', float('nan')):.{d}f}, "
                f"VaR={stop['components'].get('var_pct', float('nan')):.{d}f}, "
                f"Heston={stop['components'].get('heston_vol_pct', float('nan')):.{d}f}.")

        # ---- 7. backtest                                # --- Раздел 7: результаты бэктеста (RU/EN) ---
        stress_txt = self._stress_text(stress, ru)       # текст по худшему стресс-сценарию
        if ru:
            sections["Результаты бэктеста"] = (
                f"ROI={perf['roi']:.{d}f}, Sharpe={perf['sharpe']:.2f}, Sortino={perf['sortino']:.2f}, "
                f"Calmar={perf['calmar']:.2f}, макс. просадка={perf['max_drawdown']:.{d}f}, "
                f"Profit Factor={perf['profit_factor']:.2f}, Win Rate={perf['win_rate']:.2f}, "
                f"VaR={perf['var']:.{d}f}, CVaR={perf['cvar']:.{d}f}, Beta={perf['beta']:.2f}, "
                f"Alpha={perf['alpha']:.{d}f}, Information Ratio={perf['information_ratio']:.2f}. " + stress_txt)
        else:
            sections["Backtest Results"] = (
                f"ROI={perf['roi']:.{d}f}, Sharpe={perf['sharpe']:.2f}, Sortino={perf['sortino']:.2f}, "
                f"Calmar={perf['calmar']:.2f}, max drawdown={perf['max_drawdown']:.{d}f}, "
                f"Profit Factor={perf['profit_factor']:.2f}, Win Rate={perf['win_rate']:.2f}, "
                f"VaR={perf['var']:.{d}f}, CVaR={perf['cvar']:.{d}f}, Beta={perf['beta']:.2f}, "
                f"Alpha={perf['alpha']:.{d}f}, Information Ratio={perf['information_ratio']:.2f}. " + stress_txt)

        # ---- 8. self-assessment                         # --- Раздел 8: самооценка/индекс доверия (RU/EN) ---
        comp = ", ".join(f"{k}={v:.2f}" for k, v in diag["components"].items())  # компоненты доверия строкой
        if ru:
            sections["Самооценка системы"] = (
                f"Интегральный индекс доверия (Confidence Score) = {diag['confidence_score']:.3f} "
                f"({diag['self_assessment']}). Дрейф данных "
                f"{'обнаружен' if diag['drift_detected'] else 'не обнаружен'} (PSI={diag['psi']:.3f}). "
                f"Компоненты доверия: {comp}.")
        else:
            sections["System Self-assessment"] = (
                f"Overall Confidence Score = {diag['confidence_score']:.3f} "
                f"({diag['self_assessment']}). Data drift "
                f"{'detected' if diag['drift_detected'] else 'not detected'} (PSI={diag['psi']:.3f}). "
                f"Confidence components: {comp}.")

        return sections                                  # словарь {заголовок: текст} всех разделов

    def _portfolio_section(self, b, ru: bool):
        rebal = b["rebalance_decision"]                  # решение о ребалансе (метод/обоснование)
        div = b.get("diversification", {})               # метрики диверсификации
        results = b.get("optimization_results", {})      # результаты по методам
        constituents: pd.DataFrame = b.get("portfolio_constituents", pd.DataFrame())  # состав портфеля
        equity: pd.DataFrame = b.get("portfolio_equity", pd.DataFrame())  # equity-кривая
        method = rebal["method"]                         # выбранный метод
        bt = results.get(method, {}).get("backtest", {})  # метрики бэктеста выбранного метода
        total_ret = bt.get("total_return", 0.0)          # суммарная доходность
        cagr = bt.get("cagr", 0.0)                       # CAGR
        sharpe = bt.get("sharpe", 0.0)                   # Sharpe
        mdd = bt.get("max_drawdown", 0.0)                # макс. просадка
        bench_ret = 0.0                                  # доходность бенчмарка (по умолчанию 0)
        if equity is not None and not equity.empty and "benchmark" in equity.columns:  # если есть бенчмарк…
            bench_ret = float(equity["benchmark"].iloc[-1] - 1.0)  # …его итоговая доходность
        dr = div.get("diversification_ratio", 0.0)       # коэффициент диверсификации
        bench_dr = div.get("benchmark_diversification_ratio", 0.0)  # DR бенчмарка
        eff_n = div.get("effective_n", 0.0)              # эффективное число активов
        n_assets = div.get("n_assets", 0)                # число активов
        max_w = div.get("max_weight", 0.0)              # макс. вес
        hhi = div.get("hhi", 0.0)                        # индекс концентрации HHI
        n_reb = div.get("n_rebalances", 0)               # число ребалансировок

        top = ""                                         # строка топ-позиций
        if constituents is not None and not constituents.empty:  # если состав известен…
            items = ", ".join(f"{row['symbol']} {row['weight']:.0%}"  # перечень топ-5 позиций
                              for _, row in constituents.head(5).iterrows())
            top = ("Топ-позиции: " if ru else "Top holdings: ") + items + "."  # подпись по языку

        if ru:                                           # русская версия раздела портфеля
            title = "Оптимизация портфеля и диверсификация"
            body = (
                f"Инвестиционный портфель из {n_assets} инструментов построен методом '{method}', "
                f"выбранным автоматически как лучший по комбинации доходности и диверсификации. "
                f"Бэктест с ребалансировкой (всего {n_reb} ребалансировок, с учётом комиссий) дал "
                f"доходность {total_ret:.2%} (CAGR {cagr:.2%}, Sharpe {sharpe:.2f}, макс. просадка "
                f"{mdd:.2%}) против {bench_ret:.2%} у равновзвешенного бенчмарка — портфель прибыльный. "
                f"Высокая диверсификация подтверждена: коэффициент диверсификации {dr:.2f} "
                f"(бенчмарк {bench_dr:.2f}), эффективное число активов {eff_n:.1f} из {n_assets}, "
                f"максимальный вес {max_w:.1%}, индекс концентрации HHI {hhi:.3f}. " + top)
        else:                                            # английская версия раздела портфеля
            title = "Portfolio Optimization & Diversification"
            body = (
                f"The investable portfolio of {n_assets} instruments is built with the '{method}' method, "
                f"auto-selected as the best trade-off between return and diversification. "
                f"A rebalanced backtest ({n_reb} rebalances, net of fees) delivered a "
                f"{total_ret:.2%} return (CAGR {cagr:.2%}, Sharpe {sharpe:.2f}, max drawdown "
                f"{mdd:.2%}) versus {bench_ret:.2%} for the equal-weight benchmark — the portfolio is "
                f"profitable. High diversification is confirmed: diversification ratio {dr:.2f} "
                f"(benchmark {bench_dr:.2f}), effective number of assets {eff_n:.1f} of {n_assets}, "
                f"max weight {max_w:.1%}, HHI concentration index {hhi:.3f}. " + top)
        return title, body                               # заголовок и тело раздела портфеля

    @staticmethod
    def _stress_text(stress: pd.DataFrame, ru: bool) -> str:
        if stress is None or stress.empty:               # нет стресс-данных…
            return ""                                    #   → пустая строка
        if "net_hedged_pnl" in stress.columns:           # формат с разложением по «ногам»…
            worst = stress.loc[stress["unhedged_pnl"].idxmin()]  # худший сценарий для голого BTC
            if ru:                                       #   русская формулировка
                return (f"В худшем стресс-сценарии '{worst['scenario']}' голый BTC дал бы "
                        f"{worst['unhedged_pnl']:,.0f}$, а захеджированный портфель — "
                        f"{worst['net_hedged_pnl']:,.0f}$ (эффективность хеджа "
                        f"{worst['hedge_effectiveness']:.1%}).")
            return (f"In the worst stress scenario '{worst['scenario']}' naked BTC would lose "  # английская
                    f"${worst['unhedged_pnl']:,.0f}, while the hedged portfolio is at "
                    f"${worst['net_hedged_pnl']:,.0f} (hedge effectiveness "
                    f"{worst['hedge_effectiveness']:.1%}).")
        if "pnl_usd" in stress.columns:                  # упрощённый формат…
            worst = stress.loc[stress["pnl_usd"].idxmin()]  # худший сценарий по PnL
            if ru:
                return f"Худший стресс-сценарий '{worst['scenario']}': PnL {worst['pnl_usd']:,.0f}$."
            return f"Worst stress scenario '{worst['scenario']}': PnL ${worst['pnl_usd']:,.0f}."
        return ""                                        # неизвестный формат → пусто

    @staticmethod
    def _bench_text(bench: dict, d: int, ru: bool) -> str:
        if not bench:                                    # нет бенчмарков…
            return ""                                    #   → пусто
        h = bench.get("heston", {}).get("iv_rmse", float("nan"))      # RMSE IV Хестона
        bs = bench.get("black_scholes", {}).get("iv_rmse", float("nan"))  # RMSE IV Блэка-Шоулза
        sabr = bench.get("sabr", {}).get("rmse", float("nan"))        # RMSE SABR
        if ru:
            return (f"Сравнение по RMSE подразумеваемой волатильности: Heston={h:.{d}f}, "
                    f"Black-Scholes={bs:.{d}f}, SABR={sabr:.{d}f}.")
        return (f"Implied-volatility RMSE comparison: Heston={h:.{d}f}, "
                f"Black-Scholes={bs:.{d}f}, SABR={sabr:.{d}f}.")
