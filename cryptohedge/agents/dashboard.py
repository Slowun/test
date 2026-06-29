"""Dashboard Agent.

Role: assemble the monitoring dashboard and persist a fully Russian and a fully
English self-contained HTML file. Panels: synchronized spot + PnL + theta/vega +
hedge; greeks imbalance indicators; the *portfolio* (constituents, value dynamics
with rebalancing, weight evolution, diversification diagnostics); a greeks
heatmap; the stress-test comparison; the key-metrics block; hedge costs and
rebalancing frequency; and the instrument ranking. Section titles are rendered as
HTML headings (not Plotly titles) so labels never overlap the charts.

================================ КАРТА МОДУЛЯ ================================
АГЕНТ:       11 / 11 — DashboardAgent (последний этап пайплайна).
НАЗНАЧЕНИЕ:  собирает интерактивный HTML-дашборд (Plotly) и сохраняет ПОЛНОСТЬЮ
             русскую и полностью английскую версии. Панели: спот/PnL/греки/хедж,
             индикаторы дисбаланса, портфель (состав/equity/веса/диверсификация),
             тепловая карта греков, стресс-тесты, метрики, издержки, рейтинг.
ВХОД (consumes):  EXPLANATION_READY (от агента 10).
ВЫХОД (produces): DASHBOARD_READY → оркестратору (конец пайплайна).
КЛАДЁТ НА ДОСКУ:  dashboard_path, dashboard_paths.
ЧИТАЕТ С ДОСКИ:   hedge_history, chain_greeks, stress_table, backtest_metrics,
                  rankings_df, trailing_stops, hedge_status, portfolio_*, …
ИМПОРТИРУЕТ:
  - plotly.graph_objects/subplots : построение интерактивных фигур.
  - services.i18n.{t,metric_label,relationship_label} : двуязычные подписи.
КОНСТАНТЫ:
  - _POS/_NEG  : цвета прибыли/убытка; _PALETTE : палитра для стэка весов.
КОНФИГ:  config.dashboard (языки/тема/размеры), config.hedging (зоны дельты).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from datetime import datetime, timezone                  # временна́я метка генерации
from typing import Dict, List, Optional                  # аннотации

import numpy as np                                       # числовые операции
import pandas as pd                                       # таблицы артефактов
import plotly.graph_objects as go                        # объекты графиков Plotly
from plotly.subplots import make_subplots                # сетки подграфиков

from cryptohedge.core.agent import BaseAgent             # контракт агента
from cryptohedge.core.context import AgentContext        # контекст
from cryptohedge.core.message import Message, MessageType # сообщения
from cryptohedge.services.i18n import metric_label, relationship_label, t  # двуязычные подписи

_POS = "#66bb6a"                                          # цвет «прибыль» (зелёный)
_NEG = "#ef5350"                                          # цвет «убыток» (красный)
_PALETTE = ["#4fc3f7", "#66bb6a", "#ffa726", "#ab47bc", "#26c6da", "#ec407a",  # палитра для стэка весов
            "#9ccc65", "#ff7043", "#5c6bc0", "#26a69a", "#d4e157", "#8d6e63",
            "#42a5f5", "#ffca28", "#7e57c2"]


class DashboardAgent(BaseAgent):
    name = "dashboard"                                   # имя агента / id этапа
    consumes = [MessageType.EXPLANATION_READY]           # принимает EXPLANATION_READY
    produces = MessageType.DASHBOARD_READY               # выпускает DASHBOARD_READY
    checkpoint_keys = ["dashboard_path", "dashboard_paths"]  # ключи чекпойнта

    def execute(self, context: AgentContext, message: Message) -> Message:
        log = context.logger(self.name)                  # логгер агента
        cfg = context.config.dashboard                   # секция конфига дашборда
        languages = list(cfg.languages) or ["ru"]        # список языков (минимум русский)

        paths: Dict[str, str] = {}                       # пути к файлам по языкам
        rendered: Dict[str, str] = {}                    # отрендеренный HTML по языкам
        for lang in languages:                           # по каждому языку…
            html = self._render(context, lang, languages)  # рендерим HTML дашборда
            rendered[lang] = html                        # запоминаем HTML
            out = context.root / cfg.output_dir / f"dashboard_{lang}.html"  # путь файла языка
            out.parent.mkdir(parents=True, exist_ok=True)  # создаём каталог
            out.write_text(html, encoding="utf-8")       # пишем HTML-файл
            paths[lang] = str(out)                        # сохраняем путь

        # keep a stable default path (RU first, else first language) for back-compat  # дефолтный путь
        default_lang = "ru" if "ru" in rendered else languages[0]  # язык по умолчанию (RU при наличии)
        default_path = context.root / cfg.output_html    # путь основного дашборда
        default_path.parent.mkdir(parents=True, exist_ok=True)  # создаём каталог
        default_path.write_text(rendered[default_lang], encoding="utf-8")  # пишем дефолтный файл

        context.put("dashboard_paths", paths)            # пути по языкам → на доску
        context.put("dashboard_path", str(default_path))  # дефолтный путь → на доску
        log.decision("dashboard generated", languages=languages, **{f"path_{k}": v for k, v in paths.items()})  # лог

        return Message(self.produces, self.name, "orchestrator",  # DASHBOARD_READY оркестратору (конец)
                       payload={"paths": paths}, correlation_id=message.correlation_id)

    # ------------------------------------------------------------------ render
    def _render(self, context: AgentContext, lang: str, languages: List[str]) -> str:
        b = context.blackboard                           # короткая ссылка на доску
        cfg = context.config.dashboard                   # секция конфига дашборда

        history: pd.DataFrame = b["hedge_history"]       # история хеджа (обязательна)
        chain: pd.DataFrame = b.get("chain_greeks", pd.DataFrame())  # греки по страйкам
        stress: pd.DataFrame = b.get("stress_table", pd.DataFrame())  # стресс-тесты
        perf: dict = b.get("backtest_metrics", {})       # метрики бэктеста
        rankings_df: pd.DataFrame = b.get("rankings_df", pd.DataFrame())  # рейтинг инструментов
        trailing: pd.DataFrame = b.get("trailing_stops", pd.DataFrame())  # траектория трейлинга
        status = b.get("hedge_status", {})               # статус баланса дельты
        constituents: pd.DataFrame = b.get("portfolio_constituents", pd.DataFrame())  # состав портфеля
        equity: pd.DataFrame = b.get("portfolio_equity", pd.DataFrame())  # equity-кривая
        weights_path: pd.DataFrame = b.get("portfolio_weights_path", pd.DataFrame())  # путь весов
        rebalances = b.get("portfolio_rebalances", [])   # даты ребалансировок
        diversification = b.get("diversification", {})   # метрики диверсификации
        method_comparison: pd.DataFrame = b.get("method_comparison", pd.DataFrame())  # сравнение методов

        # ordered (section-key, figure-or-figures)        # упорядоченный список панелей (ключ раздела + фигуры)
        panels = [
            ("sec_timeseries", self._timeseries_fig(history, trailing, cfg, lang)),  # спот/PnL/греки/хедж
            ("sec_greeks", self._greeks_panel(history, status, context, lang)),      # индикаторы греков
            ("sec_portfolio_constituents", self._constituents_table(constituents, lang)),  # состав портфеля
            ("sec_portfolio_equity", self._equity_fig(equity, rebalances, cfg, lang)),     # equity портфеля
            ("sec_portfolio_weights", self._weights_fig(weights_path, cfg, lang)),         # эволюция весов
            ("sec_diversification", self._diversification_figs(diversification, method_comparison, cfg, lang)),  # диверсификация
            ("sec_heatmap", self._heatmap(chain, cfg, lang)),     # тепловая карта греков
            ("sec_stress", self._stress_fig(stress, cfg, lang)),  # стресс-тесты
            ("sec_metrics", self._metrics_table(perf, lang)),     # таблица метрик
            ("sec_costs", self._costs_fig(history, cfg, lang)),   # издержки/частота ребаланса
            ("sec_rankings", self._rankings_table(rankings_df, lang)),  # рейтинг инструментов
        ]

        body: List[str] = []                             # HTML-фрагменты тела страницы
        first = True                                     # флаг: первая фигура подключает plotly.js (CDN)
        for sec_key, figs in panels:                     # по каждой панели…
            if figs is None:                             #   нет данных — пропускаем
                continue
            fig_list = figs if isinstance(figs, list) else [figs]  # нормализуем к списку фигур
            fig_list = [f for f in fig_list if f is not None]  # отбрасываем пустые
            if not fig_list:                             #   ничего не осталось — пропускаем
                continue
            body.append(f"<h2>{t(lang, sec_key)}</h2>")  #   заголовок раздела (HTML, не Plotly)
            for fig in fig_list:                         #   по каждой фигуре раздела…
                body.append(fig.to_html(full_html=False,  #     встраиваем фигуру в HTML
                                        include_plotlyjs=("cdn" if first else False)))  # plotly.js один раз
                first = False                            #     дальше js уже подключён

        sections = (b.get(f"explanation_sections_{lang}")  # текстовые разделы объяснения на нужном языке
                    or b.get("explanation_sections", {}))  # либо дефолтные (RU)
        explanation_html = self._explanation_html(sections)  # объяснение → HTML
        other = [l for l in languages if l != lang]      # другие языки (для переключателя)
        switch = ""                                      # ссылка-переключатель языка
        if other:                                        # если есть другой язык…
            ol = other[0]                                #   первый из других
            switch = f"<a class='lang' href='dashboard_{ol}.html'>{t(lang, 'lang_switch')}</a>"  # ссылка
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")  # метка времени генерации

        return (                                         # собираем итоговый HTML-документ
            "<!DOCTYPE html><html lang='" + lang + "'><head><meta charset='utf-8'>"
            f"<title>{t(lang, 'doc_title')}</title>"
            "<style>"                                    # встроенные стили (тёмная тема)
            "body{background:#0f1115;color:#e6edf3;font-family:Segoe UI,Arial,sans-serif;margin:24px;max-width:1320px;}"
            "h1{color:#4fc3f7;margin-bottom:2px;} "
            "h2{color:#80cbc4;border-bottom:1px solid #263238;padding-bottom:6px;margin-top:34px;} "
            "h3{color:#9ccc65;margin-bottom:4px;} p{line-height:1.55;color:#cfd8dc;} "
            ".lang{float:right;color:#0f1115;background:#4fc3f7;padding:6px 12px;border-radius:6px;"
            "text-decoration:none;font-weight:600;} .ts{color:#78909c;font-size:13px;}"
            "</style></head><body>"
            f"{switch}"                                  # переключатель языка
            f"<h1>{t(lang, 'main_header')}</h1>"         # главный заголовок
            f"<div class='ts'>{t(lang, 'generated')}: {stamp}</div>"  # время генерации
            + "".join(body)                              # все панели
            + f"<h2>{t(lang, 'explanation_header')}</h2>"  # заголовок раздела объяснений
            + explanation_html                           # текст объяснений
            + "</body></html>"
        )

    # ------------------------------------------------------------------ panels
    def _timeseries_fig(self, h: pd.DataFrame, trailing: pd.DataFrame, cfg, lang: str) -> go.Figure:
        # Панель из 4 синхронизированных подграфиков: спот, PnL, theta/vega, объёмы хеджа.
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.075,  # 4 строки, общая ось X
            subplot_titles=(t(lang, "ts_sub_spot"), t(lang, "ts_sub_pnl"),  # подписи подграфиков
                            t(lang, "ts_sub_thetavega"), t(lang, "ts_sub_hedges")),
        )
        fig.add_trace(go.Scatter(x=h["ts"], y=h["spot"], name=t(lang, "ser_spot"),  # спот
                                 line=dict(color="#4fc3f7")), 1, 1)
        if trailing is not None and not trailing.empty:  # если есть траектория трейлинг-стопа…
            fig.add_trace(go.Scatter(x=trailing["ts"], y=trailing["stop_price"],  # линия стопа
                                     name=t(lang, "ser_trailing_stop"),
                                     line=dict(color="#ef5350", dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["pnl"], name=t(lang, "ser_pnl"),  # PnL
                                 line=dict(color="#66bb6a")), 2, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["fee"], name=t(lang, "ser_fees_cum"),  # накопленные комиссии
                                 line=dict(color="#ffa726")), 2, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["theta"], name=t(lang, "ser_theta"),  # theta
                                 line=dict(color="#ab47bc")), 3, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["vega"], name=t(lang, "ser_vega"),  # vega
                                 line=dict(color="#26c6da")), 3, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["delta_hedge"], name=t(lang, "ser_delta_hedge"),  # объём Δ-хеджа
                                 line=dict(color="#42a5f5")), 4, 1)
        fig.add_trace(go.Scatter(x=h["ts"], y=h["vega_hedge"], name=t(lang, "ser_vega_hedge"),  # объём ν-хеджа
                                 line=dict(color="#ec407a")), 4, 1)
        fig.update_layout(template=cfg.theme, height=1050, margin=dict(t=50, b=30, l=60, r=140),  # оформление
                          legend=dict(orientation="v", yanchor="top", y=1.0, x=1.02))
        return fig                                       # готовая фигура временны́х рядов

    def _greeks_panel(self, h: pd.DataFrame, status: dict, context, lang: str) -> go.Figure:
        # Панель из 4 индикаторов: gauge баланса дельты + числовые gamma/vega/theta.
        last = h.iloc[-1]                                # последнее состояние книги
        green = context.config.hedging.delta_green_zone  # порог зелёной зоны
        red = context.config.hedging.delta_red_zone      # порог красной зоны
        frac = float(status.get("delta_fraction", 0.0))  # текущая доля дельты от капитала
        top = max(red * 2, 0.3)                           # верх шкалы gauge
        fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.09,  # 4 индикатора в ряд
                            specs=[[{"type": "indicator"}] * 4])
        fig.add_trace(go.Indicator(                      # gauge баланса дельты с зонами
            mode="gauge+number", value=frac, title={"text": t(lang, "ind_delta_ratio"), "font": {"size": 14}},
            number={"font": {"size": 26}},
            gauge={"axis": {"range": [0, top]}, "bar": {"color": "#263238"},
                   "steps": [{"range": [0, green], "color": "#2e7d32"},  # зелёная зона
                             {"range": [green, red], "color": "#f9a825"},  # жёлтая зона
                             {"range": [red, top], "color": "#c62828"}]}), 1, 1)  # красная зона
        for i, (key, lbl) in enumerate(                  # числовые индикаторы gamma/vega/theta
                [("gamma", "ind_gamma"), ("vega", "ind_vega"), ("theta", "ind_theta")], start=2):
            fig.add_trace(go.Indicator(mode="number", value=float(last[key]),  # значение грека
                                       number={"font": {"size": 30}},
                                       title={"text": t(lang, lbl), "font": {"size": 14}}), 1, i)
        fig.update_layout(template=context.config.dashboard.theme, height=300,  # оформление
                          margin=dict(t=60, b=30, l=30, r=30))
        return fig                                       # готовая панель индикаторов

    def _constituents_table(self, df: pd.DataFrame, lang: str) -> Optional[go.Figure]:
        # Таблица состава портфеля (символ, вес, доходность, волатильность, связь с BTC).
        if df is None or df.empty:                       # нет данных — нет таблицы
            return None
        d = df.copy()                                    # копия для форматирования
        rel = [relationship_label(lang, str(r)) for r in d.get("relationship", ["" for _ in range(len(d))])]  # связь→подпись
        header = [t(lang, "col_symbol"), t(lang, "col_weight"), t(lang, "col_exp_return"),  # заголовки колонок
                  t(lang, "col_vol"), t(lang, "col_relationship")]
        cells = [                                        # значения ячеек по колонкам
            d["symbol"].tolist(),
            [f"{w:.1%}" for w in d["weight"]],           # вес в процентах
            [f"{r:.1%}" for r in d["exp_return_annual"]],  # доходность в процентах
            [f"{v:.1%}" for v in d["vol_annual"]],        # волатильность в процентах
            rel,                                         # связь с BTC (локализованная)
        ]
        fig = go.Figure(go.Table(                        # строим таблицу Plotly
            columnwidth=[110, 70, 130, 120, 110],
            header=dict(values=header, fill_color="#263238", font=dict(color="white", size=13), height=30),
            cells=dict(values=cells, fill_color="#1b242b", font=dict(color="#e6edf3", size=12), height=26)))
        fig.update_layout(height=min(560, 90 + 28 * len(d)), margin=dict(t=10, b=10, l=10, r=10))  # высота по строкам
        return fig                                       # готовая таблица состава

    def _equity_fig(self, equity: pd.DataFrame, rebalances, cfg, lang: str) -> Optional[go.Figure]:
        # equity-кривая портфеля против равновзвешенного бенчмарка + маркеры ребаланса.
        if equity is None or equity.empty:               # нет данных — нет графика
            return None
        ts = pd.to_datetime(equity["ts"])                # ось времени
        fig = go.Figure()                                # пустая фигура
        fig.add_trace(go.Scatter(x=ts, y=equity["equity"], name=t(lang, "ser_portfolio"),  # портфель
                                 line=dict(color="#66bb6a", width=2)))
        if "benchmark" in equity.columns:                # если есть бенчмарк…
            fig.add_trace(go.Scatter(x=ts, y=equity["benchmark"], name=t(lang, "ser_benchmark_eqw"),  # 1/n
                                     line=dict(color="#90a4ae", dash="dash")))
        # rebalance markers on the portfolio curve        # маркеры ребалансировок на кривой
        if rebalances:                                   # если есть даты ребаланса…
            reb = pd.to_datetime([str(r) for r in rebalances])  # парсим даты
            idx = pd.Index(ts)                           # индекс времени
            eq_at = []                                   # значения equity в точках ребаланса
            xs = []                                       # соответствующие даты
            for r in reb:                                # по каждой дате ребаланса…
                pos = idx.get_indexer([r], method="nearest")[0]  # ближайшая точка кривой
                if pos >= 0:                             #   если найдена…
                    xs.append(ts.iloc[pos]); eq_at.append(equity["equity"].iloc[pos])  # копим маркер
            fig.add_trace(go.Scatter(x=xs, y=eq_at, mode="markers", name=t(lang, "ser_rebalance"),  # маркеры
                                     marker=dict(color="#ffca28", size=8, symbol="diamond")))
        fig.update_layout(template=cfg.theme, height=400, margin=dict(t=30, b=40, l=60, r=30),  # оформление
                          yaxis_title=t(lang, "axis_equity"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        return fig                                       # готовая equity-кривая

    def _weights_fig(self, weights_path: pd.DataFrame, cfg, lang: str) -> Optional[go.Figure]:
        # Эволюция весов портфеля во времени (стэк-площади), топ-18 активов.
        if weights_path is None or weights_path.empty:   # нет данных — нет графика
            return None
        ts = pd.to_datetime(weights_path["ts"])          # ось времени
        asset_cols = [c for c in weights_path.columns if c != "ts"]  # колонки активов
        # order by average weight; cap at 18 series to keep the legend readable  # топ по среднему весу
        means = weights_path[asset_cols].mean().sort_values(ascending=False)  # средние веса
        show = list(means.head(18).index)                # показываем максимум 18 активов
        fig = go.Figure()                                # пустая фигура
        for i, col in enumerate(show):                   # по каждому показываемому активу…
            fig.add_trace(go.Scatter(x=ts, y=weights_path[col], name=col, mode="lines",  # стэк-площадь
                                     line=dict(width=0.5, color=_PALETTE[i % len(_PALETTE)]),
                                     stackgroup="one"))
        fig.update_layout(template=cfg.theme, height=430, margin=dict(t=30, b=40, l=60, r=30),  # оформление
                          yaxis_title=t(lang, "axis_weight"), yaxis_range=[0, 1],
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)))
        return fig                                       # готовый график весов

    def _diversification_figs(self, div: dict, methods: pd.DataFrame, cfg, lang: str) -> Optional[List[go.Figure]]:
        # Две фигуры: индикаторы диверсификации + сравнение методов (доходность vs DR).
        if not div and (methods is None or methods.empty):  # нет данных — ничего не строим
            return None
        figs: List[go.Figure] = []                       # список фигур раздела
        if div:                                          # если есть метрики диверсификации…
            ind = make_subplots(rows=1, cols=4, horizontal_spacing=0.09,  # 4 индикатора в ряд
                                specs=[[{"type": "indicator"}] * 4])
            ind.add_trace(go.Indicator(mode="number", value=float(div.get("diversification_ratio", 0.0)),  # DR
                                       number={"valueformat": ".2f", "font": {"size": 30}},
                                       title={"text": t(lang, "ind_div_ratio"), "font": {"size": 13}}), 1, 1)
            ind.add_trace(go.Indicator(mode="number", value=float(div.get("effective_n", 0.0)),  # эфф. число активов
                                       number={"valueformat": ".1f", "font": {"size": 30}},
                                       title={"text": t(lang, "ind_eff_n"), "font": {"size": 13}}), 1, 2)
            ind.add_trace(go.Indicator(mode="number", value=float(div.get("max_weight", 0.0)),  # макс. вес
                                       number={"valueformat": ".1%", "font": {"size": 30}},
                                       title={"text": t(lang, "ind_max_weight"), "font": {"size": 13}}), 1, 3)
            ind.add_trace(go.Indicator(mode="number", value=float(div.get("hhi", 0.0)),  # индекс HHI
                                       number={"valueformat": ".3f", "font": {"size": 30}},
                                       title={"text": t(lang, "ind_hhi"), "font": {"size": 13}}), 1, 4)
            ind.update_layout(template=cfg.theme, height=240, margin=dict(t=50, b=20, l=30, r=30))  # оформление
            figs.append(ind)                             # добавляем фигуру индикаторов

        if methods is not None and not methods.empty:    # если есть сравнение методов…
            m = methods.copy()                           # копия таблицы методов
            bar = make_subplots(specs=[[{"secondary_y": True}]])  # две оси Y (доходность и DR)
            colors = ["#66bb6a" if c else "#455a64" for c in m.get("chosen", [False] * len(m))]  # выделить выбранный
            bar.add_trace(go.Bar(x=m["method"], y=m["total_return"], name=t(lang, "div_bar_return"),  # столбцы доходности
                                 marker_color=colors), secondary_y=False)
            bar.add_trace(go.Scatter(x=m["method"], y=m["diversification_ratio"], mode="markers+lines",  # линия DR
                                     name=t(lang, "div_bar_dr"), marker=dict(color="#ffca28", size=10)),
                          secondary_y=True)
            bar.update_yaxes(title_text=t(lang, "div_ret_axis"), tickformat=".0%", secondary_y=False)  # ось доходности
            bar.update_yaxes(title_text=t(lang, "div_dr_axis"), secondary_y=True)  # ось DR
            bar.update_layout(template=cfg.theme, height=380, margin=dict(t=70, b=40, l=60, r=60),  # оформление
                              title=dict(text=t(lang, "div_methods_title"), font=dict(size=14),
                                         x=0.0, xanchor="left", y=0.99, yanchor="top"),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.32))
            figs.append(bar)                             # добавляем фигуру сравнения методов
        return figs                                      # список фигур раздела диверсификации

    def _heatmap(self, chain: pd.DataFrame, cfg, lang: str) -> Optional[go.Figure]:
        # Тепловая карта греков по сетке страйков (moneyness × грек).
        if chain is None or chain.empty:                 # нет данных — нет карты
            return None
        cols = [c for c in ["delta", "gamma", "vega", "theta", "vanna", "volga", "charm"] if c in chain]  # греки
        z = chain[cols].to_numpy().T                     # матрица значений (греки × страйки)
        fig = go.Figure(go.Heatmap(z=z, x=[f"{m:.2f}" for m in chain["moneyness"]], y=cols,  # тепловая карта
                                   colorscale="RdBu", zmid=0, colorbar=dict(thickness=14)))
        fig.update_layout(template=cfg.theme, height=360, margin=dict(t=20, b=50, l=70, r=30),  # оформление
                          xaxis_title=t(lang, "heatmap_x"))
        return fig                                       # готовая тепловая карта

    def _stress_fig(self, stress: pd.DataFrame, cfg, lang: str) -> Optional[go.Figure]:
        # Сравнение PnL при стрессах: голый BTC vs захеджированный портфель.
        if stress is None or stress.empty:               # нет данных — нет графика
            return None
        if "net_hedged_pnl" in stress.columns:           # формат с разложением…
            fig = go.Figure()                            # пустая фигура
            fig.add_trace(go.Bar(x=stress["scenario"], y=stress["unhedged_pnl"],  # голый BTC
                                 name=t(lang, "ser_unhedged"), marker_color=_NEG))
            fig.add_trace(go.Bar(x=stress["scenario"], y=stress["net_hedged_pnl"],  # хеджированный
                                 name=t(lang, "ser_hedged"), marker_color=_POS))
            fig.update_layout(template=cfg.theme, height=380, barmode="group",  # сгруппированные столбцы
                              margin=dict(t=30, b=40, l=70, r=30), yaxis_title=t(lang, "stress_yaxis"),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            return fig                                    # сгруппированная диаграмма
        col = "pnl_usd" if "pnl_usd" in stress.columns else stress.columns[-1]  # упрощённый формат
        colors = [_POS if v >= 0 else _NEG for v in stress[col]]  # цвет по знаку
        fig = go.Figure(go.Bar(x=stress["scenario"], y=stress[col], marker_color=colors))  # столбцы PnL
        fig.update_layout(template=cfg.theme, height=350, margin=dict(t=30, b=40, l=70, r=30),  # оформление
                          yaxis_title=t(lang, "stress_yaxis"))
        return fig                                       # готовая диаграмма стрессов

    def _metrics_table(self, perf: dict, lang: str) -> Optional[go.Figure]:
        # Таблица ключевых метрик производительности.
        if not perf:                                     # нет метрик — нет таблицы
            return None
        keys = [k for k in ["roi", "cagr", "sharpe", "sortino", "calmar", "max_drawdown",  # порядок метрик
                            "profit_factor", "win_rate", "var", "cvar", "expected_shortfall",
                            "beta", "alpha", "information_ratio", "volatility"] if k in perf]
        fig = go.Figure(go.Table(                        # таблица метрика→значение
            columnwidth=[200, 120],
            header=dict(values=[t(lang, "col_metric"), t(lang, "col_value")],
                        fill_color="#263238", font=dict(color="white", size=13), height=30),
            cells=dict(values=[[metric_label(lang, k) for k in keys],  # локализованные названия метрик
                               [f"{perf[k]:.4f}" for k in keys]],       # значения
                       fill_color="#1b242b", font=dict(color="#e6edf3", size=12), height=26)))
        fig.update_layout(height=90 + 28 * len(keys), margin=dict(t=10, b=10, l=10, r=10))  # высота по строкам
        return fig                                       # готовая таблица метрик

    def _costs_fig(self, h: pd.DataFrame, cfg, lang: str) -> go.Figure:
        # Накопленные комиссии и частота ребалансировок во времени.
        trades = h["pos_spot"].diff().abs().fillna(0) + h["pos_vega_option"].diff().abs().fillna(0)  # изменения позиций
        rebs = (trades > 1e-9).astype(int)               # признак сделки (был ли ребаланс)
        fig = make_subplots(specs=[[{"secondary_y": True}]])  # две оси Y
        fig.add_trace(go.Scatter(x=h["ts"], y=h["fee"], name=t(lang, "ser_fees_cum"),  # накопленные комиссии
                                 line=dict(color="#ffa726")), secondary_y=False)
        fig.add_trace(go.Bar(x=h["ts"], y=rebs, name=t(lang, "ser_rebalances"),  # столбики ребалансов
                             marker_color="#5c6bc0", opacity=0.5), secondary_y=True)
        fig.update_layout(template=cfg.theme, height=340, margin=dict(t=30, b=40, l=60, r=60),  # оформление
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        fig.update_yaxes(title_text=t(lang, "axis_usd"), secondary_y=False)  # ось издержек
        return fig                                       # готовый график издержек

    def _rankings_table(self, rankings_df: pd.DataFrame, lang: str) -> Optional[go.Figure]:
        # Таблица топ-10 инструментов хеджа из рейтинга.
        if rankings_df is None or rankings_df.empty:     # нет рейтинга — нет таблицы
            return None
        df = rankings_df.head(10).copy()                 # топ-10 строк
        cols = [c for c in ["symbol", "score", "pearson", "spearman", "kendall", "dcc_mean",  # колонки рейтинга
                            "cointegrated", "stability", "relationship"] if c in df.columns]
        values = []                                      # значения по колонкам
        for c in cols:                                   # по каждой колонке…
            if c == "relationship":                      #   связь → локализованная подпись
                values.append([relationship_label(lang, str(v)) for v in df[c]])
            elif df[c].dtype.kind in "fc":               #   числа → округляем
                values.append(df[c].round(3).tolist())
            else:                                        #   прочее → как есть
                values.append(df[c].tolist())
        fig = go.Figure(go.Table(                        # таблица рейтинга
            header=dict(values=cols, fill_color="#263238", font=dict(color="white", size=12), height=30),
            cells=dict(values=values, fill_color="#1b242b", font=dict(color="#e6edf3", size=11), height=24)))
        fig.update_layout(height=90 + 26 * len(df), margin=dict(t=10, b=10, l=10, r=10))  # высота по строкам
        return fig                                       # готовая таблица рейтинга

    @staticmethod
    def _explanation_html(sections: dict) -> str:
        # Превращает словарь {заголовок: текст} в HTML (h3 + параграфы, переносы строк → <br>).
        return "".join(f"<h3>{title}</h3><p>{str(body).replace(chr(10), '<br>')}</p>"
                       for title, body in sections.items())
