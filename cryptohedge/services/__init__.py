"""Computational use-cases (services) used by the agents.

Services are stateless, dependency-light and individually unit-testable. Agents
compose them; this separation keeps the agent layer thin and the numerics reusable.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (вычислительный слой) — маркер-пакет.
НАЗНАЧЕНИЕ:  пакет «сервисов» — чистых вычислительных use-case'ов (волатильность,
             корреляции, калибровка, греки, хедж, оптимизация, метрики, стопы,
             дрейф, walk-forward, i18n, провайдеры данных). Сервисы без состояния,
             легко тестируются по отдельности; агенты лишь компонуют их.
ЭКСПОРТ:     файл намеренно пустой (маркер пакета); сервисы импортируются по
             именам подмодулей, например `from cryptohedge.services import metrics`.
=============================================================================
"""
