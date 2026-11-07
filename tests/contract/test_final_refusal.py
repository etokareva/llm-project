"""Итоговый отказ имеет приоритет над предшествующими рассуждениями."""

import pytest

from assistant.agent.agent import run_agent
from assistant.graphrag.search import is_refusal
from assistant.infra.testing import FakeLLM


@pytest.mark.parametrize(
    "ending",
    [
        "Ответ: В базе знаний не найдено.",
        "**Ответ:** В базе знаний не найдено.",
        "**Ответ: В базе знаний не найдено.**",
        "Итог: Нет данных для ответа.",
        "Вывод: Данные не найдены. Уточните модель.",
        "В базе знаний не найдено.",
    ],
)
def test_final_refusal_overrides_intermediate_reasoning(ending):
    text = "BZ-110 и BZ-120 имеют по две связи FAILED.\nОднозначный выбор не подтверждён.\n" + ending
    assert is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Цена P-1042B — 2630 руб. Нет данных об остатке.",
        "Ответ: Цена P-1042B — 2630 руб. Нет данных об остатке.",
        "В базе знаний не найдено.\nОтвет: Пусковое реле.",
        "Ответ: В базе знаний не найдено.\nОтвет: Циркуляционный насос.",
    ],
)
def test_partial_or_revised_successful_conclusion_is_not_a_refusal(text):
    assert not is_refusal(text)


def test_agent_returns_refusal_for_final_answer_after_reasoning():
    llm = FakeLLM(["BZ-110 и BZ-120 имеют по две связи FAILED.\nОтвет: В базе знаний не найдено."])
    result = run_agent("У какой модели больше всего отказов: BZ-110 или BZ-120?", llm, {})
    assert result.refused
    assert len(llm.requests) == 1
