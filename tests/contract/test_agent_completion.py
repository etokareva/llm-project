"""Проверки завершения агента, частичных ответов и недопустимых чисел."""

import pytest

from assistant.agent.agent import run_agent
from assistant.agent.validation import extract_answer_numbers, find_answer_errors
from assistant.graphrag.search import is_refusal
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.testing import FakeLLM


def test_erp_failure_remains_visible_at_step_limit():
    attempts = []

    def unavailable(**kwargs):
        attempts.append(kwargs)
        raise ErpUnavailable("offline")

    llm = FakeLLM([LLMResponse("", [ToolCall("c1", "get_price", {"part_id": "P-1042B"})])])
    result = run_agent("Сколько стоит P-1042B?", llm, {"get_price": unavailable}, max_steps=1)
    assert len(attempts) == 3
    assert len(llm.requests) == 1
    assert "лимит" in result.answer
    assert "Данные ERP получить не удалось" in result.answer
    assert result.steps[0].error


def test_created_order_survives_two_invalid_model_answers():
    arguments = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}
    calls = []

    def create_order(**kwargs):
        calls.append(kwargs)
        return {"order_id": "SO-000001", **kwargs}

    llm = FakeLLM(
        [
            LLMResponse("", [ToolCall("c1", "create_service_order", arguments)]),
            "Цена 9999 руб.",
            "Цена 9999 руб.",
        ]
    )
    result = run_agent("Закажи 2 штуки P-1042B для AR-210.", llm, {"create_service_order": create_order}, confirm=True)
    assert calls == [arguments]
    assert "SO-000001" in result.answer
    assert "9999" not in result.answer
    assert "В базе знаний не найдено" not in result.answer
    assert not result.refused


def test_erp_failure_remains_visible_while_asking_for_confirmation():
    def unavailable(**kwargs):
        raise ErpUnavailable("offline")

    def forbidden(**kwargs):
        pytest.fail("Заказ без подтверждения не выполняется")

    arguments = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}
    llm = FakeLLM(
        [
            LLMResponse("", [ToolCall("c1", "get_price", {"part_id": "P-1042B"})]),
            LLMResponse("", [ToolCall("c2", "create_service_order", arguments)]),
        ]
    )
    result = run_agent(
        "Закажи 2 штуки P-1042B для AR-210.",
        llm,
        {"get_price": unavailable, "create_service_order": forbidden},
    )
    assert result.needs_confirmation
    assert result.pending_action["args"] == arguments
    assert "Данные ERP получить не удалось" in result.answer


@pytest.mark.parametrize(
    "text",
    [
        "Цена P-1042B: -9999 руб.",
        "Цена P-1042B: −9999 руб.",
        "Цена - 9999.",
        "Остаток P-1042B: -99 шт.",
        "Остаток: −99.",
        "В наличии -3 шт.",
    ],
)
def test_negative_prices_and_stock_are_rejected(text):
    assert find_answer_errors(text, "Проверь P-1042B.", [])


def test_negative_number_in_question_does_not_authorize_negative_price():
    assert find_answer_errors("Цена: -10 руб.", "Цена P-1042B действительно -10 руб.?", [])


def test_signed_numbers_preserve_codes_and_decimal_values():
    assert extract_answer_numbers("AR-210, P-1042B. Цена: +2 630,50 руб. Остаток: 0.") == ({"2630.5"}, {"0"})


@pytest.mark.parametrize(
    "text",
    [
        "Цена P-1042B — 2630 руб. Нет данных об остатке.",
        "Нет данных об остатке. Цена P-1042B — 2630 руб.",
        "Цена P-1042B — 2630 руб., но нет данных об остатке.",
        "Цена P-1042B — 2630 руб., нет данных об остатке.",
    ],
)
def test_partial_answer_is_not_a_complete_refusal(text):
    assert not is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "В базе знаний не найдено.",
        "Ошибка E-99 не найдена. Пожалуйста, уточните код ошибки.",
        "Деталь P-9999 не найдена. Проверьте правильность кода.",
        "Нет информации. Уточните модель оборудования.",
    ],
)
def test_refusal_with_clarification_remains_a_refusal(text):
    assert is_refusal(text)


def test_agent_preserves_successful_price_in_partial_answer():
    llm = FakeLLM(
        [
            LLMResponse("", [ToolCall("c1", "get_price", {"part_id": "P-1042B"})]),
            "Цена P-1042B — 2630 руб. Нет данных об остатке.",
        ]
    )
    result = run_agent(
        "Сколько стоит P-1042B и есть ли она на складе?",
        llm,
        {"get_price": lambda **kwargs: {"part_id": "P-1042B", "price_rub": 2630}},
    )
    assert not result.refused
    assert "2630" in result.answer


def test_successful_order_and_erp_failure_both_survive_invalid_answer():
    def unavailable(**kwargs):
        raise ErpUnavailable("offline")

    arguments = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}
    llm = FakeLLM(
        [
            LLMResponse("", [ToolCall("c1", "create_service_order", arguments)]),
            LLMResponse("", [ToolCall("c2", "get_price", {"part_id": "P-1042B"})]),
            "Цена 9999 руб.",
            "Цена 9999 руб.",
        ]
    )
    result = run_agent(
        "Закажи 2 штуки P-1042B для AR-210 и узнай цену.",
        llm,
        {
            "create_service_order": lambda **kwargs: {"order_id": "SO-000001", **kwargs},
            "get_price": unavailable,
        },
        confirm=True,
    )
    assert "SO-000001" in result.answer
    assert "Данные ERP получить не удалось" in result.answer
    assert "9999" not in result.answer
    assert not result.refused


def test_no_data_at_step_limit_remains_an_explicit_refusal():
    llm = FakeLLM([LLMResponse("", [ToolCall("c1", "get_price", {"part_id": "P-9999"})])])
    result = run_agent(
        "Сколько стоит P-9999?",
        llm,
        {"get_price": lambda **kwargs: {"error": "Деталь не найдена", "error_type": "not_found"}},
        max_steps=1,
    )
    assert result.refused
    assert "В базе знаний не найдено" in result.answer
    assert "лимит" in result.answer
