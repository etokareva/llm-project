"""Граничные сценарии, не покрытые исходными контрактными тестами."""

import json

import networkx as nx
import pytest

from assistant.agent.agent import run_agent
from assistant.agent.validation import (
    extract_answer_numbers,
    find_answer_errors,
    normalize_number,
    quantity_is_explicit,
    validate_tool_call,
)
from assistant.graphrag.communities import detect_communities
from assistant.graphrag.extraction import parse_response
from assistant.graphrag.graph_builder import build_graph
from assistant.graphrag.search import is_refusal
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.schemas import Entity, ExtractionResult, Relation, ToolStep
from assistant.infra.testing import FakeLLM


@pytest.mark.parametrize("value", [[], {}, None, True, 12])
def test_malformed_entity_type_is_a_validation_error(value):
    body = {"entities": [{"type": value, "name": "P-1042"}], "relations": []}
    with pytest.raises(ValueError):
        parse_response(json.dumps(body))


@pytest.mark.parametrize("field", ["subject_type", "object_type"])
def test_malformed_relation_endpoint_is_a_validation_error(field):
    relation = {
        "subject_type": "Part",
        "subject": "P-1042",
        "type": "REPLACED_BY",
        "object_type": "Part",
        "object": "P-1042B",
        field: {},
    }
    with pytest.raises(ValueError):
        parse_response(json.dumps({"entities": [], "relations": [relation]}))


def test_graph_adds_relation_endpoints_with_complete_attributes():
    old, new = Entity("Part", "P-1042"), Entity("Part", "P-1042B")
    graph = build_graph([ExtractionResult("a#0", "a", [], [Relation(old, "REPLACED_BY", new)])])
    assert graph.nodes[old.node_id] == {"type": "Part", "name": "P-1042", "sources": ["a#0"]}
    assert graph.nodes[new.node_id]["sources"] == ["a#0"]


def test_louvain_sums_opposite_weights(monkeypatch):
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=2)
    graph.add_edge("b", "a", weight=3)

    def partition(undirected, **kwargs):
        assert undirected.edges["a", "b"]["weight"] == 5
        return [{"a", "b"}]

    monkeypatch.setattr(nx.community, "louvain_communities", partition)
    assert detect_communities(graph)[0].nodes == ["a", "b"]
    assert graph.edges["a", "b"]["weight"] == 2


@pytest.mark.parametrize(
    "question, qty, expected",
    [
        ("Закажи для ar 210", 210, False),
        ("Закажи 1 штука P-1042B", 1, True),
        ("Закажи 2 штуки P-1042B", 2, True),
        ("Закажи 5 штук P-1042B", 5, True),
        ("Количество: 2, деталь P-1042B", 2, True),
        ("Закажи -2 шт.", 2, False),
        ("Закажи 1.5 шт.", 5, False),
        ("Закажи 1 шт.", True, False),
    ],
)
def test_quantity_requires_an_explicit_whole_number(question, qty, expected):
    assert quantity_is_explicit(question, qty) is expected


@pytest.mark.parametrize("arguments", [{}, {"part_id": "P-1042B", "kb": "injected"}, {"part_id": []}])
def test_schema_validation_rejects_arguments_even_for_kwargs_tools(arguments):
    assert validate_tool_call("get_price", arguments, {"get_price": lambda **kwargs: {}}, {"P-1042B"})


def test_number_formats_and_zero_are_preserved():
    prices, quantities = extract_answer_numbers("Цена: 2 630,00 рублей. Остаток на складе: 0.")
    assert prices == {"2630"}
    assert quantities == {"0"}
    assert normalize_number("0,00") == "0"


def test_supported_price_and_zero_stock_do_not_trigger_correction():
    steps = [
        ToolStep("get_price", {}, {"price_rub": 2630}),
        ToolStep("check_stock", {}, {"qty": 0}),
    ]
    assert find_answer_errors("Цена: 2 630 руб. В наличии 0 шт.", "?", steps) == []


def test_invented_order_number_is_rejected():
    assert find_answer_errors("Создан заказ SO-999999.", "Оформи заказ.", [])


def test_price_of_another_part_does_not_confirm_the_answer():
    steps = [ToolStep("get_price", {"part_id": "P-1042B"}, {"part_id": "P-1042B", "price_rub": 2630})]
    assert find_answer_errors("Цена P-1010: 2630 руб.", "Сравни P-1010 и P-1042B.", steps)


def test_stock_of_another_part_does_not_confirm_the_answer():
    steps = [ToolStep("check_stock", {"part_id": "P-1042B"}, {"part_id": "P-1042B", "qty": 3})]
    assert find_answer_errors("P-1010: в наличии 3 штуки.", "Сравни P-1010 и P-1042B.", steps)


def test_pending_order_is_recorded_without_executing_it():
    args = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}

    def forbidden(**kwargs):
        pytest.fail("Неподтверждённый заказ не должен исполняться")

    llm = FakeLLM([LLMResponse("", [ToolCall("c1", "create_service_order", args)])])
    result = run_agent("Закажи 2 штуки P-1042B для AR-210.", llm, {"create_service_order": forbidden})
    assert result.needs_confirmation
    assert len(result.steps) == 1
    assert result.steps[0].args == args


def test_unavailable_erp_is_reported_even_if_model_omits_it():
    def unavailable(**kwargs):
        raise ErpUnavailable("offline")

    llm = FakeLLM(
        [
            LLMResponse("", [ToolCall("c1", "get_price", {"part_id": "P-1042B"})]),
            "Не получилось.",
        ]
    )
    result = run_agent("Сколько стоит P-1042B?", llm, {"get_price": unavailable})
    assert "Данные ERP получить не удалось" in result.answer


def test_created_order_number_is_not_lost_in_a_short_model_answer():
    args = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}
    llm = FakeLLM([LLMResponse("", [ToolCall("c1", "create_service_order", args)]), "Готово."])
    result = run_agent(
        "Закажи 2 штуки P-1042B для AR-210.",
        llm,
        {"create_service_order": lambda **kwargs: {"order_id": "SO-000001", **kwargs}},
        confirm=True,
    )
    assert "SO-000001" in result.answer


def test_empty_final_answer_gets_one_correction():
    llm = FakeLLM(["", "Уточните модель оборудования."])
    result = run_agent("Нужна помощь.", llm, {})
    assert result.answer == "Уточните модель оборудования."
    assert len(llm.requests) == 2


@pytest.mark.parametrize("text", ["Деталь не найдена.", "Данные не найдены.", "Нет информации."])
def test_refusal_does_not_depend_on_grammatical_gender(text):
    assert is_refusal(text)
