"""Дополнительные проверки аргументов и числовых фактов агента."""

from assistant.agent.agent import (
    find_answer_errors,
    quantity_is_explicit,
)


def test_model_digits_are_not_order_quantity():
    question = "Закажи P-1042B для ar 210, склад Казань."
    assert not quantity_is_explicit(question, 210)


def test_explicit_quantity_is_accepted():
    assert quantity_is_explicit(
        "Закажи 2 штуки P-1042B для AR-210.",
        2,
    )


def test_unsupported_stock_without_units_is_rejected():
    errors = find_answer_errors(
        "Остаток на складе: 99.",
        "Есть ли P-1042B?",
        [],
    )
    assert errors


def test_unsupported_price_without_units_is_rejected():
    errors = find_answer_errors(
        "Цена: 4850.",
        "Сколько стоит P-1042B?",
        [],
    )
    assert errors