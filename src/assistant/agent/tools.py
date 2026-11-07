"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable
from functools import partial

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": ("Найти по документам сведения о моделях, ошибках, деталях и их актуальных заменах."),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Конкретный вопрос по документам.",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": ("Ответить на обзорный вопрос по сводкам сообществ графа."),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Обзорный вопрос по корпусу документов.",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": ("Получить остаток детали на выбранном складе и признак снятия с производства."),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "pattern": "^P-[0-9]{4}[A-Z]?$",
                        "description": "Код детали, например P-1042B.",
                    },
                    "warehouse": {
                        "type": "string",
                        "enum": WAREHOUSES,
                    },
                },
                "required": ["part_id", "warehouse"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": ("Получить текущую цену детали в рублях и признак снятия с производства."),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "pattern": "^P-[0-9]{4}[A-Z]?$",
                        "description": "Код детали, например P-1042B.",
                    },
                },
                "required": ["part_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": (
                "Создать заказ детали. Агент должен получить подтверждение "
                "человека до выполнения инструмента. "
                "Количество должно быть явно указано пользователем."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "pattern": "^P-[0-9]{4}[A-Z]?$",
                    },
                    "warehouse": {
                        "type": "string",
                        "enum": WAREHOUSES,
                    },
                    "qty": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Количество деталей в заказе.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Модель оборудования, например AR-210.",
                    },
                },
                "required": ["part_id", "warehouse", "qty", "model"],
                "additionalProperties": False,
            },
        },
    },
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    return {
        "search_knowledge": partial(search_knowledge, kb=kb, llm=llm),
        "summarize_topic": partial(summarize_topic, kb=kb, llm=llm),
        "check_stock": partial(check_stock, erp=erp),
        "get_price": partial(get_price, erp=erp),
        "create_service_order": partial(create_service_order, erp=erp),
    }


def search_knowledge(
    question: str,
    *,
    kb: KnowledgeBase,
    llm: LLMClient,
) -> dict:
    """Возвращает ответ локального поиска, факты и документы-источники."""
    if not isinstance(question, str) or not question.strip():
        return {"error": "Вопрос должен быть непустой строкой."}

    answer = local_search(question, kb, llm)
    return {
        "answer": answer.text,
        "facts": answer.facts,
        "sources": answer.sources,
        "refused": answer.refused,
    }


def summarize_topic(
    question: str,
    *,
    kb: KnowledgeBase,
    llm: LLMClient,
) -> dict:
    """Возвращает обзорный ответ по сводкам и документы-источники."""
    if not isinstance(question, str) or not question.strip():
        return {"error": "Вопрос должен быть непустой строкой."}

    answer = global_search(question, kb, llm)
    return {
        "answer": answer.text,
        "sources": answer.sources,
        "refused": answer.refused,
    }


def check_stock(part_id: str, warehouse: str, *, erp: ErpClient) -> dict:
    """Возвращает остаток детали после проверки кода и склада."""
    error = validate_part_id(part_id)
    if error:
        return {"error": error}

    if warehouse not in WAREHOUSES:
        return {"error": "Неизвестный склад."}

    return call_erp(erp.get_stock, part_id=part_id, warehouse=warehouse)


def get_price(part_id: str, *, erp: ErpClient) -> dict:
    """Возвращает текущую цену детали после проверки её кода."""
    error = validate_part_id(part_id)
    if error:
        return {"error": error}

    return call_erp(erp.get_price, part_id=part_id)


def create_service_order(
    part_id: str,
    warehouse: str,
    qty: int,
    model: str,
    *,
    erp: ErpClient,
) -> dict:
    """Создаёт заказ после проверки аргументов.

    Подтверждение человеком должен проверить агент до вызова инструмента.
    """
    error = validate_part_id(part_id)
    if error:
        return {"error": error}

    if warehouse not in WAREHOUSES:
        return {"error": "Неизвестный склад."}

    if type(qty) is not int or qty <= 0:
        return {"error": "Количество должно быть положительным целым числом."}

    if not isinstance(model, str) or not re.fullmatch(r"(?:AR|BZ|GS)-[0-9]{3}", model):
        return {"error": "Некорректный код модели оборудования."}

    return call_erp(
        erp.create_order,
        part_id=part_id,
        warehouse=warehouse,
        qty=qty,
        model=model,
    )


def validate_part_id(part_id: str) -> str | None:
    """Возвращает описание ошибки формата кода детали или None."""
    if not isinstance(part_id, str) or not re.fullmatch(r"P-[0-9]{4}[A-Z]?", part_id):
        return "Код детали должен иметь вид P-1042 или P-1042B."

    return None


def call_erp(operation: Callable[..., dict], **arguments) -> dict:
    """Вызывает ERP, различая отсутствие данных и отклонённый запрос.

    ErpUnavailable передаёт вызывающему коду: повторы выполняет агент.
    """
    try:
        return operation(**arguments)
    except ErpNotFound as error:
        return {
            "error": f"Данные в ERP не найдены: {error}",
            "error_type": "not_found",
        }
    except ErpRejected as error:
        return {"error": f"Запрос отклонён ERP: {error}"}
