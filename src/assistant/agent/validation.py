"""Проверки аргументов агента и происхождения фактов в ответе."""

import json
import re
from decimal import Decimal

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.schemas import ToolStep

NUMBER = r"(?<![\w.,-])((?:[0-9]{1,3}(?:[ \u00a0\u202f][0-9]{3})+|[0-9]+)(?:[.,][0-9]+)?)(?![\w-]|[.,][0-9])"
PRICE_PATTERNS = (
    re.compile(NUMBER + r"\s*(?:руб(?:лей|ля|ль)?\.?|₽)", re.IGNORECASE),
    re.compile(r"\bцен[аыу]\s*(?::|=|—|-|составляет)?\s*" + NUMBER, re.IGNORECASE),
)
QUANTITY_PATTERNS = (
    re.compile(NUMBER + r"\s*(?:штук[аи]?|шт\.?)(?!\w)", re.IGNORECASE),
    re.compile(
        r"\bостат(?:ок|ка)\b(?:\s+на\s+складе)?\s*(?::|=|—|-|составляет|равен)?\s*" + NUMBER,
        re.IGNORECASE,
    ),
    re.compile(r"\bв\s+наличии\s*(?::|=|—|-)?\s*" + NUMBER, re.IGNORECASE),
)
ORDER_CODE = re.compile(r"\bSO-[0-9]+\b", re.IGNORECASE)
CODE_CANDIDATE = re.compile(r"[A-Za-zА-Яа-яЁё]{1,2}[- ]?[0-9]{2,4}[A-Za-zА-Яа-яЁё]?")
PARAMETERS = {schema["function"]["name"]: schema["function"]["parameters"] for schema in TOOL_SCHEMAS}


def validate_tool_call(name: str, arguments: dict, tools: dict, known_codes: set[str]) -> str | None:
    """Проверяет модельные аргументы по публичной схеме, не вызывая инструмент."""
    if name not in tools or name not in PARAMETERS:
        return f"Неизвестный инструмент: {name}."
    if not isinstance(arguments, dict) or "_raw" in arguments:
        return "Аргументы инструмента не удалось разобрать как JSON."

    parameters = PARAMETERS[name]
    properties = parameters["properties"]
    if set(parameters.get("required", [])) - arguments.keys():
        return "Не переданы обязательные аргументы инструмента."
    if arguments.keys() - properties.keys():
        return "Переданы неизвестные аргументы инструмента."

    for key, value in arguments.items():
        spec = properties[key]
        if spec["type"] == "string" and not isinstance(value, str):
            return f"Аргумент {key} должен быть строкой."
        if spec["type"] == "integer" and type(value) is not int:
            return f"Аргумент {key} должен быть целым числом."
        if "enum" in spec and value not in spec["enum"]:
            return f"Недопустимое значение аргумента {key}."
        if "pattern" in spec and re.fullmatch(spec["pattern"], value) is None:
            return f"Некорректный формат аргумента {key}."
        if "minimum" in spec and value < spec["minimum"]:
            return f"Аргумент {key} меньше допустимого значения."

    if "part_id" in arguments and arguments["part_id"] not in known_codes:
        return "Номер детали не подтверждён. Сначала вызови search_knowledge."
    return None


def collect_known_codes(question: str, steps: list[ToolStep]) -> set[str]:
    """Собирает коды из вопроса и успешных результатов, включая номера заказов."""
    texts = [
        question,
        *(
            json.dumps(step.result, ensure_ascii=False)
            for step in steps
            if step.error is None and step.result is not None
        ),
    ]
    return {
        code for text in texts for code in (*find_codes(text), *(match.upper() for match in ORDER_CODE.findall(text)))
    }


def quantity_is_explicit(question: str, qty: int) -> bool:
    """Принимает положительное количество с единицами или явной подписью."""
    if type(qty) is not int or qty <= 0:
        return False
    text = CODE_CANDIDATE.sub(lambda match: " " if normalize_code(match[0]) else match[0], question)
    patterns = (
        r"(?<![\w.,-])([0-9]+)\s*(?:штук[аи]?|шт\.?)(?!\w)",
        r"\bколичество\s*[:=]?\s*([0-9]+)(?!\w|[.,][0-9])",
    )
    return any(str(qty) in re.findall(pattern, text, re.IGNORECASE) for pattern in patterns)


def normalize_number(value: str) -> str:
    """Нормализует десятичное число без потери точности через float."""
    number = Decimal(re.sub(r"\s+", "", value).replace(",", "."))
    return format(number, "f").rstrip("0").rstrip(".") if number % 1 else str(int(number))


def extract_answer_numbers(text: str) -> tuple[set[str], set[str]]:
    """Распознаёт числовые цены и количества в поддерживаемых текстовых формах."""
    text = CODE_CANDIDATE.sub(lambda match: " " if normalize_code(match[0]) else match[0], text)
    prices = {normalize_number(value) for pattern in PRICE_PATTERNS for value in pattern.findall(text)}
    quantities = {normalize_number(value) for pattern in QUANTITY_PATTERNS for value in pattern.findall(text)}
    return prices, quantities


def collect_supported_numbers(steps: list[ToolStep]) -> tuple[set[str], set[str]]:
    """Собирает цены и количества из успешных структурированных результатов."""
    prices: set[str] = set()
    quantities: set[str] = set()
    for step in steps:
        if step.error is not None or step.result is None:
            continue
        if step.tool == "get_price" and "price_rub" in step.result:
            prices.add(normalize_number(str(step.result["price_rub"])))
        if step.tool in ("check_stock", "create_service_order") and "qty" in step.result:
            quantities.add(normalize_number(str(step.result["qty"])))
    return prices, quantities


def find_answer_errors(text: str, question: str, steps: list[ToolStep]) -> list[str]:
    """Проверяет коды и числовые факты; не является проверкой всех утверждений текста."""
    known_codes = collect_known_codes(question, steps)
    answer_codes = set(find_codes(text)) | {code.upper() for code in ORDER_CODE.findall(text)}
    prices, quantities = collect_supported_numbers(steps)
    question_prices, question_quantities = extract_answer_numbers(question)
    answer_prices, answer_quantities = extract_answer_numbers(text)
    errors = [f"неизвестный код {code}" for code in sorted(answer_codes - known_codes)]
    errors.extend(f"неподтверждённая цена {value}" for value in sorted(answer_prices - prices - question_prices))
    errors.extend(
        f"неподтверждённое количество {value}" for value in sorted(answer_quantities - quantities - question_quantities)
    )
    errors.extend(find_part_number_errors(text, question, steps))
    return list(dict.fromkeys(errors))


def find_part_number_errors(text: str, question: str, steps: list[ToolStep]) -> list[str]:
    """Проверяет привязку чисел к детали в предложениях с одним кодом детали.

    Сложные предложения с несколькими деталями не интерпретирует.
    """
    errors = []
    question_prices, question_quantities = extract_answer_numbers(question)
    for fragment in re.split(r"\n|(?<=[.!?])\s+", text):
        parts = [code for code in find_codes(fragment) if code.startswith("P-")]
        if len(parts) != 1:
            continue
        part_id = parts[0]
        relevant = [
            step
            for step in steps
            if step.result is not None and step.result.get("part_id", step.args.get("part_id")) == part_id
        ]
        prices, quantities = collect_supported_numbers(relevant)
        answer_prices, answer_quantities = extract_answer_numbers(fragment)
        if answer_prices - prices - question_prices:
            errors.append(f"цена не подтверждена для детали {part_id}")
        if answer_quantities - quantities - question_quantities:
            errors.append(f"количество не подтверждено для детали {part_id}")
    return errors
