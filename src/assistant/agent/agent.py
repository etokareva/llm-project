"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import inspect
import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Ты ассистент сервисной службы. "
                "Используй инструменты для поиска фактов, цен и остатков. "
                "Не придумывай номера деталей и параметры заказа. "
                "Если количество не указано, уточни его у пользователя. "
                "При ошибке инструмента учитывай её в дальнейшем ответе."
            ),
        },
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    completed: dict[str, ToolStep] = {}

    while len(steps) < max_steps:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())

        if not response.tool_calls:
            return finalize_answer(
                response.content,
                question,
                steps,
                sources,
                messages,
                llm,
            )

        for call in response.tool_calls:
            if len(steps) >= max_steps:
                break

            error = validate_tool_call(
                call.name,
                call.arguments,
                tools,
                collect_known_codes(question, steps),
            )

            if error:
                step = ToolStep(
                    tool=call.name,
                    args=call.arguments,
                    error=error,
                )
                record_tool_step(call, step, steps, messages, sources)
                continue

            pause = check_order_pause(question, call, confirm)
            if pause is not None:
                pause.steps = list(steps)
                pause.sources = sorted(sources)
                return pause

            step = execute_cached_tool(
                call,
                tools[call.name],
                completed,
            )
            record_tool_step(call, step, steps, messages, sources)

    observations = [
        step.result
        for step in steps
        if step.error is None and step.result is not None
    ]

    return AgentResult(
        answer=(
            "Работа прервана: достигнут лимит вызовов инструментов. "
            "Полученные данные: "
            + json.dumps(observations, ensure_ascii=False)
        ),
        steps=steps,
        sources=sorted(sources),
    )


def finalize_answer(
    text: str,
    question: str,
    steps: list[ToolStep],
    sources: set[str],
    messages: list[dict],
    llm: LLMClient,
) -> AgentResult:
    """Проверяет ответ и допускает одну попытку исправления.

    При повторном расхождении возвращает отказ без неподтверждённых фактов.
    """
    errors = find_answer_errors(text, question, steps)

    if errors:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Исправь ответ: обнаружены неподтверждённые сведения: "
                    + "; ".join(errors)
                    + ". Используй только вопрос и результаты инструментов. "
                    "Верни окончательный текст без вызова инструментов."
                ),
            }
        )
        correction = llm.chat(messages)
        text = correction.content

        if correction.tool_calls or find_answer_errors(text, question, steps):
            return AgentResult(
                answer=(
                    "Не удалось подтвердить сведения в ответе. "
                    "В базе знаний не найдено достаточно проверенных данных."
                ),
                steps=list(steps),
                sources=sorted(sources),
                refused=True,
            )

    return AgentResult(
        answer=text,
        steps=list(steps),
        sources=sorted(sources),
        refused="не найдено" in text.lower(),
    )


def find_answer_errors(
    text: str,
    question: str,
    steps: list[ToolStep],
) -> list[str]:
    """Находит неизвестные коды и неподтверждённые цены и остатки.

    Цены распознаёт по обозначениям рублей, количества — по шт./штук.
    Числовые значения сверяет с успешными результатами инструментов.
    """
    errors = []
    known_codes = collect_known_codes(question, steps)

    for code in find_codes(text):
        if code not in known_codes:
            errors.append(f"неизвестный код {code}")

    prices: set[str] = set()
    quantities: set[str] = set()

    for step in steps:
        if step.error is not None or step.result is None:
            continue

        if step.tool == "get_price" and "price_rub" in step.result:
            prices.add(normalize_number(str(step.result["price_rub"])))

        if step.tool in ("check_stock", "create_service_order"):
            if "qty" in step.result:
                quantities.add(normalize_number(str(step.result["qty"])))

    number = r"([0-9]+(?:[ \u00a0\u202f][0-9]{3})*(?:[.,][0-9]+)?)"

    for value in re.findall(
        number + r"\s*(?:руб(?:лей|ля|ль)?\.?|₽)",
        text,
        flags=re.IGNORECASE,
    ):
        if normalize_number(value) not in prices:
            errors.append(f"неподтверждённая цена {value}")

    for value in re.findall(
        number + r"\s*(?:шт(?:ук[аи]?)?\.?)",
        text,
        flags=re.IGNORECASE,
    ):
        if normalize_number(value) not in quantities:
            errors.append(f"неподтверждённое количество {value}")

    return errors


def normalize_number(value: str) -> str:
    """Убирает разделители разрядов и нормализует десятичную запись."""
    compact = re.sub(r"\s+", "", value).replace(",", ".")
    if "." in compact:
        compact = compact.rstrip("0").rstrip(".")
    return compact


def check_order_pause(
    question: str,
    call: ToolCall,
    confirm: bool,
) -> AgentResult | None:
    """Возвращает запрос уточнения или подтверждения перед заказом.

    Для остальных инструментов и подтверждённого заказа возвращает None.
    Ожидает предварительно проверенные аргументы вызова.
    """
    if call.name != "create_service_order":
        return None

    if not quantity_is_explicit(question, call.arguments.get("qty")):
        return AgentResult(
            answer="Уточните, сколько штук нужно заказать.",
        )

    if not confirm:
        return AgentResult(
            answer="Подтвердите создание заказа.",
            needs_confirmation=True,
            pending_action={
                "tool": call.name,
                "args": dict(call.arguments),
            },
        )

    return None


def execute_cached_tool(
    call: ToolCall,
    tool: Callable[..., dict],
    completed: dict[str, ToolStep],
) -> ToolStep:
    """Выполняет инструмент или возвращает ранее полученный результат.

    Дополняет completed. Для повторного вызова создаёт отдельный шаг,
    сохраняя прежние результат и ошибку.
    """
    key = json.dumps(
        {"tool": call.name, "args": call.arguments},
        sort_keys=True,
        ensure_ascii=False,
    )

    if key not in completed:
        completed[key] = execute_tool(call.name, call.arguments, tool)

    previous = completed[key]
    return ToolStep(
        tool=call.name,
        args=dict(call.arguments),
        result=previous.result,
        error=previous.error,
    )


def record_tool_step(
    call: ToolCall,
    step: ToolStep,
    steps: list[ToolStep],
    messages: list[dict],
    sources: set[str],
) -> None:
    """Записывает шаг, собирает источники и передаёт наблюдение модели."""
    steps.append(step)

    if step.error is None and step.result is not None:
        sources.update(step.result.get("sources", []))

    observation = (
        {"error": step.error}
        if step.error is not None
        else step.result
    )
    messages.append(tool_message(call, observation))


def execute_tool(
    name: str,
    arguments: dict,
    tool: Callable[..., dict],
) -> ToolStep:
    """Вызывает инструмент, повторяя запрос при недоступности ERP.

    Выполняет не более трёх попыток. Возвращает шаг с результатом
    или ошибкой. Проверку разрешения на действие выполняет агент
    до вызова этой функции.
    """
    for attempt in range(3):
        try:
            result = tool(**arguments)
        except ErpUnavailable as error:
            if attempt < 2:
                continue

            return ToolStep(
                tool=name,
                args=arguments,
                error=f"Данные ERP получить не удалось: {error}",
            )

        return ToolStep(
            tool=name,
            args=arguments,
            result=result,
            error=result.get("error"),
        )

    raise RuntimeError("Цикл попыток завершился без результата")


def validate_tool_call(
    name: str,
    arguments: dict,
    tools: dict[str, Callable[..., dict]],
    known_codes: set[str],
) -> str | None:
    """Проверяет инструмент, аргументы и происхождение номера детали.

    Возвращает описание ошибки или None, если вызов допустим.
    Подтверждение заказа и количество проверяются отдельно.
    """
    if name not in tools:
        return f"Неизвестный инструмент: {name}."

    if not isinstance(arguments, dict):
        return "Аргументы инструмента должны быть словарём."

    if "_raw" in arguments:
        return "Аргументы инструмента не удалось разобрать как JSON."

    try:
        inspect.signature(tools[name]).bind(**arguments)
    except TypeError:
        return "Неверные аргументы инструмента."

    if "part_id" in arguments:
        part_id = arguments["part_id"]

        if not isinstance(part_id, str) or part_id not in known_codes:
            return (
                "Номер детали не встречался в вопросе или результатах "
                "инструментов. Сначала вызови search_knowledge."
            )

    return None


def collect_known_codes(question: str, steps: list[ToolStep]) -> set[str]:
    """Собирает коды из вопроса и успешных результатов инструментов."""
    codes = set(find_codes(question))

    for step in steps:
        if step.error is None and step.result is not None:
            text = json.dumps(step.result, ensure_ascii=False)
            codes.update(find_codes(text))

    return codes


def quantity_is_explicit(question: str, qty: int) -> bool:
    """Проверяет положительное целое количество среди чисел вопроса.

    Исключает коды оборудования, деталей, ошибок и регламентов,
    чтобы их цифры не принимались за количество.
    """
    if type(qty) is not int or qty <= 0:
        return False

    text = question
    for code in find_codes(question):
        text = re.sub(
            re.escape(code),
            "",
            text,
            flags=re.IGNORECASE,
        )

    quantities = re.findall(r"(?<!\w)[0-9]+(?!\w)", text)
    return str(qty) in quantities