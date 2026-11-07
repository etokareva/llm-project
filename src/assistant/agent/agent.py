"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
from collections.abc import Callable
from copy import deepcopy

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.agent.validation import (
    collect_known_codes,
    find_answer_errors,
    quantity_is_explicit,
    validate_tool_call,
)
from assistant.graphrag.search import is_refusal
from assistant.infra.erp_client import ErpUnavailable
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
                pause.steps = [
                    *steps,
                    ToolStep(
                        tool=call.name,
                        args=dict(call.arguments),
                        result={"needs_confirmation": pause.needs_confirmation},
                    ),
                ]
                return complete_result(pause, pause.steps, sources)

            step = execute_cached_tool(
                call,
                tools[call.name],
                completed,
            )
            record_tool_step(call, step, steps, messages, sources)

    observations = [step.result for step in steps if step.error is None and step.result is not None]

    return complete_result(
        AgentResult(
            answer=(
                "Работа прервана: достигнут лимит вызовов инструментов. "
                "Полученные данные: " + json.dumps(observations, ensure_ascii=False)
            ),
            refused=tools_report_no_data(steps),
        ),
        steps,
        sources,
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
    if not text.strip():
        errors.append("пустой ответ")

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

        if not text.strip() or correction.tool_calls or find_answer_errors(text, question, steps):
            return complete_result(
                AgentResult(
                    answer=(
                        "Не удалось подтвердить сведения в ответе. "
                        "В базе знаний не найдено достаточно проверенных данных."
                    ),
                    refused=True,
                ),
                steps,
                sources,
            )

    refused = tools_report_no_data(steps) or is_refusal(text)

    return complete_result(AgentResult(answer=text, refused=refused), steps, sources)


def complete_result(result: AgentResult, steps: list[ToolStep], sources: set[str]) -> AgentResult:
    """Завершает любой исход агента с учётом уже выполненных действий и ошибок ERP.

    Сохраняет шаги и источники. Номер созданного заказа не зависит от ответа
    модели, а недоступность ERP остаётся видимой даже при исчерпании лимита.
    """
    result.steps = list(steps)
    result.sources = sorted(sources)

    order_ids = [
        str(step.result["order_id"])
        for step in steps
        if step.tool == "create_service_order" and step.error is None and step.result and "order_id" in step.result
    ]
    if order_ids and result.refused:
        result.answer = "Заказ создан. Дополнительные сведения из ответа модели не удалось подтвердить."
        result.refused = False
    if result.refused and "в базе знаний не найдено" not in result.answer.lower():
        result.answer += "\nВ базе знаний не найдено."
    for order_id in dict.fromkeys(order_ids):
        if order_id not in result.answer:
            result.answer += f"\nНомер созданного заказа: {order_id}."

    if any(step.error and step.error.startswith("Данные ERP получить не удалось:") for step in steps):
        notice = "Данные ERP получить не удалось."
        if notice.lower().rstrip(".") not in result.answer.lower():
            result.answer = f"{result.answer}\n{notice}"

    return result


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
        result=deepcopy(previous.result),
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

    observation = {"error": step.error} if step.error is not None else step.result
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


def tools_report_no_data(steps: list[ToolStep]) -> bool:
    """Определяет отсутствие данных по структурированным результатам.

    Возвращает True, если инструменты сообщили об отсутствии данных
    и ни один успешный вызов не предоставил содержательного результата.
    Нулевой остаток считается полученными данными.
    """
    no_data = False
    has_data = False

    for step in steps:
        result = step.result
        if result is None:
            continue

        if result.get("error_type") == "not_found":
            no_data = True
            continue

        if step.error is not None:
            continue

        if result.get("refused") is True:
            no_data = True
            continue

        if step.tool in ("search_knowledge", "summarize_topic"):
            has_data = has_data or bool(result.get("answer"))
        elif step.tool == "get_price":
            has_data = has_data or "price_rub" in result
        elif step.tool == "check_stock":
            has_data = has_data or "qty" in result
        elif step.tool == "create_service_order":
            has_data = has_data or "order_id" in result

    return no_data and not has_data
