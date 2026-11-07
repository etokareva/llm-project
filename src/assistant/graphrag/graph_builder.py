"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult, Relation


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты извлечения в направленный граф знаний.

    Объединяет сущности по Entity.node_id, сохраняя type, name и sources.
    Добавляет соответствующие схеме связи с атрибутами type, weight и sources.
    Источники представлены уникальными chunk_id; вес связи равен их числу.
    Разрешает встречные замены деталей и записывает число конфликтующих пар
    в graph.graph["conflicts"]. Возвращает построенный граф.
    """

    graph = nx.DiGraph()

    for result in results:
        for entity in result.entities:
            add_entity(graph, entity, result.chunk_id)

        for relation in result.relations:
            add_relation(graph, relation, result.chunk_id)

    graph.graph["conflicts"] = resolve_replacement_conflicts(graph)

    return graph


def add_entity(graph: nx.DiGraph, entity: Entity, chunk_id: str) -> None:
    """Добавляет узел сущности или дополняет его источники.

    Использует Entity.node_id как идентификатор узла. При создании сохраняет
    type, name и список sources. Добавляет chunk_id в sources без повторов.
    Изменяет переданный граф.
    """
    node_id = entity.node_id

    if node_id not in graph:
        graph.add_node(
            node_id,
            type=entity.type,
            name=entity.name,
            sources=[],
        )

    sources = graph.nodes[node_id]["sources"]
    if chunk_id not in sources:
        sources.append(chunk_id)


def add_relation(graph: nx.DiGraph, relation: Relation, chunk_id: str) -> None:
    """Добавляет связь, соответствующую схеме, и обновляет её источники и вес.

    Пропускает связь, если Relation.matches_schema() возвращает False.
    Сохраняет тип связи и уникальные chunk_id в sources.
    Вес weight равен числу уникальных источников.
    Ожидает, что узлы обоих концов уже добавлены с атрибутами.
    Изменяет переданный граф.
    """
    if not relation.matches_schema():
        return

    for entity in (relation.subject, relation.object):
        add_entity(graph, entity, chunk_id)

    subject_id = relation.subject.node_id
    object_id = relation.object.node_id

    if not graph.has_edge(subject_id, object_id):
        graph.add_edge(
            subject_id,
            object_id,
            type=relation.type,
            weight=0,
            sources=[],
        )

    edge = graph.edges[subject_id, object_id]
    if chunk_id not in edge["sources"]:
        edge["sources"].append(chunk_id)

    edge["weight"] = len(edge["sources"])


def resolve_replacement_conflicts(graph: nx.DiGraph) -> int:
    """Удаляет противоречащие друг другу связи замены деталей.

    Конфликтом считается пара разных узлов с REPLACED_BY в обе стороны.
    Сохраняет направление с большим весом; при равных весах удаляет обе связи.
    Каждую конфликтующую пару учитывает один раз.
    Изменяет переданный граф и возвращает число найденных конфликтов.
    """
    conflicts = 0
    checked: set[frozenset[str]] = set()

    for subject_id, object_id in list(graph.edges):
        if not graph.has_edge(subject_id, object_id):
            continue

        edge = graph.edges[subject_id, object_id]
        if edge["type"] != "REPLACED_BY" or subject_id == object_id:
            continue

        pair = frozenset((subject_id, object_id))
        if pair in checked:
            continue
        checked.add(pair)

        if not graph.has_edge(object_id, subject_id):
            continue

        reverse = graph.edges[object_id, subject_id]
        if reverse["type"] != "REPLACED_BY":
            continue

        conflicts += 1
        forward_weight = edge["weight"]
        reverse_weight = reverse["weight"]

        if forward_weight <= reverse_weight:
            graph.remove_edge(subject_id, object_id)
        if reverse_weight <= forward_weight:
            graph.remove_edge(object_id, subject_id)

    return conflicts
