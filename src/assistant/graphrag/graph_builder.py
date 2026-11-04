"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult, Relation


def add_entity(graph: nx.DiGraph, entity: Entity, chunk_id: str) -> None:
    """Добавляет сущность в граф и сохраняет источник без повторов."""
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
    """Добавляет допустимую связь и считает вес по уникальным источникам."""
    if not relation.matches_schema():
        return

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
    """Разрешает встречные REPLACED_BY и возвращает число конфликтующих пар.

    Сохраняет связь с большим весом. При равных весах удаляет обе.
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


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты извлечения в направленный граф знаний.

    Узлы содержат type, name и sources. Связи содержат type, weight и sources.
    Вес связи равен числу уникальных чанков, где она найдена.
    Число конфликтующих пар замен сохраняется в graph.graph["conflicts"].
    """
    graph = nx.DiGraph()

    for result in results:
        for entity in result.entities:
            add_entity(graph, entity, result.chunk_id)

        for relation in result.relations:
            add_relation(graph, relation, result.chunk_id)

    graph.graph["conflicts"] = resolve_replacement_conflicts(graph)
    
    return graph