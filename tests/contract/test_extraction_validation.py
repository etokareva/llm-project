"""Дополнительные проверки некорректных ответов при извлечении."""

import json

import pytest

from assistant.graphrag.extraction import parse_response


@pytest.mark.parametrize("relation_type", [[], {}, None, 123])
def test_invalid_relation_type_raises_value_error(relation_type):
    content = json.dumps(
        {
            "entities": [],
            "relations": [
                {
                    "subject_type": "Part",
                    "subject": "P-1042",
                    "type": relation_type,
                    "object_type": "Part",
                    "object": "P-1042B",
                }
            ],
        }
    )

    with pytest.raises(ValueError):
        parse_response(content)