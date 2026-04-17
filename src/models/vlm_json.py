from __future__ import annotations

import json
import re
from typing import Any, TypedDict


class DefectJson(TypedDict):
    coordinates: list[list[int]]
    defect_type: str
    reasoning: str


def _extract_json_block(text: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced_match is not None:
        fenced_candidate = fenced_match.group(1)
        object_match = re.search(r"\{[\s\S]*\}", fenced_candidate)
        if object_match is not None:
            return object_match.group(0)

    match = re.search(r"\{[\s\S]*\}", text)
    if match is None:
        msg = "Model output does not contain a JSON object"
        raise ValueError(msg)

    return match.group(0)


def _normalize_coordinates_payload(coordinates: object) -> list[list[object]]:
    if (
        isinstance(coordinates, list)
        and len(coordinates) == 4
        and all(not isinstance(item, list) for item in coordinates)
    ):
        return [coordinates]
    if not isinstance(coordinates, list):
        msg = "coordinates must be a list"
        raise TypeError(msg)
    return coordinates


def _coerce_coordinate_value(value: object) -> int:
    if isinstance(value, bool):
        msg = "boolean coordinate values are not allowed"
        raise TypeError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)

    msg = "all coordinate values must be integers"
    raise TypeError(msg)


def _extract_partial_defect_json(text: str) -> DefectJson:
    coordinates_anchor = re.search(r'"coordinates"\s*:', text)
    if coordinates_anchor is None:
        msg = "Model output does not contain a JSON object"
        raise ValueError(msg)

    coordinates_tail = text[coordinates_anchor.end() :]
    coordinate_values = re.findall(r"-?\d+(?:\.\d+)?", coordinates_tail)
    if len(coordinate_values) < 4:
        msg = "Model output does not contain complete coordinate values"
        raise ValueError(msg)

    candidate_box = [float(value) for value in coordinate_values[:4]]

    defect_type_match = re.search(r'"defect_type"\s*:\s*"([^"\n}]*)', text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"\n}]*)', text)

    defect_type = (
        defect_type_match.group(1).strip()
        if defect_type_match is not None and defect_type_match.group(1).strip()
        else "Unknown"
    )
    reasoning = (
        reasoning_match.group(1).strip()
        if reasoning_match is not None and reasoning_match.group(1).strip()
        else "Model output truncated"
    )

    partial_payload = {
        "coordinates": [candidate_box],
        "defect_type": defect_type,
        "reasoning": reasoning,
    }

    return validate_defect_json(partial_payload)


def extract_defect_json(text: str) -> DefectJson:
    try:
        parsed = json.loads(_extract_json_block(text))
    except (ValueError, json.JSONDecodeError):
        return _extract_partial_defect_json(text)

    if not isinstance(parsed, dict):
        msg = "Parsed output is not a JSON object"
        raise TypeError(msg)

    return validate_defect_json(parsed)


def validate_defect_json(parsed: dict[str, Any]) -> DefectJson:
    required_keys = {"coordinates", "defect_type", "reasoning"}
    if set(parsed.keys()) != required_keys:
        msg = f"Unexpected JSON keys: {sorted(parsed.keys())}"
        raise ValueError(msg)

    coordinates = _normalize_coordinates_payload(parsed["coordinates"])
    defect_type = parsed["defect_type"]
    reasoning = parsed["reasoning"]

    validated_coordinates: list[list[int]] = []
    for box in coordinates:
        if len(box) != 4:
            msg = "each coordinates element must be a list of 4 integers"
            raise ValueError(msg)
        validated_box: list[int] = []
        for value in box:
            validated_box.append(_coerce_coordinate_value(value))
        validated_coordinates.append(validated_box)

    if not isinstance(defect_type, str) or not defect_type.strip():
        msg = "defect_type must be a non-empty string"
        raise ValueError(msg)

    if not isinstance(reasoning, str) or not reasoning.strip():
        msg = "reasoning must be a non-empty string"
        raise ValueError(msg)

    return {
        "coordinates": validated_coordinates,
        "defect_type": defect_type.strip(),
        "reasoning": reasoning.strip(),
    }
