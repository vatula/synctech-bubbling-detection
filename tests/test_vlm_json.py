from src.models.vlm_json import extract_defect_json


def test_extract_defect_json_valid_schema() -> None:
    raw = (
        "prefix "
        '{"coordinates": [[1, 2, 3, 4]], '
        '"defect_type": "bubbling", '
        '"reasoning": "surface blister"} '
        "suffix"
    )
    parsed = extract_defect_json(raw)

    assert parsed["coordinates"] == [[1, 2, 3, 4]]
    assert parsed["defect_type"] == "bubbling"
    assert parsed["reasoning"] == "surface blister"


def test_extract_defect_json_handles_fenced_flat_coordinates() -> None:
    raw = (
        "```json\n"
        '{"coordinates": [10, 20, 30, 40], '
        '"defect_type": "Bubbling", '
        '"reasoning": "Visible blister region"}'
        "\n```"
    )

    parsed = extract_defect_json(raw)
    assert parsed["coordinates"] == [[10, 20, 30, 40]]


def test_extract_defect_json_accepts_integral_float_coordinates() -> None:
    raw = (
        '{"coordinates": [[1.0, 2.0, 3.0, 4.0]], '
        '"defect_type": "bubbling", '
        '"reasoning": "surface blister"}'
    )

    parsed = extract_defect_json(raw)
    assert parsed["coordinates"] == [[1, 2, 3, 4]]
