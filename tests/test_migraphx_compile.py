from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
import json
import sys
from pathlib import Path

import onnx
import pytest

from src.models import migraphx_compile

_REQUIRE_MIGRAPHX = migraphx_compile._require_migraphx


def _build_resize_model(path: Path, mode: str) -> None:
    input_tensor = onnx.helper.make_tensor_value_info(
        "x",
        onnx.TensorProto.FLOAT,
        [1, 1, 2, 2],
    )
    output_tensor = onnx.helper.make_tensor_value_info(
        "y",
        onnx.TensorProto.FLOAT,
        [1, 1, 4, 4],
    )
    roi_tensor = onnx.helper.make_tensor(
        "roi",
        onnx.TensorProto.FLOAT,
        [0],
        [],
    )
    scales_tensor = onnx.helper.make_tensor(
        "scales",
        onnx.TensorProto.FLOAT,
        [4],
        [1.0, 1.0, 2.0, 2.0],
    )
    resize_node = onnx.helper.make_node(
        "Resize",
        inputs=["x", "roi", "scales"],
        outputs=["y"],
        name="resize_node",
        mode=mode,
        coordinate_transformation_mode="half_pixel",
        nearest_mode="floor",
    )
    graph = onnx.helper.make_graph(
        nodes=[resize_node],
        name="resize_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[roi_tensor, scales_tensor],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    onnx.save(model, str(path))


def _read_resize_mode(path: Path) -> str:
    model = onnx.load(str(path))
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        for attribute in node.attribute:
            if attribute.name == "mode":
                return attribute.s.decode("utf-8")

    msg = "Resize mode attribute not found"
    raise AssertionError(msg)


def test_require_migraphx_loads_from_configured_search_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stub_path = tmp_path / "migraphx.py"
    stub_path.write_text("MARKER = 'stub'\n", encoding="utf-8")

    def _search_paths() -> list[Path]:
        return [tmp_path]

    monkeypatch.setattr(migraphx_compile, "migraphx", None)
    monkeypatch.setattr(
        migraphx_compile,
        "_migraphx_import_error",
        ImportError("No module named 'migraphx'"),
    )
    monkeypatch.setattr(migraphx_compile, "_migraphx_search_paths", _search_paths)
    sys.modules.pop("migraphx", None)

    module = _REQUIRE_MIGRAPHX()

    assert getattr(module, "MARKER", None) == "stub"
    sys.modules.pop("migraphx", None)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


def test_require_migraphx_raises_actionable_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _search_paths() -> list[Path]:
        return []

    monkeypatch.setattr(migraphx_compile, "migraphx", None)
    monkeypatch.setattr(
        migraphx_compile,
        "_migraphx_import_error",
        ImportError("No module named 'migraphx'"),
    )
    monkeypatch.setattr(migraphx_compile, "_migraphx_search_paths", _search_paths)
    sys.modules.pop("migraphx", None)

    with pytest.raises(RuntimeError, match="MIGRAPHX_PYTHON_PATH"):
        _REQUIRE_MIGRAPHX()


def test_rewrite_resize_modes_for_migraphx_rewrites_unsupported_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.onnx"
    sanitized = tmp_path / "sanitized.onnx"
    _build_resize_model(path=source, mode="cubic")

    rewritten_path = migraphx_compile._rewrite_resize_modes_for_migraphx(
        onnx_path=source,
        sanitized_onnx_path=sanitized,
    )

    assert rewritten_path == sanitized
    assert _read_resize_mode(path=source) == "cubic"
    assert _read_resize_mode(path=rewritten_path) == "linear"


def test_rewrite_resize_modes_for_migraphx_keeps_supported_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.onnx"
    sanitized = tmp_path / "sanitized.onnx"
    _build_resize_model(path=source, mode="linear")

    rewritten_path = migraphx_compile._rewrite_resize_modes_for_migraphx(
        onnx_path=source,
        sanitized_onnx_path=sanitized,
    )

    assert rewritten_path == source
    assert _read_resize_mode(path=rewritten_path) == "linear"
    assert not sanitized.exists()


def test_compile_program_retries_without_fast_math_on_runtime_error() -> None:
    class _Module:
        @staticmethod
        def get_target(name: str) -> str:
            return name

    class _Program:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def compile(self, target: str, **kwargs: object) -> None:
            self.calls.append((target, bool(kwargs)))
            if kwargs:
                raise RuntimeError("tuning failed")

    program = _Program()

    compile_target = migraphx_compile._compile_program(
        module=_Module(),
        program=program,
    )

    assert compile_target == "gpu"
    assert program.calls == [("gpu", True), ("gpu", False)]


def test_compile_program_falls_back_to_ref_target_on_gpu_failure() -> None:
    class _Module:
        @staticmethod
        def get_target(name: str) -> str:
            return name

    class _Program:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def compile(self, target: str, **kwargs: object) -> None:
            self.calls.append((target, bool(kwargs)))
            if target == "gpu":
                raise RuntimeError("gpu compilation failed")

    program = _Program()

    compile_target = migraphx_compile._compile_program(
        module=_Module(),
        program=program,
    )

    assert compile_target == "ref"
    assert program.calls == [("gpu", True), ("gpu", False), ("ref", False)]


def test_compile_program_falls_back_to_ref_target_on_memory_error() -> None:
    class _Module:
        @staticmethod
        def get_target(name: str) -> str:
            return name

    class _Program:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def compile(self, target: str, **kwargs: object) -> None:
            self.calls.append((target, bool(kwargs)))
            if target == "gpu":
                raise MemoryError("std::bad_alloc")

    program = _Program()

    compile_target = migraphx_compile._compile_program(
        module=_Module(),
        program=program,
    )

    assert compile_target == "ref"
    assert program.calls == [("gpu", True), ("gpu", False), ("ref", False)]


def test_compile_onnx_with_migraphx_continues_after_single_artifact_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    onnx_dir = tmp_path / "onnx"
    output_dir = tmp_path / "mx"
    failing_onnx = onnx_dir / "a_fail.onnx"
    passing_onnx = onnx_dir / "b_ok.onnx"
    failing_onnx.parent.mkdir(parents=True, exist_ok=True)
    failing_onnx.write_text("stub", encoding="utf-8")
    passing_onnx.write_text("stub", encoding="utf-8")

    class _Program:
        def __init__(self, path: Path) -> None:
            self.path = path

    monkeypatch.setattr(migraphx_compile, "_require_migraphx", lambda: object())

    def _rewrite(onnx_path: Path, sanitized_onnx_path: Path) -> Path:
        _ = sanitized_onnx_path
        return onnx_path

    monkeypatch.setattr(
        migraphx_compile,
        "_rewrite_resize_modes_for_migraphx",
        _rewrite,
    )

    def _parse(module: object, onnx_path: Path, default_dim_value: int) -> object:
        _ = (module, default_dim_value)
        if onnx_path.name.startswith("a_fail"):
            raise RuntimeError("parse failed")
        return _Program(path=onnx_path)

    monkeypatch.setattr(migraphx_compile, "_parse_onnx_graph", _parse)

    def _quantize(module: object, program: object) -> str:
        _ = (module, program)
        return "fp16"

    def _compile(module: object, program: object) -> str:
        _ = (module, program)
        return "gpu"

    monkeypatch.setattr(migraphx_compile, "_quantize_fp16", _quantize)
    monkeypatch.setattr(migraphx_compile, "_compile_program", _compile)

    def _save(module: object, program: object, output_path: Path) -> None:
        _ = module
        assert isinstance(program, _Program)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("compiled", encoding="utf-8")

    monkeypatch.setattr(migraphx_compile, "_save_program", _save)

    def _profile(
        module: object,
        program: object,
        warmup_iterations: int,
        measured_iterations: int,
    ) -> tuple[float, float, float]:
        _ = (module, program, warmup_iterations, measured_iterations)
        return (100.0, 90.0, 110.0)

    monkeypatch.setattr(migraphx_compile, "_profile_latency_us", _profile)

    args = argparse.Namespace(
        onnx_dir=onnx_dir,
        output_dir=output_dir,
        default_dim_value=1,
        warmup_iterations=1,
        measured_iterations=2,
    )

    reports = migraphx_compile.compile_onnx_with_migraphx(args=args)

    assert len(reports) == 1
    assert reports[0]["onnx_path"].endswith("b_ok.onnx")

    profile_path = output_dir / "latency_profile.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert len(payload["artifacts"]) == 1
    assert len(payload["failed_artifacts"]) == 1
