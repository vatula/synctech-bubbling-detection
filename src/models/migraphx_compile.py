from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from src.utils.logger import get_logger, setup_project

try:
    import migraphx  # type: ignore
except ImportError:
    migraphx = None

log = get_logger("migraphx_compile")


class CompileReport(TypedDict):
    onnx_path: str
    compiled_path: str
    quantization: str
    iterations: int
    latency_us_mean: float
    latency_us_p50: float
    latency_us_p95: float


def _require_migraphx() -> object:
    if migraphx is None:
        msg = "migraphx Python module is not available in the current environment"
        raise RuntimeError(msg)
    return migraphx


def _parse_onnx_graph(
    module: object,
    onnx_path: Path,
    default_dim_value: int,
) -> object:
    parse_onnx = cast(
        Callable[..., object] | None,
        getattr(module, "parse_onnx", None),
    )
    if parse_onnx is None:
        msg = "migraphx.parse_onnx is unavailable"
        raise RuntimeError(msg)

    try:
        return parse_onnx(str(onnx_path), default_dim_value=default_dim_value)
    except TypeError:
        return parse_onnx(str(onnx_path))


def _quantize_fp16(module: object, program: object) -> str:
    quantize_fp16 = cast(
        Callable[[object], object] | None,
        getattr(module, "quantize_fp16", None),
    )
    if callable(quantize_fp16):
        quantize_fp16(program)
        return "fp16"

    maybe_method = cast(
        Callable[[], object] | None,
        getattr(program, "quantize_fp16", None),
    )
    if callable(maybe_method):
        maybe_method()
        return "fp16"

    return "none"


def _compile_program(module: object, program: object) -> None:
    get_target = cast(
        Callable[[str], object] | None,
        getattr(module, "get_target", None),
    )
    if not callable(get_target):
        msg = "migraphx.get_target is unavailable"
        raise RuntimeError(msg)

    compile_program = cast(
        Callable[..., object] | None,
        getattr(program, "compile", None),
    )
    if not callable(compile_program):
        msg = "Compiled MIGraphX program does not expose compile()"
        raise RuntimeError(msg)

    target = get_target("gpu")
    try:
        compile_program(target, offload_copy=True, fast_math=True)
    except TypeError:
        compile_program(target)


def _save_program(module: object, program: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_program = cast(
        Callable[[object, str], object] | None,
        getattr(module, "save", None),
    )
    if callable(save_program):
        save_program(program, str(output_path))
        return

    maybe_method = cast(
        Callable[[str], object] | None,
        getattr(program, "save", None),
    )
    if callable(maybe_method):
        maybe_method(str(output_path))
        return

    msg = "Unable to persist compiled program (save API not found)"
    raise RuntimeError(msg)


def _shape_to_lens(shape: object) -> tuple[int, ...]:
    lens_attr = cast(
        Callable[[], Iterable[int]] | None,
        getattr(shape, "lens", None),
    )
    if callable(lens_attr):
        return tuple(int(v) for v in lens_attr())

    dims = cast(Iterable[int], shape)
    return tuple(int(v) for v in dims)


def _build_random_argument(module: object, shape: object) -> object:
    generate_argument = cast(
        Callable[[object], object] | None,
        getattr(module, "generate_argument", None),
    )
    if callable(generate_argument):
        return generate_argument(shape)

    argument_ctor = cast(
        Callable[[np.ndarray], object] | None,
        getattr(module, "argument", None),
    )
    if callable(argument_ctor):
        lens = _shape_to_lens(shape)
        tensor = np.random.rand(*lens).astype(np.float32)
        return argument_ctor(tensor)

    msg = "Unable to build MIGraphX input argument"
    raise RuntimeError(msg)


def _profile_latency_us(
    module: object,
    program: object,
    warmup_iterations: int,
    measured_iterations: int,
) -> tuple[float, float, float]:
    get_shapes = cast(
        Callable[[], Mapping[str, object]] | None,
        getattr(program, "get_parameter_shapes", None),
    )
    if not callable(get_shapes):
        msg = "Compiled program does not expose get_parameter_shapes()"
        raise RuntimeError(msg)

    parameter_shapes = dict(get_shapes())
    if not parameter_shapes:
        msg = "No input parameter shapes found for compiled program"
        raise RuntimeError(msg)

    args = {
        name: _build_random_argument(module=module, shape=shape)
        for name, shape in parameter_shapes.items()
    }

    run_program = cast(
        Callable[[Mapping[str, object]], object] | None,
        getattr(program, "run", None),
    )
    if not callable(run_program):
        msg = "Compiled program does not expose run()"
        raise RuntimeError(msg)

    for _ in range(warmup_iterations):
        run_program(args)

    latencies_us: list[float] = []
    for _ in range(measured_iterations):
        started = time.perf_counter_ns()
        run_program(args)
        ended = time.perf_counter_ns()
        latencies_us.append((ended - started) / 1000.0)

    latency_array = np.array(latencies_us, dtype=np.float64)
    return (
        float(np.mean(latency_array)),
        float(np.percentile(latency_array, 50)),
        float(np.percentile(latency_array, 95)),
    )


def _discover_onnx_artifacts(onnx_dir: Path) -> list[Path]:
    artifacts = sorted(path for path in onnx_dir.rglob("*.onnx") if path.is_file())
    if not artifacts:
        msg = f"No ONNX artifacts found under {onnx_dir}"
        raise FileNotFoundError(msg)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile ONNX artifacts with AMD MIGraphX and profile microsecond latency"
        )
    )
    parser.add_argument(
        "--onnx_dir",
        type=Path,
        default=Path("results/phase6/onnx"),
        help="Directory containing ONNX artifacts from Task 6.1.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/phase6/migraphx"),
        help="Directory where compiled MIGraphX artifacts and report are saved.",
    )
    parser.add_argument(
        "--default_dim_value",
        type=int,
        default=1,
        help="Fallback dimension value used by MIGraphX for dynamic dimensions.",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Number of warmup inference iterations before profiling.",
    )
    parser.add_argument(
        "--measured_iterations",
        type=int,
        default=100,
        help="Number of measured inference iterations for latency profiling.",
    )
    return parser.parse_args()


def compile_onnx_with_migraphx(args: argparse.Namespace) -> list[CompileReport]:
    module = _require_migraphx()
    onnx_dir = args.onnx_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _discover_onnx_artifacts(onnx_dir=onnx_dir)
    reports: list[CompileReport] = []

    for onnx_path in artifacts:
        relative = onnx_path.relative_to(onnx_dir)
        compiled_path = Path((output_dir / relative).with_suffix(".mxr"))

        log.info("Compiling ONNX graph with MIGraphX", onnx_path=str(onnx_path))
        program = _parse_onnx_graph(
            module=module,
            onnx_path=onnx_path,
            default_dim_value=args.default_dim_value,
        )
        quantization = _quantize_fp16(module=module, program=program)
        _compile_program(module=module, program=program)
        _save_program(module=module, program=program, output_path=compiled_path)

        mean_us, p50_us, p95_us = _profile_latency_us(
            module=module,
            program=program,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
        )

        report: CompileReport = {
            "onnx_path": str(onnx_path),
            "compiled_path": str(compiled_path),
            "quantization": quantization,
            "iterations": int(args.measured_iterations),
            "latency_us_mean": mean_us,
            "latency_us_p50": p50_us,
            "latency_us_p95": p95_us,
        }
        reports.append(report)
        log.info("MIGraphX compilation/profile complete", **report)

    report_path = output_dir / "latency_profile.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump({"artifacts": reports}, handle, indent=2)

    log.info(
        "Persisted MIGraphX latency profile",
        report_path=str(report_path),
        artifact_count=len(reports),
    )
    return reports


def main() -> None:
    setup_project()
    args = parse_args()
    compile_onnx_with_migraphx(args=args)


if __name__ == "__main__":
    main()
