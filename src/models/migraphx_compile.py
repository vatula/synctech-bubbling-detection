from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import onnx

from src.utils.logger import get_logger, setup_project

try:
    import migraphx as _migraphx_module  # type: ignore
except ImportError as error:
    _migraphx_module = None
    _migraphx_import_error: ImportError | None = error
else:
    _migraphx_import_error = None

migraphx: object | None = _migraphx_module

log = get_logger("migraphx_compile")


class CompileReport(TypedDict):
    onnx_path: str
    parsed_onnx_path: str
    compiled_path: str
    compile_target: str
    quantization: str
    iterations: int
    latency_us_mean: float
    latency_us_p50: float
    latency_us_p95: float


class CompileFailureReport(TypedDict):
    onnx_path: str
    parsed_onnx_path: str
    error: str


_MIGRAPHX_SUPPORTED_RESIZE_MODES = frozenset({"nearest", "linear"})
_MIGRAPHX_RESIZE_FALLBACK_MODE = "linear"


def _migraphx_search_paths() -> list[Path]:
    raw_candidates: list[str] = []
    configured_path = os.environ.get("MIGRAPHX_PYTHON_PATH")
    if configured_path:
        raw_candidates.extend(
            [entry for entry in configured_path.split(os.pathsep) if entry]
        )

    raw_candidates.extend(
        [
            "/opt/rocm/lib",
            "/opt/rocm/lib64",
        ]
    )
    raw_candidates.extend(str(path) for path in sorted(Path("/opt").glob("rocm-*/lib")))

    deduped: list[Path] = []
    seen: set[str] = set()
    for raw_path in raw_candidates:
        normalized = str(Path(raw_path).expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(Path(normalized))

    return deduped


def _try_import_migraphx_from_paths() -> object | None:
    global migraphx, _migraphx_import_error

    if migraphx is not None:
        return migraphx

    for candidate in _migraphx_search_paths():
        if not candidate.is_dir():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    try:
        migraphx = importlib.import_module("migraphx")
    except ImportError as error:
        _migraphx_import_error = error
        return None

    _migraphx_import_error = None
    return migraphx


def _require_migraphx() -> object:
    module = migraphx or _try_import_migraphx_from_paths()
    if module is None:
        import_error = str(_migraphx_import_error) if _migraphx_import_error else "N/A"
        msg = (
            "migraphx Python module is not available in the current environment. "
            "Install MIGraphX runtime bindings and ensure Python can resolve them "
            "(for ROCm images usually `/opt/rocm/lib`). Optionally set "
            "`MIGRAPHX_PYTHON_PATH` to the bindings directory. "
            f"Original import error: {import_error}"
        )
        raise RuntimeError(msg)
    return module


def _rewrite_resize_modes_for_migraphx(
    onnx_path: Path,
    sanitized_onnx_path: Path,
) -> Path:
    model = onnx.load(str(onnx_path))
    rewrites: list[dict[str, str]] = []

    for node in model.graph.node:
        if node.op_type != "Resize":
            continue

        for attribute in node.attribute:
            if attribute.name != "mode":
                continue
            if attribute.type != onnx.AttributeProto.STRING:
                continue

            mode = attribute.s.decode("utf-8")
            if mode in _MIGRAPHX_SUPPORTED_RESIZE_MODES:
                continue

            attribute.s = _MIGRAPHX_RESIZE_FALLBACK_MODE.encode("utf-8")
            rewrites.append(
                {
                    "node": node.name or "<unnamed>",
                    "from": mode,
                    "to": _MIGRAPHX_RESIZE_FALLBACK_MODE,
                }
            )
            break

    if not rewrites:
        return onnx_path

    sanitized_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(sanitized_onnx_path))
    log.warning(
        "Rewrote unsupported ONNX Resize mode(s) for MIGraphX compatibility",
        source_onnx_path=str(onnx_path),
        sanitized_onnx_path=str(sanitized_onnx_path),
        rewrites=rewrites,
        supported_modes=sorted(_MIGRAPHX_SUPPORTED_RESIZE_MODES),
    )
    return sanitized_onnx_path


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


def _compile_program(module: object, program: object) -> str:
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
        return "gpu_fast_math"
    except TypeError:
        log.warning(
            "MIGraphX compile does not accept fast_math/offload_copy; "
            "retrying default GPU compile"
        )
    except (MemoryError, RuntimeError) as error:
        log.warning(
            "MIGraphX compile with fast_math/offload_copy failed; "
            "retrying default compile",
            error=str(error),
        )

    try:
        compile_program(target)
        return "gpu"
    except (MemoryError, RuntimeError) as retry_error:
        log.warning(
            "MIGraphX GPU compile failed; retrying reference target",
            gpu_compile_error=str(retry_error),
        )
        ref_target = get_target("ref")
        compile_program(ref_target)
        return "ref"


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
    failed_reports: list[CompileFailureReport] = []

    for onnx_path in artifacts:
        relative = onnx_path.relative_to(onnx_dir)
        compiled_path = Path((output_dir / relative).with_suffix(".mxr"))
        sanitized_onnx_path = output_dir / "_migraphx_compatible_onnx" / relative
        parse_onnx_path = onnx_path

        try:
            parse_onnx_path = _rewrite_resize_modes_for_migraphx(
                onnx_path=onnx_path,
                sanitized_onnx_path=sanitized_onnx_path,
            )

            log.info(
                "Compiling ONNX graph with MIGraphX",
                onnx_path=str(onnx_path),
                parse_onnx_path=str(parse_onnx_path),
            )
            program = _parse_onnx_graph(
                module=module,
                onnx_path=parse_onnx_path,
                default_dim_value=args.default_dim_value,
            )
            quantization = _quantize_fp16(module=module, program=program)
            compile_target = _compile_program(module=module, program=program)
            _save_program(module=module, program=program, output_path=compiled_path)

            mean_us, p50_us, p95_us = _profile_latency_us(
                module=module,
                program=program,
                warmup_iterations=args.warmup_iterations,
                measured_iterations=args.measured_iterations,
            )

            report: CompileReport = {
                "onnx_path": str(onnx_path),
                "parsed_onnx_path": str(parse_onnx_path),
                "compiled_path": str(compiled_path),
                "compile_target": compile_target,
                "quantization": quantization,
                "iterations": int(args.measured_iterations),
                "latency_us_mean": mean_us,
                "latency_us_p50": p50_us,
                "latency_us_p95": p95_us,
            }
            reports.append(report)
            log.info("MIGraphX compilation/profile complete", **report)
        except Exception as error:  # noqa: BLE001
            failure: CompileFailureReport = {
                "onnx_path": str(onnx_path),
                "parsed_onnx_path": str(parse_onnx_path),
                "error": str(error),
            }
            failed_reports.append(failure)
            log.warning(
                "MIGraphX compilation/profile failed for artifact; continuing",
                **failure,
            )

    if not reports:
        msg = "MIGraphX compilation failed for all ONNX artifacts"
        raise RuntimeError(msg)

    report_path = output_dir / "latency_profile.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "artifacts": reports,
                "failed_artifacts": failed_reports,
            },
            handle,
            indent=2,
        )

    log.info(
        "Persisted MIGraphX latency profile",
        report_path=str(report_path),
        artifact_count=len(reports),
        failed_artifact_count=len(failed_reports),
    )
    return reports


def main() -> None:
    setup_project()
    args = parse_args()
    compile_onnx_with_migraphx(args=args)


if __name__ == "__main__":
    main()
