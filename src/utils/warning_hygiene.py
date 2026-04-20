from __future__ import annotations

import re
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TextIO

import structlog


@dataclass(frozen=True)
class WarningFilterRule:
    key: str
    message_regex: str
    category: type[Warning]
    source: str | None = None


_LOG = structlog.get_logger("warning_hygiene")
_is_configured = False
_REPORTED_RULES: set[str] = set()
_ORIGINAL_SHOWWARNING = warnings.showwarning
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr

WARNING_FILTER_RULES: Final[tuple[WarningFilterRule, ...]] = (
    WarningFilterRule(
        key="xformers_fallback",
        message_regex=r"xFormers is not available .*",
        category=UserWarning,
    ),
    WarningFilterRule(
        key="lightning_pytree_deprecation",
        message_regex=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        category=UserWarning,
        source=r"lightning/pytorch/utilities/_pytree\.py",
    ),
    WarningFilterRule(
        key="predict_dataloader_workers_advisory",
        message_regex=r"The 'predict_dataloader' does not have many workers.*",
        category=UserWarning,
        source=r"lightning/pytorch/trainer/connectors/data_connector\.py",
    ),
    WarningFilterRule(
        key="lightning_ambiguous_batch_size",
        message_regex=(
            r"Trying to infer the `batch_size` from an ambiguous collection\..*"
        ),
        category=UserWarning,
        source=r"lightning/pytorch/utilities/data\.py",
    ),
    WarningFilterRule(
        key="multiprocessing_fork_deprecation",
        message_regex=(
            r"This process .* is multi-threaded, use of fork\(\) may lead "
            r"to deadlocks in the child\."
        ),
        category=DeprecationWarning,
        source=r"multiprocessing/popen_fork\.py",
    ),
    WarningFilterRule(
        key="anomalib_image_resource_warning",
        message_regex=r"unclosed file <_io\.BufferedReader name='.*'>",
        category=ResourceWarning,
        source=r"anomalib/data/utils/image\.py",
    ),
    WarningFilterRule(
        key="anomalib_visualizer_resource_warning",
        message_regex=r"unclosed file <_io\.BufferedReader name='.*'>",
        category=ResourceWarning,
        source=r"anomalib/visualization/image/item_visualizer\.py",
    ),
)

_STREAM_NOISE_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("startup_null_file_hint", r"^\(null\): No such file or directory$"),
    (
        "openvino_optional_backend",
        r"OpenVINO is possibly not installed in the environment\..*",
    ),
    (
        "lightning_cloud_tip",
        (
            r"Tip: For seamless cloud logging and experiment tracking, "
            r"try installing \[litlogger\].*"
        ),
    ),
    (
        "ckpt_path_not_provided",
        r"ckpt_path is not provided\. Model weights will not be loaded\.",
    ),
    (
        "leafspec_deprecation_text",
        r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    ),
    (
        "lightning_ambiguous_batch_size_text",
        r"Trying to infer the `batch_size` from an ambiguous collection\..*",
    ),
    (
        "dinov2_mlp_ffn_fallback_text",
        r"^using MLP layer as FFN$",
    ),
)


class WarningHygieneStream:
    def __init__(self, wrapped: TextIO) -> None:
        self._wrapped = wrapped
        self._compiled_patterns = [
            (key, re.compile(pattern)) for key, pattern in _STREAM_NOISE_PATTERNS
        ]

    def write(self, message: str) -> int:
        normalized = message.strip()
        for key, pattern in self._compiled_patterns:
            if pattern.search(normalized):
                _report_allowlisted_pattern_once(key=key, message=normalized)
                return len(message)
        return int(self._wrapped.write(message))

    def flush(self) -> None:
        self._wrapped.flush()

    def __getattr__(self, item: str) -> object:
        return getattr(self._wrapped, item)


def _report_allowlisted_pattern_once(key: str, message: str) -> None:
    if key in _REPORTED_RULES:
        return
    _REPORTED_RULES.add(key)
    _LOG.info(
        "Allowlisted warning/noise intercepted",
        key=key,
        message_length=len(message),
    )


def _warning_matches_allowlist(
    message: str,
    category: type[Warning],
    filename: str,
) -> str | None:
    for rule in WARNING_FILTER_RULES:
        if not issubclass(category, rule.category):
            continue
        if not re.search(rule.message_regex, message):
            continue
        if rule.source is not None and not re.search(rule.source, filename):
            continue
        return rule.key
    return None


def _showwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    text = str(message)
    rule_key = _warning_matches_allowlist(
        message=text,
        category=category,
        filename=filename,
    )
    if rule_key is not None:
        _report_allowlisted_pattern_once(key=rule_key, message=text)
        return
    _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file=file, line=line)


def install_warning_hygiene() -> None:
    global _is_configured
    if _is_configured:
        return

    warnings.simplefilter("default")

    warnings.showwarning = _showwarning
    if not isinstance(sys.stdout, WarningHygieneStream):
        sys.stdout = WarningHygieneStream(sys.stdout)
    if not isinstance(sys.stderr, WarningHygieneStream):
        sys.stderr = WarningHygieneStream(sys.stderr)

    _is_configured = True
    _LOG.info(
        "Warning hygiene installed",
        warning_filter_count=len(WARNING_FILTER_RULES),
        stream_filter_count=len(_STREAM_NOISE_PATTERNS),
    )


def get_warning_filter_rules() -> Sequence[WarningFilterRule]:
    return WARNING_FILTER_RULES


def reset_warning_hygiene_for_tests() -> None:
    global _is_configured
    _is_configured = False
    _REPORTED_RULES.clear()
    warnings.showwarning = _ORIGINAL_SHOWWARNING
    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR
