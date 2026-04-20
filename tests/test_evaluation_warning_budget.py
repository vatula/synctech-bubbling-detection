from __future__ import annotations

import re
import sys
import warnings

import pytest

from src.utils.warning_hygiene import (
    install_warning_hygiene,
    reset_warning_hygiene_for_tests,
)

_KNOWN_WARNING_PATTERNS = (
    r"xFormers is not available",
    r"`isinstance\(treespec, LeafSpec\)` is deprecated",
    r"Trying to infer the `batch_size` from an ambiguous collection",
    r"OpenVINO is possibly not installed in the environment",
    r"^using MLP layer as FFN$",
    r"Tip: For seamless cloud logging and experiment tracking",
    r"The 'predict_dataloader' does not have many workers",
    r"ckpt_path is not provided\. Model weights will not be loaded\.",
    r"^\(null\): No such file or directory$",
)


def test_warning_hygiene_enforces_known_warning_budget(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reset_warning_hygiene_for_tests()
    install_warning_hygiene()

    warnings.warn("xFormers is not available (SwiGLU)", UserWarning, stacklevel=2)
    sys.stderr.write(
        "`isinstance(treespec, LeafSpec)` is deprecated, use "
        "`isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.\n"
    )
    sys.stdout.write(
        "Tip: For seamless cloud logging and experiment tracking, "
        "try installing [litlogger] to enable LitLogger.\n"
    )
    sys.stdout.write("OpenVINO is possibly not installed in the environment.\n")
    sys.stdout.write("using MLP layer as FFN\n")
    sys.stderr.write("(null): No such file or directory\n")
    sys.stderr.write("ckpt_path is not provided. Model weights will not be loaded.\n")
    sys.stderr.write(
        "Trying to infer the `batch_size` from an ambiguous collection. "
        "The batch size we found is 3.\n"
    )

    captured = capsys.readouterr()
    terminal_output = f"{captured.out}\n{captured.err}"
    for pattern in _KNOWN_WARNING_PATTERNS:
        assert re.search(pattern, terminal_output, flags=re.MULTILINE) is None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn(
            "unexpected warning should remain visible",
            UserWarning,
            stacklevel=2,
        )
    assert any(
        str(warning.message) == "unexpected warning should remain visible"
        for warning in caught
    )

    reset_warning_hygiene_for_tests()
