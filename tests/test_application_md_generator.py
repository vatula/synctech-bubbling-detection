from __future__ import annotations

from pathlib import Path

from generate_application_md import (
    build_application_markdown,
    generate_application_markdown,
    iter_source_files,
)


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_iter_source_files_filters_and_sorts(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    _write_file(source_dir / "zeta.py", "z = 1\n")
    _write_file(source_dir / "alpha.py", "a = 1\n")
    _write_file(source_dir / "notes.txt", "ignore\n")
    _write_file(source_dir / "pkg" / "module.py", "m = 1\n")
    _write_file(source_dir / "__pycache__" / "cached.py", "c = 1\n")

    files = iter_source_files(source_dir)

    relative = [file.relative_to(source_dir).as_posix() for file in files]
    assert relative == ["alpha.py", "pkg/module.py", "zeta.py"]


def test_build_application_markdown_uses_headers_and_fences(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_file = source_dir / "module.py"
    _write_file(source_file, 'DOC = """contains ``` backticks"""\n')

    markdown = build_application_markdown(source_dir, [source_file])
    lines = markdown.splitlines()
    fence_line = next(line for line in lines if line.endswith("python"))

    assert "### src/module.py" in markdown
    assert fence_line == "````python"
    assert markdown.endswith("\n")


def test_generate_application_markdown_writes_expected_output(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    _write_file(source_dir / "b.py", "b = 2\n")
    _write_file(source_dir / "a.py", "a = 1\n")
    output_path = tmp_path / "application.md"

    generate_application_markdown(source_dir, output_path)

    output = output_path.read_text(encoding="utf-8")
    assert output_path.exists()
    assert output.index("### src/a.py") < output.index("### src/b.py")
    assert output.count("```python") == 2
