import argparse
import re
import sys
from pathlib import Path


def _normalize_evidence_line(line: str) -> str:
    normalized = line.strip()
    normalized = re.sub(r"^[-*]\s*", "", normalized)
    return normalized.strip().lower()


def validate_checklist(file_path: Path) -> None:
    content = file_path.read_text()

    # Pattern to find tasks: ### Bx-XXX: ...\nStatus: [...]
    # We want to capture the status and the task ID
    task_pattern = re.compile(r"### ([A-Z0-9]+-\d+): .*\nStatus: \[(.)\]")
    tasks = task_pattern.findall(content)

    # Split content by task to validate evidence
    task_sections = re.split(r"### ", content)[1:]

    if not tasks:
        print("No tasks found in the checklist.")
        sys.exit(1)

    # Statuses
    # [ ] - not started
    # [~] - in progress
    # [x] - done
    # [!] - blocked

    in_progress = 0
    tasks_info = []

    for section in task_sections:
        match = re.search(r"([A-Z0-9]+-\d+): .*\nStatus: \[(.)\]", section)
        if not match:
            continue
        task_id, status = match.groups()
        tasks_info.append({"id": task_id, "status": status})

        if status == "~":
            in_progress += 1

        if status == "x":
            if "Evidence:" not in section:
                print(f"Error: Task {task_id} is [x] but missing 'Evidence:' section.")
                sys.exit(1)

            evidence_match = re.search(r"Evidence:\n(.*)", section, re.DOTALL)
            if not evidence_match:
                print(f"Error: Task {task_id} is [x] but missing 'Evidence:' section.")
                sys.exit(1)

            evidence_section = evidence_match.group(1)
            if "---" in evidence_section:
                evidence_section = evidence_section.split("---")[0]

            evidence_lines = [
                _normalize_evidence_line(line)
                for line in evidence_section.splitlines()
                if _normalize_evidence_line(line)
            ]
            if not evidence_lines:
                print(f"Error: Task {task_id} has empty evidence.")
                sys.exit(1)
            if any(line in {"pending", "pending."} for line in evidence_lines):
                print(f"Error: Task {task_id} has 'Pending.' evidence.")
                sys.exit(1)

            evidence_text = "\n".join(evidence_lines)
            speculative_tokens = ["assumed", "maybe", "not sure", "haven't"]
            for token in speculative_tokens:
                if token in evidence_text:
                    print(f"Error: Task {task_id} has speculative evidence: '{token}'.")
                    sys.exit(1)

    # Rules
    # 1. Any later task is [x] while earlier task is not [x].
    # 2. More than one task is [~].
    # 3. Unknown status symbol exists.

    if in_progress > 1:
        print(f"Error: More than one task is in progress: {in_progress}")
        sys.exit(1)

    first_not_done = False
    for task in tasks_info:
        if task["status"] == "x":
            if first_not_done:
                print(
                    f"Error: Task {task['id']} is done, but earlier task was not done."
                )
                sys.exit(1)
        elif task["status"] in [" ", "~", "!"]:
            first_not_done = True
        else:
            print(
                f"Error: Unknown status symbol '{task['status']}' in task {task['id']}"
            )
            sys.exit(1)

    print("Checklist valid.")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()
    validate_checklist(args.file)
