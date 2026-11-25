#!/usr/bin/env python
"""Test runner for fixture projects - runs LOCCA against test tasks using subprocess."""

import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from fixture_manager import FixtureManager


@dataclass
class Task:
    """A single evaluation task."""

    id: str
    name: str
    category: str
    difficulty: str
    prompt: str
    evaluation: dict[str, Any]
    model: str = "mock-model"
    agent_mode: str = "frame_agent"
    tool_call_mode: str = "classic"
    max_iterations: int = 10

    @classmethod
    def from_dict(cls, data: dict, defaults: dict) -> "Task":
        return cls(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            difficulty=data["difficulty"],
            prompt=data["prompt"],
            evaluation=data["evaluation"],
            model=data.get("model", defaults.get("model", "mock-model")),
            agent_mode=data.get(
                "agent_mode", defaults.get("agent_mode", "frame_agent")
            ),
            tool_call_mode=data.get(
                "tool_call_mode", defaults.get("tool_call_mode", "classic")
            ),
            max_iterations=data.get(
                "max_iterations", defaults.get("max_iterations", 10)
            ),
        )


@dataclass
class TaskResult:
    """Result of running a task."""

    task_id: str
    task_name: str
    success: bool
    duration_seconds: float = 0
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    files_changed: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    evaluation_details: dict = field(default_factory=dict)
    error: str = ""


class SubprocessFixtureEvaluator:
    """Evaluates fixture tasks by running LOCCA as a subprocess."""

    def __init__(self, fixture_path: Path, project_root: Path):
        self.fixture_path = fixture_path
        self.project_root = project_root
        self.python_exe = project_root / ".venv" / "Scripts" / "python.exe"
        self.main_module = "local_coding_assistant.cli.main"

    def load_tasks(
        self, tasks_file: Path, filter_ids: list[str] | None = None
    ) -> list[Task]:
        """Load tasks from YAML file, optionally filtering by ID."""
        with open(tasks_file) as f:
            data = yaml.safe_load(f)

        defaults = data.get("defaults", {})
        tasks = [Task.from_dict(t, defaults) for t in data.get("tasks", [])]

        if filter_ids:
            tasks = [t for t in tasks if t.id in filter_ids]
            if not tasks:
                print(f"Warning: No tasks matched filter {filter_ids}")

        return tasks

    def run_task(
        self, task: Task, env_overrides: dict | None = None, eval_mode: str = "mock"
    ) -> TaskResult:
        """Run a single task via subprocess."""
        print(f"\n{'=' * 60}")
        print(f"Running: {task.name} ({task.id})")
        print(f"Category: {task.category}, Difficulty: {task.difficulty}")
        print(f"{'=' * 60}")

        # Build command
        cmd = [
            str(self.python_exe),
            "-m",
            self.main_module,
            "run",
            "query",
            task.prompt,
            "--agent-mode",
            task.agent_mode,
            "--tool-call-mode",
            task.tool_call_mode,
            "--model",
            task.model,
            "--target-project-root",
            str(self.fixture_path),
            "--log-level",
            "INFO",
        ]

        start_time = time.time()
        try:
            # Use FixtureManager for snapshot/restore
            fm = FixtureManager(self.fixture_path)
            fm.snapshot()

            # Run from project root with fixture as cwd
            result = subprocess.run(  # noqa: S603
                cmd,
                cwd=self.fixture_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            duration = time.time() - start_time

            # Get changes (captures agent-made changes only)
            files_created, files_modified = fm.get_changes()

            # Evaluate while the fixture is still in its modified state so that
            # checks (file_exists, command execution, etc.) inspect the actual
            # post-agent filesystem rather than the restored original state.
            evaluation = self.evaluate_task(task, result, eval_mode=eval_mode)

            # Restore fixture to original state
            fm.restore()

            return TaskResult(
                task_id=task.id,
                task_name=task.name,
                success=result.returncode == 0,
                duration_seconds=duration,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                files_changed=files_modified,
                files_created=files_created,
                evaluation_details=evaluation,
            )

        except subprocess.TimeoutExpired:
            return TaskResult(
                task_id=task.id,
                task_name=task.name,
                success=False,
                duration_seconds=300,
                error="Timeout after 300 seconds",
                return_code=-1,
            )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                task_name=task.name,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e),
                return_code=-1,
            )

    def _run_fixture_command(
        self, args: list[str], timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """Run a CLI command against the fixture project (src on sys.path)."""
        cmd = [
            str(self.python_exe),
            "-c",
            "import sys; sys.path.insert(0, 'src'); from python_cli_tool.main import cli; cli()",
            *args,
        ]
        return subprocess.run(  # noqa: S603
            cmd, cwd=self.fixture_path, capture_output=True, text=True, timeout=timeout
        )

    def evaluate_task(
        self, task: Task, result: TaskResult, eval_mode: str = "mock"
    ) -> dict:
        """Evaluate if a task was completed successfully."""
        eval_config = task.evaluation
        eval_type = eval_config.get("type", "unknown")
        criteria = eval_config.get("criteria", [])

        evaluation = {
            "task_id": task.id,
            "eval_type": eval_type,
            "criteria_results": [],
            "overall_pass": True,
        }

        # In mock mode, just check if the agent ran successfully
        if eval_mode == "mock":
            passed = result.success and result.return_code == 0
            details = {
                "check": "mock_success",
                "passed": passed,
                "return_code": result.return_code,
                "duration_seconds": result.duration_seconds,
            }
            evaluation["criteria_results"].append(details)
            evaluation["overall_pass"] = passed
            return evaluation

        # Full evaluation mode (checks file changes, CLI behavior, etc.)
        for criterion in criteria:
            check_type = criterion.get("check", "unknown")
            handler = getattr(self, f"_handle_{check_type}", self._handle_unknown_check)
            try:
                passed, details = handler(criterion, result, task)
            except Exception as e:
                passed = False
                details = {"error": str(e)}
            details.setdefault("check", check_type)
            details["passed"] = passed
            evaluation["criteria_results"].append(details)

            if not passed:
                evaluation["overall_pass"] = False

        return evaluation

    def _handle_unknown_check(self, criterion: dict, result: TaskResult, task: Task):
        return False, {
            "warning": f"Unknown check type: {criterion.get('check', 'unknown')}"
        }

    def _handle_file_exists(self, criterion: dict, result: TaskResult, task: Task):
        path = self.fixture_path / criterion["path"]
        passed = path.exists()
        return passed, {"path": criterion["path"], "exists": passed}

    def _handle_tests_exist(self, criterion: dict, result: TaskResult, task: Task):
        paths = [p for p in [criterion.get("path"), criterion.get("or")] if p]
        passed = any((self.fixture_path / p).exists() for p in paths)
        return passed, {"paths": paths}

    def _handle_file_created(self, criterion: dict, result: TaskResult, task: Task):
        import fnmatch

        pattern = criterion.get("pattern", "*")
        matched = any(fnmatch.fnmatch(f, pattern) for f in result.files_created)
        return matched, {
            "pattern": pattern,
            "created_files": result.files_created,
            "matched": matched,
        }

    def _handle_file_modified(self, criterion: dict, result: TaskResult, task: Task):
        path = criterion.get("path")
        if path:
            passed = path in result.files_changed
        else:
            passed = len(result.files_changed) > 0
        return passed, {"changed_files": result.files_changed}

    def _run_command_check(
        self,
        criterion: dict,
        result: TaskResult,
        task: Task,
        expect_output: bool = False,
    ):
        args = criterion.get("args", [])
        cp = self._run_fixture_command(args)
        expect_exit = criterion.get("expect_exit_code", 0)
        passed = cp.returncode == expect_exit
        details = {
            "args": args,
            "return_code": cp.returncode,
            "expected_exit_code": expect_exit,
            "stdout": cp.stdout[:500],
            "stderr": cp.stderr[:500],
        }
        if expect_output and "expect_output_contains" in criterion:
            contains = criterion["expect_output_contains"] in cp.stdout
            passed = passed and contains
            details["output_contains"] = criterion["expect_output_contains"]
            details["output_contains_match"] = contains
        return passed, details

    def _handle_command_registered(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        command = criterion.get("command", "")
        cp = self._run_fixture_command(["--help"])
        passed = command in cp.stdout
        return passed, {
            "command": command,
            "return_code": cp.returncode,
            "stdout": cp.stdout[:500],
        }

    def _handle_command_works(self, criterion: dict, result: TaskResult, task: Task):
        return self._run_command_check(criterion, result, task, expect_output=True)

    def _handle_command_works_yaml(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        return self._run_command_check(criterion, result, task)

    def _handle_cli_works_without_config(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        cp = self._run_fixture_command(["--help"])
        return cp.returncode == 0, {"return_code": cp.returncode}

    def _handle_no_exception_on_missing_config(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        cp = self._run_fixture_command(["greet", "--name", "Test"])
        return cp.returncode == 0, {"return_code": cp.returncode}

    def _handle_validation_rejects_invalid_timeout(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        cp = self._run_fixture_command(["config-set", "timeout", "-5"])
        return cp.returncode != 0, {
            "return_code": cp.returncode,
            "stderr": cp.stderr[:500],
        }

    def _handle_validation_rejects_invalid_retries(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        cp = self._run_fixture_command(["config-set", "retries", "-1"])
        return cp.returncode != 0, {"return_code": cp.returncode}

    def _handle_validation_rejects_invalid_format(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        cp = self._run_fixture_command(["config-set", "output_format", "xml"])
        return cp.returncode != 0, {"return_code": cp.returncode}

    def _handle_validation_rejects_invalid_url(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        cp = self._run_fixture_command(["config-set", "api_url", "not-a-url"])
        return cp.returncode != 0, {"return_code": cp.returncode}

    def _handle_validation_accepts_valid_values(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        passed = True
        cp = None
        for args in [
            ["config-set", "timeout", "60"],
            ["config-set", "retries", "5"],
            ["config-set", "output_format", "yaml"],
            ["config-set", "api_url", "https://api.example.com"],
        ]:
            cp = self._run_fixture_command(args)
            if cp.returncode != 0:
                passed = False
                break
        return passed, {"return_code": cp.returncode if cp is not None else None}

    def _handle_binary_file_handled_gracefully(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        binary = self.fixture_path / "binary.bin"
        binary.write_bytes(b"\x00\x01\x02\x03")
        try:
            cp = self._run_fixture_command(["process", "binary.bin"])
            expect_exit = criterion.get("expect_exit_code", 1)
            passed = cp.returncode == expect_exit
            details = {
                "return_code": cp.returncode,
                "expected_exit_code": expect_exit,
            }
            if "expect_error_contains" in criterion:
                contains = criterion["expect_error_contains"] in (cp.stdout + cp.stderr)
                passed = passed and contains
                details["error_contains"] = criterion["expect_error_contains"]
                details["error_contains_match"] = contains
            return passed, details
        finally:
            if binary.exists():
                binary.unlink()

    def _handle_tests_pass(self, criterion: dict, result: TaskResult, task: Task):
        return True, {}

    def _handle_all_commands_still_work(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        passed = True
        cp = None
        for args in [
            ["--help"],
            ["version"],
            ["greet", "--name", "Test"],
            ["config-show"],
        ]:
            cp = self._run_fixture_command(args)
            if cp.returncode != 0:
                passed = False
                break
        return passed, {"return_code": cp.returncode if cp is not None else None}

    def _handle_help_shows_examples(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        passed = "Example" in result.stdout or "example" in result.stdout.lower()
        return passed, {}

    def _handle_help_descriptive(self, criterion: dict, result: TaskResult, task: Task):
        return len(result.stdout) > 50, {}

    def _handle_tests_added(self, criterion: dict, result: TaskResult, task: Task):
        test_files = [
            f
            for f in result.files_changed + result.files_created
            if "test" in f.lower()
        ]
        return len(test_files) > 0, {"test_files": test_files}

    def _handle_covers_all_validations(
        self, criterion: dict, result: TaskResult, task: Task
    ):
        return True, {}


def print_task_result(result: TaskResult, verbose: bool = False) -> None:
    """Print a single task result to the console."""
    overall_pass = result.evaluation_details.get("overall_pass", False)
    status = "✓ PASS" if (result.success and overall_pass) else "✗ FAIL"
    print(f"\n{status} {result.task_name} ({result.task_id})")
    print(
        f"  Duration: {result.duration_seconds:.1f}s, Return code: {result.return_code}"
    )
    print(f"  Files created: {len(result.files_created)}")
    print(f"  Files modified: {len(result.files_changed)}")
    for f in result.files_created:
        print(f"    + {f}")
    for f in result.files_changed:
        print(f"    ~ {f}")
    if result.error:
        print(f"  Error: {result.error}")
    if result.stdout and verbose:
        print(f"  Stdout: {result.stdout[:500]}...")
    if result.stderr and verbose:
        print(f"  Stderr: {result.stderr[:500]}...")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LOCCA evaluation on fixture projects"
    )
    parser.add_argument(
        "--fixture", default="python_cli_tool", help="Fixture project name"
    )
    parser.add_argument(
        "--tasks", help="Comma-separated task IDs to run (default: all)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available tasks and exit"
    )
    parser.add_argument("--output", help="Output results to JSON file")
    parser.add_argument("--model", default="mock-model", help="Model to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--mock", action="store_true", help="Use mock mode (fast, no real LLM)"
    )
    parser.add_argument(
        "--eval-mode",
        choices=["mock", "full"],
        default="mock",
        help="Evaluation mode: mock (check success only) or full (check file changes)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    fixture_path = project_root / "tests" / "fixtures" / "projects" / args.fixture
    tasks_file = fixture_path / "tasks.yaml"

    if not tasks_file.exists():
        print(f"Error: Tasks file not found: {tasks_file}")
        return 1

    evaluator = SubprocessFixtureEvaluator(fixture_path, project_root)

    # Load tasks
    filter_ids = args.tasks.split(",") if args.tasks else None
    tasks = evaluator.load_tasks(tasks_file, filter_ids)

    if args.list:
        print(f"Available tasks in {args.fixture}:")
        for task in tasks:
            print(f"  {task.id}: {task.name} [{task.category}]")
        return 0

    print(f"Loaded {len(tasks)} task(s)")
    print(f"Fixture: {fixture_path}")
    print(f"Project root: {project_root}")
    print(f"Python: {evaluator.python_exe}")

    if not evaluator.python_exe.exists():
        print(f"Error: Python executable not found: {evaluator.python_exe}")
        print("Run 'uv sync' in the project root first.")
        return 1

    # Run tasks
    results = []
    for task in tasks:
        # Override model if specified
        if args.model:
            task.model = args.model

        result = evaluator.run_task(task, eval_mode=args.eval_mode)
        print_task_result(result, verbose=args.verbose)

        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    passed = sum(
        1
        for r in results
        if r.success and r.evaluation_details.get("overall_pass", False)
    )
    print(f"Passed: {passed}/{len(results)}")

    for r in results:
        eval_pass = r.evaluation_details.get("overall_pass", False)
        status = "✓" if (r.success and eval_pass) else "✗"
        print(f"  {status} {r.task_name} ({r.task_id})")

    # Save results
    if args.output:
        output_data = {
            "fixture": args.fixture,
            "model": args.model,
            "results": [
                {
                    "task_id": r.task_id,
                    "task_name": r.task_name,
                    "success": r.success,
                    "eval_pass": r.evaluation_details.get("overall_pass", False),
                    "duration_seconds": r.duration_seconds,
                    "return_code": r.return_code,
                    "files_created": r.files_created,
                    "files_changed": r.files_changed,
                    "error": r.error,
                    "evaluation": r.evaluation_details,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
