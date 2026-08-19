"""Run the developer's own commands and report what they said."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .commands import failure_line, is_command

TaskGrade = Literal["verified", "refuted", "ungraded"]
Outcome = Literal["pass", "fail", "unavailable", "unrunnable"]

GLOBAL_ORDER: tuple[str, ...] = ("build", "lint", "test")


@dataclass(frozen=True, slots=True)
class CommandResult:
    """What one command did."""

    exit_code: int = 0
    output: str = ""
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@runtime_checkable
class CommandRunner(Protocol):
    """Runs a shell command. Supplied only when the host actually has a shell."""

    def run(self, command: str, *, timeout_s: int = 0) -> CommandResult:
        """Run ``command`` and return what it said."""


@dataclass(frozen=True, slots=True)
class Step:
    """One command's contribution to the verdict."""

    name: str
    command: str = ""
    outcome: Outcome = "unavailable"
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.outcome == "fail"


@dataclass(frozen=True, slots=True)
class Verdict:
    """The grade, and every step that produced it."""

    grade: TaskGrade = "ungraded"
    steps: tuple[Step, ...] = ()
    reason: str = ""

    @property
    def verified(self) -> bool:
        return self.grade == "verified"

    def summary(self) -> str:
        parts = [f"{step.name}={step.outcome}" for step in self.steps]
        return f"{self.grade}" + (f" ({', '.join(parts)})" if parts else "")


def grade(
    *,
    own_check: str = "",
    commands: Mapping[str, str] | None = None,
    runner: CommandRunner | None = None,
    timeout_s: int = 0,
) -> Verdict:
    """Grade one task. Never consults a model, and never grades a claimed failure."""
    if runner is None:
        return Verdict(reason="no command runner — nothing was executed")

    configured = dict(commands or {})
    steps: list[Step] = []

    own = _run_step("check", own_check, runner, timeout_s)
    steps.append(own)
    if own.failed:
        return Verdict(grade="refuted", steps=tuple(steps), reason=own.detail)

    for name in GLOBAL_ORDER:
        step = _run_step(name, configured.get(name, ""), runner, timeout_s)
        steps.append(step)
        if step.failed:
            return Verdict(grade="refuted", steps=tuple(steps), reason=step.detail)

    return Verdict(grade=_verdict(steps), steps=tuple(steps), reason=_reason(steps))


def _run_step(name: str, command: str, runner: CommandRunner, timeout_s: int) -> Step:
    """Gate the string, run it, read the failure. In that order."""
    if not command.strip():
        return Step(name=name, outcome="unavailable")

    check = is_command(command)
    if not check.runnable:
        return Step(name=name, command=command, outcome="unrunnable", detail=check.reason)

    result = runner.run(command, timeout_s=timeout_s)
    if result.passed:
        return Step(name=name, command=command, outcome="pass")

    detail = "timed out" if result.timed_out else failure_line(result.output)
    return Step(name=name, command=command, outcome="fail", detail=detail or "exited non-zero")


def _verdict(steps: Sequence[Step]) -> TaskGrade:
    """No failures. Verified if something ran and the own check was not rejected."""
    own = next((step for step in steps if step.name == "check"), None)
    if own is not None and own.outcome == "unrunnable":
        return "ungraded"
    return "verified" if any(step.outcome == "pass" for step in steps) else "ungraded"


def _reason(steps: Sequence[Step]) -> str:
    rejected = [step.name for step in steps if step.outcome == "unrunnable"]
    if any(name == "check" for name in rejected):
        return "the task's own check was rejected as prose — nothing tested the claim"

    ran = [step.name for step in steps if step.outcome == "pass"]
    if not ran:
        return "nothing was configured to run — this workspace has not opted in"
    return f"passed: {', '.join(ran)}"


@dataclass(frozen=True, slots=True)
class GradeRequest:
    """What a caller must decide before grading is worth doing."""

    claimed_success: bool
    own_check: str = ""
    commands: Mapping[str, str] = field(default_factory=dict)

    @property
    def worth_running(self) -> bool:
        return self.claimed_success
