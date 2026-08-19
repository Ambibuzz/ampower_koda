"""The nudges, verbatim, and the state that decides when each one fires."""

from __future__ import annotations

from dataclasses import dataclass

BUDGET = (
    "The token budget for this turn is exhausted. Do not call another tool. "
    "Answer now from the evidence already gathered, and state any uncertainty."
)

ROUNDS = (
    "This is the last round of this turn. Do not call another tool — there is no "
    "round left to receive the result. Answer now from what you have already read, "
    "name what you did not get to, and do not pretend the gap is not there."
)

DRY = (
    "The last few rounds of tools returned nothing you had not already seen. Do not "
    "call another tool. Answer now from the evidence already gathered, and say "
    "plainly what you could not find."
)

CONTINUE = (
    "Your previous message stopped because it reached the output limit. Continue from "
    "exactly where it stopped. Do not repeat anything already written, do not restate "
    "the question, and do not start over — continue the sentence."
)

COVERAGE = (
    "You are about to answer without having opened {files}, which rank among the most "
    "central files for this question. Open what is relevant, or say explicitly why it "
    "is not."
)

CUT_OFF = (
    "[the model's output was cut off at the output limit and could not be continued]"
)

MAX_CONTINUATIONS = 2


@dataclass(frozen=True, slots=True)
class Nudge:
    """One intervention: the text, and which gate produced it."""

    kind: str
    text: str

    @property
    def forces_terminal(self) -> bool:
        """Whether this nudge comes with ``tool_choice: none``."""
        return self.kind != "coverage"


def budget() -> Nudge:
    return Nudge(kind="budget", text=BUDGET)


def rounds() -> Nudge:
    return Nudge(kind="rounds", text=ROUNDS)


def dry() -> Nudge:
    return Nudge(kind="dry", text=DRY)


def continuation() -> Nudge:
    return Nudge(kind="continue", text=CONTINUE)


def coverage(files: tuple[str, ...]) -> Nudge:
    return Nudge(kind="coverage", text=COVERAGE.format(files=", ".join(files)))
