"""Closing the loop on an estimator that is knowingly wrong."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import CALIBRATOR_ALPHA, CALIBRATOR_CLAMP


@dataclass(frozen=True, slots=True)
class TokenCalibrator:
    """An EMA correcting the heuristic against what providers actually charge."""

    ratio: float = 1.0
    samples: int = 0
    discarded: int = 0
    """Observations rejected by the clamp. Worth counting: a calibrator that has
    discarded more than it accepted is not calibrating, it is being fed
    something other than what it estimated, and the ratio it reports is a
    number nobody should act on."""

    def estimate(self, heuristic: int) -> int:
        """Correct a raw heuristic estimate."""
        return max(0, round(heuristic * self.ratio))

    def observe(self, heuristic: int, actual_input: int) -> TokenCalibrator:
        """Fold one measurement in, or refuse it."""
        if heuristic <= 0 or actual_input <= 0:
            return self

        low, high = CALIBRATOR_CLAMP
        sample = actual_input / heuristic
        if not low <= sample <= high:
            return TokenCalibrator(
                ratio=self.ratio, samples=self.samples, discarded=self.discarded + 1
            )

        alpha = 1.0 if self.samples == 0 else CALIBRATOR_ALPHA
        return TokenCalibrator(
            ratio=self.ratio * (1.0 - alpha) + sample * alpha,
            samples=self.samples + 1,
            discarded=self.discarded,
        )

    @property
    def correction(self) -> str:
        """The per-turn report. ``+18%`` reads better than ``1.18`` in a log."""
        return f"{(self.ratio - 1.0) * 100:+.0f}%"
