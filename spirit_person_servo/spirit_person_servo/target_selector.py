"""Chooses which tracked person to servo onto.

Phase 1 policy: largest bbox, held with frame-to-frame continuity.

This module is the seam for the re-ID phase. Selection becomes "score every track
against the gallery descriptor, take the best above tau" and *nothing else in the
package changes* -- tracker, controller, state machine, and safety layers are all
independent of how the target is picked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tracker import Track


@dataclass
class TargetSelector:
    """Sticky largest-bbox selection.

    Continuity matters more than instantaneous size: two people at similar
    distance trade the "largest" title constantly as their boxes breathe, and
    chasing that would slew the gimbal back and forth between them.

    BoT-SORT already re-associates a briefly-occluded person to the same
    ``track_id``, so the coast window here only has to cover the frames where the
    held track is absent from the output entirely.
    """

    lock_votes_required: int = 3
    vote_window: int = 5
    max_coast_s: float = 0.4

    _held_id: int | None = field(default=None, init=False)
    _votes: list[int] = field(default_factory=list, init=False)
    _last_seen: float | None = field(default=None, init=False)

    @property
    def held_id(self) -> int | None:
        return self._held_id

    def reset(self) -> None:
        self._held_id = None
        self._votes.clear()
        self._last_seen = None

    def select(self, tracks: list[Track], now: float) -> Track | None:
        """Return the track to servo onto this cycle, or None if there is none.

        None while coasting means "no fresh target, keep holding" -- it is the
        caller's staleness timer, not this method, that decides when to give up.
        """
        by_id = {t.track_id: t for t in tracks}

        if self._held_id is not None:
            held = by_id.get(self._held_id)
            if held is not None:
                self._last_seen = now
                return held
            if self._coasting(now):
                return None
            # Coast window expired: the held target is genuinely gone.
            self.reset()

        if not tracks:
            return None

        largest = max(tracks, key=lambda t: t.area)
        self._vote(largest.track_id)
        if self._confirmed(largest.track_id):
            self._held_id = largest.track_id
            self._last_seen = now
            self._votes.clear()
            return largest
        return None

    def release(self) -> None:
        """Drop the current target (lost for good, or the loop was disarmed)."""
        self.reset()

    def _coasting(self, now: float) -> bool:
        return self._last_seen is not None and (now - self._last_seen) <= self.max_coast_s

    def _vote(self, track_id: int) -> None:
        self._votes.append(track_id)
        if len(self._votes) > self.vote_window:
            self._votes.pop(0)

    def _confirmed(self, track_id: int) -> bool:
        return self._votes.count(track_id) >= self.lock_votes_required
