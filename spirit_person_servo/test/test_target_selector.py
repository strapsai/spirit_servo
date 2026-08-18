"""Target-selection tests.

Uses synthetic Track objects, so nothing here depends on ultralytics or a GPU --
that seam is the whole reason the tracker implementation is swappable.
"""

import pytest

from spirit_person_servo.target_selector import TargetSelector
from spirit_person_servo.tracker import Detection, Track


def track(track_id: int, size: float, now: float = 0.0, x: float = 0.0) -> Track:
    return Track(
        track_id=track_id,
        detection=Detection(x=x, y=0.0, w=size, h=size * 2.0, confidence=0.9),
        last_seen=now,
    )


def lock_onto(selector: TargetSelector, tracks: list[Track], start: float = 0.0) -> Track:
    """Feed enough frames to satisfy the acquisition vote."""
    result = None
    for i in range(selector.lock_votes_required):
        result = selector.select(tracks, now=start + i * 0.1)
    assert result is not None, "expected the selector to lock on"
    return result


class TestAcquisition:
    def test_no_tracks_selects_nothing(self):
        assert TargetSelector().select([], now=0.0) is None

    def test_requires_repeated_votes_before_locking(self):
        selector = TargetSelector(lock_votes_required=3)
        tracks = [track(1, 100.0)]
        assert selector.select(tracks, now=0.0) is None
        assert selector.select(tracks, now=0.1) is None
        assert selector.select(tracks, now=0.2) is not None

    def test_picks_the_largest_box(self):
        selector = TargetSelector()
        tracks = [track(1, 50.0), track(2, 200.0), track(3, 100.0)]
        assert lock_onto(selector, tracks).track_id == 2

    def test_flickering_largest_delays_lock(self):
        """Two similar people swapping 'largest' must not produce an instant lock."""
        selector = TargetSelector(lock_votes_required=3, vote_window=5)
        a_bigger = [track(1, 101.0), track(2, 100.0)]
        b_bigger = [track(1, 100.0), track(2, 101.0)]
        assert selector.select(a_bigger, now=0.0) is None
        assert selector.select(b_bigger, now=0.1) is None
        assert selector.select(a_bigger, now=0.2) is None
        assert selector.select(b_bigger, now=0.3) is None


class TestContinuity:
    def test_holds_target_even_when_it_stops_being_largest(self):
        selector = TargetSelector()
        held = lock_onto(selector, [track(1, 200.0), track(2, 50.0)])
        assert held.track_id == 1

        # Someone bigger walks in: we must keep following the original target.
        for i in range(20):
            got = selector.select([track(1, 200.0), track(2, 900.0)], now=1.0 + i * 0.1)
            assert got is not None and got.track_id == 1

    def test_reports_updated_geometry_for_held_track(self):
        selector = TargetSelector()
        lock_onto(selector, [track(1, 100.0)])
        moved = selector.select([track(1, 100.0, x=640.0)], now=1.0)
        assert moved is not None
        assert moved.detection.x == 640.0

    def test_brief_dropout_does_not_retarget(self):
        # lock_onto consumes t=0.0..0.2, so _last_seen is 0.2 and the 0.4 s coast
        # window runs to t=0.6.
        selector = TargetSelector(max_coast_s=0.4)
        lock_onto(selector, [track(1, 200.0), track(2, 100.0)])

        # Held track missing but within the coast window: no fresh target, but
        # crucially we must NOT switch to the other person.
        assert selector.select([track(2, 100.0)], now=0.4) is None
        assert selector.held_id == 1

        recovered = selector.select([track(1, 200.0), track(2, 100.0)], now=0.5)
        assert recovered is not None and recovered.track_id == 1

    def test_retargets_after_coast_window_expires(self):
        selector = TargetSelector(max_coast_s=0.4, lock_votes_required=1)
        lock_onto(selector, [track(1, 200.0)], start=0.0)
        assert selector.held_id == 1

        # Well past the coast window with only a different person present: the
        # stale target is dropped and, since one vote suffices, the new person is
        # acquired in the same call.
        reacquired = selector.select([track(2, 100.0)], now=10.0)
        assert reacquired is not None and reacquired.track_id == 2
        assert selector.held_id == 2

    def test_empty_frames_beyond_coast_clear_the_target(self):
        selector = TargetSelector(max_coast_s=0.4)
        lock_onto(selector, [track(1, 200.0)])
        assert selector.select([], now=0.4) is None      # coasting (window ends at 0.6)
        assert selector.held_id == 1
        assert selector.select([], now=5.0) is None      # expired
        assert selector.held_id is None

    def test_release_drops_the_target(self):
        selector = TargetSelector()
        lock_onto(selector, [track(1, 200.0)])
        selector.release()
        assert selector.held_id is None

    def test_release_allows_choosing_a_different_person(self):
        selector = TargetSelector(lock_votes_required=1)
        lock_onto(selector, [track(1, 200.0), track(2, 900.0)], start=0.0)
        assert selector.held_id == 2
        selector.release()
        got = selector.select([track(1, 200.0)], now=5.0)
        assert got is not None and got.track_id == 1


class TestVoteWindow:
    def test_stale_votes_fall_out_of_the_window(self):
        """A consistent winner must still lock even after unrelated early votes."""
        selector = TargetSelector(lock_votes_required=3, vote_window=5)
        for i in range(4):
            selector.select([track(99, 500.0 if i % 2 else 10.0), track(7, 100.0)], now=i * 0.1)
        result = None
        for i in range(3):
            result = selector.select([track(7, 100.0)], now=1.0 + i * 0.1)
        assert result is not None and result.track_id == 7

    def test_reset_clears_votes(self):
        selector = TargetSelector(lock_votes_required=3)
        selector.select([track(1, 100.0)], now=0.0)
        selector.select([track(1, 100.0)], now=0.1)
        selector.reset()
        assert selector.select([track(1, 100.0)], now=0.2) is None


@pytest.mark.parametrize("votes_required", [1, 2, 5])
def test_lock_threshold_is_configurable(votes_required):
    selector = TargetSelector(lock_votes_required=votes_required, vote_window=10)
    tracks = [track(1, 100.0)]
    for i in range(votes_required - 1):
        assert selector.select(tracks, now=i * 0.1) is None
    assert selector.select(tracks, now=votes_required * 0.1) is not None
