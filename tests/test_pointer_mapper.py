"""PointerMapper: reachable corners, steady cursor, no slingshots."""

from puppet_functions.pointer_mapper import PointerMapper

W, H = 1920, 1080


def _settled(mapper, nx, ny, t0=0.0, frames=120):
    """Hold a position still long enough for the filter to catch up."""
    t = t0
    for _ in range(frames):
        t += 1 / 60
        x, y = mapper.map(nx, ny, t)
    return x, y


def test_active_region_corners_reach_screen_corners(config):
    m = PointerMapper(W, H, config)          # default region is 0.15 .. 0.85
    assert _settled(m, 0.15, 0.15) == (0, 0)

    m = PointerMapper(W, H, config)
    x, y = _settled(m, 0.85, 0.85)
    assert (x, y) == (W - 1, H - 1)


def test_region_centre_maps_to_screen_centre(config):
    m = PointerMapper(W, H, config)
    x, y = _settled(m, 0.5, 0.5)
    assert abs(x - W // 2) < 3 and abs(y - H // 2) < 3


def test_outside_the_region_clamps_instead_of_overshooting(config):
    m = PointerMapper(W, H, config)
    x, y = _settled(m, -0.5, 2.0)
    assert x == 0 and y == H - 1


def test_freeze_pins_the_cursor(config):
    """What stops a click from landing next to the thing you aimed at."""
    m = PointerMapper(W, H, config)
    held = _settled(m, 0.5, 0.5)
    m.freeze()
    for i in range(30):
        assert m.map(0.8, 0.8, 2.0 + i / 60) == held
    m.unfreeze()
    assert m.map(0.8, 0.8, 3.0) != held


def test_unfreeze_resumes_without_a_jump(config):
    m = PointerMapper(W, H, config)
    before = _settled(m, 0.5, 0.5)
    m.freeze()
    m.map(0.5, 0.5, 3.0)
    m.unfreeze()
    after = m.map(0.5, 0.5, 3.1)
    assert abs(after[0] - before[0]) <= 1 and abs(after[1] - before[1]) <= 1


def test_unfreeze_does_not_teleport_when_the_hand_moved(config):
    """The bug behind "it clicks at a spot I didn't press": while frozen the
    filters used to sit idle, so the first sample afterwards arrived with a
    huge dt and flung the cursor to wherever the hand had drifted."""
    m = PointerMapper(W, H, config)
    aimed = _settled(m, 0.5, 0.5)

    m.freeze()
    t = 2.0
    for i in range(10):                       # hand drifts during the click
        t += 1 / 60
        assert m.map(0.5 + 0.03 * i / 10, 0.5, t) == aimed

    m.unfreeze()
    t += 1 / 60
    resumed = m.map(0.53, 0.5, t)
    assert abs(resumed[0] - aimed[0]) <= 3, f"teleported {resumed[0] - aimed[0]} px"


def test_the_offset_decays_so_aiming_stays_absolute(config):
    """Continuity must not cost accuracy: the correction washes out."""
    m = PointerMapper(W, H, config)
    _settled(m, 0.5, 0.5)
    m.freeze()
    t = 2.0
    for i in range(10):
        t += 1 / 60
        m.map(0.5 + 0.03 * i / 10, 0.5, t)
    m.unfreeze()

    for _ in range(120):                      # two seconds of holding still
        t += 1 / 60
        m.map(0.65, 0.5, t)

    reference = _settled(PointerMapper(W, H, config), 0.65, 0.5)
    assert abs(m.position[0] - reference[0]) <= 2


def test_a_held_button_keeps_the_offset(config):
    """During a drag the cursor must track the hand exactly — no creeping
    toward the absolute position while you're carrying something."""

    def run(hold):
        m = PointerMapper(W, H, config)
        _settled(m, 0.5, 0.5)
        m.freeze()
        t = 2.0
        for i in range(10):                   # hand drifts while pinned
            t += 1 / 60
            m.map(0.5 + 0.03 * i / 10, 0.5, t)
        m.unfreeze()
        m.hold_offset(hold)
        t += 1 / 60
        start = m.map(0.53, 0.5, t)
        for _ in range(60):                   # then holds still for a second
            t += 1 / 60
            m.map(0.53, 0.5, t)
        return start, m.position

    start_held, end_held = run(True)
    _, end_free = run(False)

    # Held: stays put (the few px are the 1-Euro filter finishing its
    # convergence, not the offset unwinding).
    assert abs(end_held[0] - start_held[0]) <= 8
    # Free: deliberately drifts back to the absolute position, which is a
    # meaningfully different place.
    assert abs(end_free[0] - end_held[0]) > 20


def test_reset_prevents_a_slingshot_on_re_acquisition(config):
    """Hand leaves at one corner, comes back at the other: no fly-across."""
    m = PointerMapper(W, H, config)
    _settled(m, 0.2, 0.2)
    m.reset(0.8, 0.8, 100.0)
    x, y = m.position
    assert x > W * 0.8 and y > H * 0.8


def test_a_degenerate_region_falls_back_to_the_full_frame(config):
    class Bad:
        active_region = [0.5, 0.5, 0.5, 0.5]

    m = PointerMapper(W, H, Bad())
    assert _settled(m, 0.0, 0.0) == (0, 0)
    m = PointerMapper(W, H, Bad())
    assert _settled(m, 1.0, 1.0) == (W - 1, H - 1)


def test_output_is_always_on_screen(config):
    m = PointerMapper(W, H, config)
    t = 0.0
    for nx in (0.0, 0.3, 0.5, 0.9, 1.0):
        for _ in range(10):
            t += 1 / 60
            x, y = m.map(nx, 1 - nx, t)
            assert 0 <= x < W and 0 <= y < H
