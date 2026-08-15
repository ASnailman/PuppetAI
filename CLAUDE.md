# PuppetAI — Architecture Notes

Local, offline gesture control: webcam → MediaPipe hand landmarks → mouse and
keyboard input. No cloud, no LLM. User-facing docs are in `README.md`; this
file is the map of the internals.

## Layout

```
PuppetAI/
├── puppet_overlay.py           # entry point — threading and wiring only
├── config.json                 # all tunables, hot-reloaded while running
├── requirements.txt            # direct deps only
└── puppet_functions/
    ├── camera.py               # threaded capture, always the newest frame
    ├── filters.py              # OneEuroFilter, Hysteresis, EMA
    ├── hand_model.py           # landmarks → scale/rotation-invariant features
    ├── gesture_recognizer.py   # features → posture + pinch state
    ├── pointer_mapper.py       # fingertip → screen pixel
    ├── gesture_actions.py      # gestures → pyautogui calls
    ├── config_manager.py       # defaults, validation, hot reload
    └── point_overlay.py        # transparent Tk overlay + HUD
```

Run: `python puppet_overlay.py`   Test: `python -m pytest tests -q`

## Data flow

```
Camera.read() → cv.flip → MediaPipe Hands
  → HandModel.update(landmarks)   -> HandFrame  (booleans + palm-normalised distances)
  → GestureRecognizer.update()    -> GestureState (posture, index_pinch, middle_pinch)
  → GestureActions.execute()      -> pyautogui, via PointerMapper for coordinates
  → PointOverlay.update_state()   -> HUD (marshalled onto the Tk thread)
```

Tk owns the main thread; the vision pipeline runs on a daemon worker
(`Pipeline.run`). The overlay's public methods are the thread boundary — they
all go through `root.after`.

## The two invariants everything rests on

1. **Palm-relative measurement.** Every distance is divided by
   `dist(wrist, middle_mcp)`. Thresholds are therefore in palm-lengths and hold
   at any camera distance. Never introduce a threshold in raw normalised image
   units — that was the original design's central flaw.

2. **Rotation-invariant extension.** A finger is extended when
   `dist(tip, wrist) - dist(pip, wrist)` exceeds a threshold. Never compare
   `tip.y < mcp.y`; that only works with the hand held upright.

3. **The cursor is anchored to the palm, not the fingertip.**
   `HandModel._cursor_point` projects a point one palm-length out from the
   index MCP along the wrist→MCP axis. The index tip is the *worst* possible
   cursor source, because pinching necessarily moves it — measured at ~32 px
   of drift per click. Finger articulation must never move the cursor.

Corollaries worth preserving:
- Ring and pinky are ignored during `MOVE`. They move involuntarily.
- Pinches are gated on **reach** (fingertip-to-wrist distance), never on
  extension. Some gate is needed because a fist's thumb rests on the curled
  index finger and reads as a pinch — but extension is the wrong one: pinching
  *is* bending the finger, so past ~60° of PIP flexion an extension test goes
  false at the exact moment the user wants to click.
- Any threshold on the *degree* of a pinch must clear the hardest pinch a user
  can make with real margin. `pinch_reach_on` was once 1.35, which sits between
  a firm pinch (1.44) and a very firm one (1.29) — so pinching harder, the
  natural response to a click not registering, made it fire *less* often.
  `test_the_pinch_gate_has_margin_at_both_ends` guards this.
- Postures are debounced by a time-windowed majority vote; pinches are not
  (they rely on hysteresis). Putting clicks behind the posture vote is what
  made clicking feel late.

## Gesture set

| Posture | Condition | Action |
|---|---|---|
| `MOVE` | index extended | cursor follows the index tip |
| `SCROLL` | index + middle extended, thumb tucked | offset from anchor → scroll speed |
| `ZOOM` | index + middle extended, thumb splayed | same, with Ctrl held |
| `NEUTRAL` | fist or open palm | nothing |

Pinch modifiers, tracked independently of posture: thumb↔index = click / drag,
thumb↔middle = right click.

Clicking is a plain press/release pair: `mouseDown` the moment the pinch is
detected, `mouseUp` when it opens. There is no click-vs-drag classification and
no hold timer — a drag is just a press that moved, and once the hand travels
past `drag_slop_palm` the cursor is unfrozen to follow it. Do not reintroduce
"fire `click()` on release": it made every click depend on detecting a release
within a time limit, so a long hold became a drag and a dropped frame lost the
click outright. Double-click is likewise not detected here — two presses inside
the OS's own window already register as one.

## Invariants to not break

- `GestureActions.release_all()` must be called on pause, on hand loss, and on
  shutdown. It is idempotent. Without it a vanished hand leaves a mouse button
  held down.
- `PointerMapper.freeze()` during a click; `reset()` on re-acquisition, or the
  cursor slings across the screen from the hand's last known position.
- **`map()` must feed the 1-Euro filters on every frame, including frozen
  ones**, and `unfreeze()` must rebase onto the next filtered sample. Returning
  early while frozen left the filters stale, so the next sample arrived with a
  dt spanning the whole freeze, the adaptive cutoff opened fully, and the
  cursor teleported — 45 px after a click, 126 px starting a drag. That is what
  "it clicks somewhere I didn't press" and "drag doesn't work" both were.
  `test_unfreeze_does_not_teleport_when_the_hand_moved` guards it.
- The continuity offset must be held (`hold_offset`) while a button is down, or
  the cursor creeps toward absolute mid-drag and drops things off-target.
- Three layers protect a click's aim, and all three are load-bearing: the
  palm-anchored cursor point, `GestureActions._cursor_damping` fading the gain
  to zero across the pinch approach, and the `click_settle_ms` window that
  keeps the cursor pinned while the fingers spring back apart. Dragging is
  exempt from damping or it would freeze solid.
- The camera window is driven by `Pipeline.feed_visible`, not by
  `config.show_debug_feed` directly: the config value is the default, and a
  tray click or `C` overrides it until that value itself changes. Unrelated
  config edits must leave the override alone — tuning thresholds with the feed
  open is the whole point of having it. HighGUI is not thread-safe, so
  `imshow`/`destroyAllWindows` stay on the vision thread; the tray callback
  only flips a flag, exactly like `toggle_pause`.
- `pyautogui.FAILSAFE = False` and `PAUSE = 0` are set in `main()`; the default
  100 ms pause would cap the app at 10 actions per second. But `PAUSE` was also
  silently providing a settle between moving the cursor and pressing the
  button, which the original version depended on — `click_press_delay_ms`
  replaces it explicitly, only on the press. Button events also carry their own
  coordinates so a press can't land where a dropped moveTo left the cursor.

## Tests

`tests/conftest.py` builds synthetic 21-landmark hands from a described pose
(`make_hand`), and `FakeBackend` records what would have gone to pyautogui. No
webcam or real input is involved, so the suite runs in well under a second.

When changing gesture geometry, the tests that matter are
`test_hand_model.py::test_same_pose_at_different_distances_reads_identically`
and `::test_rotated_hand_reads_identically` — they encode the two invariants
above.
