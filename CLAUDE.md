# PuppetAI — Architecture & Expansion Roadmap

## What This App Does

PuppetAI is a local gesture-based computer control system. It uses Google's MediaPipe hand model to detect hand landmarks from a webcam and translates them into mouse/keyboard actions (cursor move, scroll, click, etc.). All processing runs locally — no cloud or LLM calls.

---

## Current File Structure

```
PuppetAI/
├── puppet_overlay.py           # Main entry point (active version)
├── puppet.py                   # Legacy standalone version (reference only)
├── requirements.txt
└── puppet_functions/
    ├── point_overlay.py        # Transparent Tkinter overlay window
    ├── hand_positions.py       # GestureRecognizer stub (unimplemented)
    └── __pycache__/
```

**Run the app:** `python puppet_overlay.py` (from activated `puppet_env`)

---

## Known Bugs (Fix Before Adding Features)

1. `hands.process(frame)` passes BGR frame — should be `rgb_frame` (MediaPipe expects RGB)
2. `puppet_overlay.py` is missing cursor move and left click — they exist in `puppet.py` but were lost during the overlay refactor
3. Scroll condition `pointer.y and middle.y > prev_y` is a Python precedence bug — `and` is boolean here, not element-wise

---

## Target File Structure (After Expansion)

```
PuppetAI/
├── puppet_overlay.py           # Main entry point (refactored)
├── puppet.py                   # Retired — keep as backup
├── config.json                 # User-editable settings
├── requirements.txt            # Add pystray
└── puppet_functions/
    ├── point_overlay.py        # Updated: mode label, dot color per gesture
    ├── hand_positions.py       # IMPLEMENT: GestureRecognizer class
    ├── gesture_actions.py      # New: maps gesture names → system actions
    └── config_manager.py       # New: load/save config.json
```

---

## Implementation Phases

### Phase 1 — Bug Fixes & Consolidation (`puppet_overlay.py`)
- Pass `rgb_frame` to `hands.process()`
- Fix scroll comparison bug
- Restore cursor move gesture from `puppet.py`
- Restore left click gesture from `puppet.py` (with `clicked` flag debounce)
- Make OpenCV debug window optional via config

### Phase 2 — Gesture State Machine (`hand_positions.py`)

Implement `GestureRecognizer` to replace raw coordinate comparisons in the main loop.

```python
class GestureRecognizer:
    def classify(self, landmarks) -> str: ...
    def get_stable_gesture(self, landmarks) -> str: ...
        # Requires N consecutive matching frames before emitting
```

**Gesture dictionary (E=extended, B=bent):**

| Gesture        | Index | Middle | Ring | Pinky | Thumb               |
|----------------|-------|--------|------|-------|---------------------|
| `MOVE`         | E     | B      | B    | B     | out (x > CMC)       |
| `SCROLL`       | E     | E      | B    | B     | out                 |
| `LEFT_CLICK`   | E     | B      | B    | B     | in (< CMC - 0.12)   |
| `RIGHT_CLICK`  | B     | B      | B    | E     | out                 |
| `DOUBLE_CLICK` | E     | B      | B    | B     | in (2× fast)        |
| `DRAG`         | E     | B      | B    | B     | pinch to index tip  |
| `ZOOM_IN`      | E     | B      | B    | B     | thumb above index   |
| `ZOOM_OUT`     | E     | E      | B    | B     | thumb close to mid  |
| `NEUTRAL`      | B     | B      | B    | B     | any (fist)          |

**Debounce:** Keep rolling buffer of last N frames (default 4). Only emit when all N agree. Prevents flicker.

**Double-click:** Track timestamp of last `LEFT_CLICK`; if another fires within 400ms, emit `DOUBLE_CLICK`.

**Drag:** `DRAG` gesture → `mouseDown()`; any other gesture → `mouseUp()`.

### Phase 3 — Gesture Actions (`gesture_actions.py`)

```python
class GestureActions:
    def execute(self, gesture: str, landmarks): ...
```

- `MOVE` → `pyautogui.moveTo(x, y)`
- `SCROLL` → `pyautogui.scroll(amount)` (y-delta × scroll_multiplier)
- `LEFT_CLICK` → `pyautogui.click()`
- `RIGHT_CLICK` → `pyautogui.rightClick()`
- `DOUBLE_CLICK` → `pyautogui.doubleClick()`
- `DRAG` → `mouseDown()` / `mouseUp()`
- `ZOOM_IN` → `pyautogui.hotkey('ctrl', '=')`
- `ZOOM_OUT` → `pyautogui.hotkey('ctrl', '-')`

### Phase 4 — Config System (`config_manager.py` + `config.json`)

```json
{
  "scroll_multiplier": 250,
  "scroll_tolerance": 0.02,
  "gesture_stability_frames": 4,
  "double_click_window_ms": 400,
  "show_debug_feed": false,
  "overlay_dot_color": "red",
  "overlay_dot_radius": 7.5,
  "zoom_in_hotkey": ["ctrl", "="],
  "zoom_out_hotkey": ["ctrl", "-"]
}
```

`ConfigManager` loads at startup, provides typed attribute access, falls back to defaults if file missing.

### Phase 5 — UX Improvements (`point_overlay.py`)

1. **Mode label** — display active gesture name (e.g., "MOVE", "SCROLL") in a corner of the overlay
2. **Dot color per mode** — MOVE=red, SCROLL=blue, CLICK=green flash
3. **Gesture guide toggle** — press `G` to show/hide on-screen cheat sheet of all gestures
4. **System tray icon** (via `pystray`):
   - Pause/Resume gesture tracking
   - Open config editor (Tkinter form)
   - Quit

---

## Dependencies

**To add:** `pystray`

**Already present, unused:** `sounddevice`, `PyGetWindow` — no plans to use these.

**Pinned versions are in `requirements.txt`.**

---

## Verification Checklist

| Phase | Test |
|-------|------|
| 1 | Cursor follows index finger; left click fires; scroll works up/down |
| 2 | Gestures are stable, no flicker; neutral fist stops all action |
| 3 | Right-click, double-click, drag, zoom all fire correctly |
| 4 | Edit `config.json`, restart, confirm values changed |
| 5 | Mode label visible; tray icon shows with working menu |
