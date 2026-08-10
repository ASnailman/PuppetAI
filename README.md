# PuppetAI

Become a puppeteer. Dance, computer, dance!

PuppetAI watches your hand through your webcam and turns it into your mouse.
Point to move the cursor, tap your thumb against your index finger to click,
hold that pinch to drag, put two fingers up to scroll. It runs entirely on your
machine — no cloud, no account, no video ever leaves the room.

It's built on Google's MediaPipe hand model, which finds 21 landmarks on your
hand in every frame. PuppetAI's job is the part after that: working out what
you meant by the shape your hand is in, without making you hold still like a
statue.

## Demo

## What you need

- A webcam — any built-in laptop camera is fine
- Python 3.12
- Windows. It runs elsewhere, but the click-through overlay uses a Win32 call,
  so on other systems the overlay may swallow clicks.

## Setup

```bash
cd PuppetAI
python -m venv puppet_env
puppet_env\Scripts\activate        # macOS/Linux: source puppet_env/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
python puppet_overlay.py
```

Hold your hand up to the camera, palm toward the lens, and point with your
index finger. A coloured dot shows where PuppetAI thinks you're pointing, with
a status readout in the top-left corner.

To stop: right-click the tray icon and choose Quit.

## Gestures

| What you do                                          | What happens                 |
| ---------------------------------------------------- | ---------------------------- |
| Point with your index finger                         | The cursor follows your hand |
| Tap your thumb to your index fingertip               | Click                        |
| Tap twice, quickly                                   | Double click                 |
| Hold the pinch and move your hand                    | Click and drag               |
| Index + middle up, tap thumb to the middle fingertip | Right click                  |
| Index + middle up, move your hand up or down         | Scroll                       |
| Same, with your thumb held out to the side           | Zoom                         |
| Open your hand, or make a fist                       | Rest — nothing happens      |

A few things worth knowing:

- **Your ring and pinky fingers don't matter** while pointing. They curl on
  their own, and insisting they behave is what makes gesture control fussy.
- **Scroll and zoom are speed controls.** Form the posture, then move your hand
  away from where you started — a little offset creeps, a bigger one moves
  fast. Bring your hand back to the middle to stop.
- **The cursor doesn't come from your fingertip.** The fingertip is the one
  point on your hand that has to move in order to pinch, so PuppetAI tracks a
  point anchored to your palm instead. It steers just like pointing, but
  clicking doesn't jog your aim.
- **A ring closes around the dot as your pinch tightens**, so you can see a
  click coming before it fires.

Settings live in `config.json`, and the app re-reads it about once a second
while running — save the file and you'll feel the change immediately. Delete a
key to get its default back.

## If something's wrong

**"Could not open a webcam"** — another app has the camera. Close Zoom, Teams,
or a browser tab using it. If you have several cameras, set `camera_index` to
`1`, `2`, ...

**Nothing is detected** — the model needs reasonable light and your whole hand
in frame. Set `"show_debug_feed": true` to watch what the camera sees, with the
landmarks drawn on top. With that window focused, `G` shows the gesture guide,
`P` pauses, and `Q` quits.

**Clicks do nothing on one particular window** — Windows blocks synthetic input
from reaching programs running as administrator unless the sender is elevated
too. Either run that program normally, or start PuppetAI as administrator.

**Clicks don't always register** — set `"show_diagnostics": true` and the HUD
will show the two numbers a click depends on, live. `pinch` has to drop below
`pinch_on` and `reach` has to stay above `pinch_reach_on`; the readout says
which one isn't making it. Both are in `config.json` and take effect while the
app is running, so you can nudge them with your hand in front of the camera.

**It clicked something I didn't mean to** — pause from the tray icon. Pausing
always releases any held button, so you can't get stuck mid-drag.

**The overlay eats my clicks** — set `"show_overlay": false` and use the real
cursor.
