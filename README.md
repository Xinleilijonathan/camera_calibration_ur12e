# UR12e + AprilTag Camera Calibration

Guided, manual, high-accuracy calibration of **three independent cameras** against a
printed AprilTag grid, using a UR12e that **you** move by keyboard or Xbox controller.

The program never generates or executes robot poses on its own. It watches the board,
tells you whether the current pose is worth recording, suggests which joint to nudge
next, and records the robot's *actual* measured state when you press the record key.

---

## Status

| Phase | What | State |
|---|---|---|
| 1 | Repository inspection | done |
| 2 | Folder + config structure | done |
| 3 | Camera enumeration by serial | done |
| 4 | Live camera preview | done |
| 5 | AprilTag grid detection | done |
| 6–7 | Intrinsic collection + solver | done |
| 8–9 | Read-only robot connection + state display | done |
| 10–12 | Keyboard / gamepad control, joint-priority guidance | done |
| 13–14 | Waypoint recording, pose-diversity analysis | done |
| 15–19 | Preliminary solve, scoring, best-20 selection, final solve, validation | done |
| 20–21 | Replay dry-run, docs/tests | done |

Every script is implemented and covered by tests (458 of them, all passing, none of which
open a camera or a robot). What has **not** happened is a run against real hardware: the
whole pipeline has so far only been exercised against synthetic datasets and URSim.

Motion code now exists — `jog_controller.py`, `robot_interface.enable_motion()` and the
teleop scripts can command the arm. It is inert until you change `config/safety.yaml`
**and** type a confirmation at startup. Read the next section before you do either.

---

## Safety

Read this before the robot is ever involved.

* `config/safety.yaml` ships with `allow_motion: false` **and** `allow_physical_robot: false`.
  Both are deliberately config settings, not command-line flags, so no command you type by
  accident can enable motion.
* Motion additionally requires typed confirmation at startup (`preflight.require_user_confirmation`).
  A keypress is not enough; `robot_interface.enable_motion()` refuses without `confirm=True`
  from a caller that obtained that typed consent.
* `robot_ip` defaults to `127.0.0.1` (URSim). Everything must be proven in URSim first.
* The existing `~/leader_arm` project documents that the physical UR12e has **not** had its
  safety limits, collision checks, tool/payload configuration or e-stop interlock
  commissioned. Until that work is done, do not point this software at the real arm.
* Nothing in this project disables or works around the UR controller's own safety
  functions. `safety.yaml` only makes *this program* refuse to send commands; the
  controller's protective stop, safety planes and joint limits remain the real protection
  and must be configured on the teach pendant.
* You can collect a complete dataset with **no motion code active at all**: pass
  `--read-only` and move the arm by hand (freedrive). The robot is then read, never
  commanded. This is the recommended way to take a first dataset.

Three scripts never command the robot under any configuration, because they contain no
motion code: `list_cameras.py`, `preview_apriltag.py`, `collect_intrinsics.py`.
`verify_calibration.py` reads the robot but never commands it.

---

## Install

```bash
cd ~/camera_calibration
.venv/bin/python -m pip install -r requirements.txt
```

Most of it is already present. Verify:

```bash
.venv/bin/python -c "import cv2, numpy, scipy, yaml, pygame, rtde_receive; print('ok')"
```

### Which machine does this run on?

The cameras and the robot must be on the **same** machine as this checkout, because the
scripts open the cameras and the RTDE connection from the same process. The development
laptop (`jonathancyriane-dell-g15-5530`) sees only its own built-in webcam, so a real
session runs on the robot PC, not here.

### The RealSense backend

All three cameras on this rig are RealSense (a D405 and two D435s), so this backend is not
optional. `pyrealsense2` **is** installed in `.venv` (2.58.4). Check it:

```bash
.venv/bin/python -c "import pyrealsense2 as rs; print(rs.__version__)"
```

That wheel bundles its own `librealsense`, which shadows the source build in
`~/librealsense` (binaries in `/usr/local`). Harmless as long as only one is in play. If
you hit a firmware or version mismatch against the source build, uninstall the wheel and
build the bindings from that tree instead, with
`-DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$PWD/.venv/bin/python`.

Plain USB/UVC webcams need nothing extra — the `v4l2` backend works today.

### Running tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q
```

The environment variable is required because this machine sources `/opt/ros/jazzy` in the
login shell, which puts ROS's pytest plugins on the path; they fail to load for unrelated
reasons and have nothing to do with this project.

The suite takes about two minutes. Most of that is `test_pipeline_integration.py`, which
builds synthetic datasets with a known hand-eye transform and runs the real scripts as
subprocesses to check the pipeline recovers it — for both mountings.

---

## Before you start: this rig uses BOTH mountings

`camera_1` is the D405 on the wrist; `camera_2` and `camera_3` are the D435s fixed at the
side of the cell. That means the **board is in a different place** for the two sessions:

| Camera | Mounting | Where the board goes | What is solved |
|---|---|---|---|
| `camera_1` | `eye_in_hand` | **fixed on the table** | camera → tool flange |
| `camera_2`, `camera_3` | `eye_to_hand` | **bolted to the robot flange** | camera → robot base |

So you will physically re-rig the board between the `camera_1` session and the
`camera_2`/`camera_3` sessions. `camera_2` and `camera_3` can share one board mounting,
and if both see the board at once, one arm session.

The mode is a property of the **camera**, set as `handeye_mode` on each camera in
`config/cameras.yaml` (falling back to `calibration.yaml` → `handeye.mode`). It is never
inferred. Getting it backwards produces a result that looks numerically plausible and is
geometrically meaningless.

The D405 also has a much shorter working range than the D435 — roughly 7–50 cm against
30 cm and up. Place the board accordingly for each session.

---

## The complete process

### STEP 1 — Connect the robot (or URSim)

Start URSim:

```bash
docker run --rm -d \
  -e ROBOT_MODEL=UR12e \
  -p 5900:5900 -p 6080:6080 -p 29999:29999 -p 30001-30004:30001-30004 \
  --name ursim universalrobots/ursim_e-series
```

Open <http://localhost:6080>, power on the arm and put it in **RUNNING** mode. Confirm the
robot is reachable before continuing.

### STEP 2 — Enumerate the cameras

```bash
.venv/bin/python scripts/list_cameras.py
.venv/bin/python scripts/list_cameras.py --verbose    # every supported mode
```

Cameras are addressed by **serial number**, never by `/dev/videoN` — those indices are
reassigned on every reboot and replug, and calibrating `camera_2` with `camera_3`'s pixels
would be silent and unrecoverable.

### STEP 3 — Record the serials

```bash
.venv/bin/python scripts/list_cameras.py --emit-yaml
```

Paste the generated block into `config/cameras.yaml`, replacing the three
`REPLACE_WITH_..._SERIAL` placeholders. Give each camera a `description` recording where it
is physically mounted, so `camera_1` still means something next month. Scripts refuse to
run against a placeholder serial.

If a camera reports no serial number, use its `usb path` instead and do not move it to a
different USB port afterwards.

### STEP 4 — Enter the exact board geometry

Edit `config/calibration.yaml` → `apriltag_grid`:

```yaml
tag_family: tag36h11
rows: 6
columns: 6
tag_size_m: 0.030        # side of the BLACK SQUARE, in METRES
tag_spacing_m: 0.009     # WHITE GAP between neighbouring black squares
verified_by_user: true   # set this only after you have measured
```

**Measure the printed board with a caliper. Do not trust the PDF it came from** — printer
scaling is routinely off by a percent or two.

`tag_size_m` is the outer edge of the black border, which is exactly where the detected
corners lie. It is not the white quiet zone and not the cell pitch. If your board is
specified Kalibr-style with a spacing *ratio* `r`, then `tag_spacing_m = r * tag_size_m`.

This matters more than anything else in this file: an error in `tag_size_m` scales your
entire hand-eye translation by the same factor **without** inflating reprojection error,
so no downstream check can catch it. That is why scripts that emit metric results refuse
to run until `verified_by_user: true`.

### STEP 5 — Preview the board

```bash
.venv/bin/python scripts/preview_apriltag.py --camera camera_1
```

Shows tag outlines, IDs, corners, tag count, **VALID / INVALID**, resolution, and — once
intrinsics exist — the board pose, distance, tilt and PnP residual. Keys: `Q`/`ESC` quit,
`S` save frame, `H` toggle help. Add `--no-intrinsics` to skip the pose display, or
`--save-frame PATH` to grab a single frame and exit.

This script contains no robot code at all and cannot move anything.

Use it to sanity-check lighting and placement, and to learn what `Sharpness` reads on a
*good* frame for your camera, so you can set `detection.minimum_sharpness` sensibly.

Repeat for `camera_2` and `camera_3`.

### STEPS 6–7 — Intrinsics, per camera

```bash
.venv/bin/python scripts/collect_intrinsics.py --camera camera_1   # 20-40 varied views
.venv/bin/python scripts/solve_intrinsics.py   --camera camera_1
```

In the collector: `SPACE` records the current view, `U` undoes the last one, `Q`/`ESC`
finishes, `H` toggles help. Only valid *and genuinely novel* views count toward the target —
`--allow-similar` disables the novelty check, and `--force` overwrites an existing set.

Vary board position across the frame, orientation, tilt and distance. Thirty near-identical
images give a confidently wrong result; the collector enforces diversity for this reason.

The solver writes `data/camera_N/intrinsics/result.yaml` with `fx, fy, cx, cy`, distortion,
resolution and error statistics. Use `--dry-run` to see the numbers without writing,
`--exclude N ...` to drop specific observations, and `--auto-prune` to drop the worst
outliers automatically.

Then repeat, completely independently, for `camera_2` and `camera_3`. **Intrinsics are
never shared between cameras**, even if the cameras are the same model — sensor placement
and lens variation are per-unit.

### STEPS 8–12 — Collect 30 waypoints for camera 1

```bash
.venv/bin/python scripts/collect_waypoints.py --camera camera_1 --target-count 30 --read-only
```

The script connects read-only first and prints the camera, mounting, board spec and the
full safety state. Nothing can move until you type the confirmation phrase. With
`--read-only` nothing can move at all and you jog the arm by hand.

Three scripts share this session — they differ only in default input device and whether
recording is on:

| Script | Input | Recording |
|---|---|---|
| `collect_waypoints.py` | keyboard (`--input gamepad` to switch) | on |
| `keyboard_teleop_calibration.py` | keyboard | on (`--no-record` for framing only) |
| `gamepad_teleop_calibration.py` | gamepad (`--probe` to check the mapping) | on (`--no-record`) |

Keyboard:

```
JOINT MODE   1-6 select joint   [ / ] move it   arrows also move it
CARTESIAN    W/S +-X   A/D +-Y   R/F +-Z
             arrows pitch/roll  Q/E yaw   G base/tool frame
BOTH         TAB switch mode    - / = step size
             ENTER record       U undo     P arm/hold motion
             H help             Q quit     ESC EMERGENCY STOP
```

Gamepad: **hold LB** — it is a deadman, and nothing moves without it. `A` records, `B`
cancels, `X`/`Y` set the step size, D-pad up/down selects a joint, `BACK` switches mode.

You move the robot; the program watches and records. Move the **distal joints first**:

> Wrist 3 → Wrist 2 → Wrist 1 → Elbow → Shoulder → Base

Wrist motion buys orientation diversity for very little whole-arm travel. The elbow adds
translation. The shoulder sweeps the workspace. The base moves everything, so use it least
— but *do* use it if that is the only way to fill a gap in the geometry.

"Minimal robot movement" does **not** mean "thirty nearly identical poses". Hand-eye
calibration needs real translation and rotation spread — roughly several centimetres and
±5° to ±15°, wherever that is safe. The live diversity display tells you which axis is weak.

Board visibility always outranks the suggested joint order. If the grid drifts toward the
image edge, fix that first; if it goes invalid, recording is blocked until it recovers.

A record attempt is all-or-nothing. The board must detect, the arm must be verified
stationary across several consecutive velocity samples, the settle delay must elapse, and
the board must still be valid in the frame that is actually saved. If any check fails,
nothing is written and the waypoint count does not advance.

Use `--resume` to add to an existing set, or `--force` to archive it without being asked.

### STEPS 13–18 — Analyse, select, solve, validate

```bash
.venv/bin/python scripts/analyze_waypoints.py      --camera camera_1
.venv/bin/python scripts/select_best_waypoints.py  --camera camera_1 --count 20
.venv/bin/python scripts/solve_handeye.py          --camera camera_1 --selection best20
.venv/bin/python scripts/verify_calibration.py     --camera camera_1
```

The pipeline is:

```
30 valid observations
   -> preliminary hand-eye calibration using ALL 30
   -> per-waypoint reprojection + hand-eye residuals
   -> flag outliers (with a written reason; nothing is deleted)
   -> rank by a NORMALISED combined score
   -> greedily select 20 that are both low-error AND geometrically diverse
   -> final hand-eye calibration using only those 20
   -> validate on the 10 held out
```

Selection is deliberately *not* "sort by reprojection error, take the first 20" — that
reliably picks twenty nearly identical poses, which fit beautifully and generalise badly.
A candidate is accepted if it adds enough **translation OR rotation** diversity; requiring
both at once would reject genuinely useful poses.

The held-out 10 are for *checking*, never for fitting. If all-30 beats best-20 on that
independent set, the report says so plainly rather than assuming the selection helped.

`analyze_waypoints.py` cross-checks every OpenCV hand-eye method by default
(`--no-cross-check` to skip; `--method` to pick one). Agreement between Tsai, Park, Horaud,
Andreff and Daniilidis is evidence the data is sound; wide disagreement means it is not,
whatever the reprojection error says.

`solve_handeye.py --selection all` solves on the full set instead, for comparison.
`verify_calibration.py --live` additionally checks against the robot's current pose — it
reads the robot and never commands it; `--no-images` skips writing overlays.

### STEP 19 — Optional: replay the recorded waypoints

```bash
.venv/bin/python scripts/replay_waypoints.py --camera camera_1 --dry-run
```

Walks the recorded joint targets to confirm they are reachable and inside the safety
envelope. **A dry run by default** — it only commands the arm with `--enable-motion`, and
even then only if `safety.yaml` permits it. `--only N ...` limits it to specific waypoints,
`--pause` sets the dwell at each.

### STEPS 20–21 — Repeat for cameras 2 and 3

Identical process, from intrinsics onward — but re-rig the board first: `camera_2` and
`camera_3` are eye-to-hand, so the board moves from the table onto the robot flange.

The three datasets live in disjoint directory trees and no script ever reads one camera's
data while writing another's.

---

## Data layout

```
data/camera_1/
├── intrinsics/
│   ├── images/              raw intrinsic captures
│   ├── observations/        per-image detection metadata
│   └── result.yaml          K, distortion, resolution, error statistics
└── handeye/
    ├── images/              waypoint_001.png ... waypoint_030.png
    ├── observations/        waypoint_001.yaml ... (robot state + detection)
    ├── waypoints/
    │   ├── waypoints.yaml   master list of everything recorded
    │   └── initial_center.yaml   the session's calibration centre pose
    ├── selection/
    │   ├── waypoint_scores.csv   every metric for all 30
    │   ├── selected_20.yaml
    │   └── rejected_10.yaml      each with a written reason
    ├── sessions/<timestamp>/     archived earlier collections
    ├── preliminary_result_all_30.yaml
    ├── final_result_best_20.yaml
    └── verification/
        ├── verification_report.yaml
        └── waypoint_NNN_verify.png
```

`camera_2/` and `camera_3/` are identical and entirely separate.

**All 30 raw observations are kept forever.** Selection marks observations; it never
deletes images or metadata. Re-running a collection does not silently overwrite an existing
session — you are asked, or the old one is moved to `handeye/sessions/<timestamp>/`.

Captured images are gitignored (`data/**/images/*.png`); the directory skeleton and all
YAML metadata are tracked. Run logs go to `logs/`, one file per run, and are not tracked.

---

## Configuration files

| File | Contains |
|---|---|
| `config/cameras.yaml` | Which physical camera is `camera_1/2/3`, by serial; per-camera `handeye_mode`; capture settings |
| `config/calibration.yaml` | Board geometry, detection thresholds, collection targets, selection weights, hand-eye method |
| `config/safety.yaml` | Robot IP, motion interlocks, speeds, workspace box, joint limits, jog steps, gamepad deadman, preflight checks |

Four settings are `false` on purpose and must be set by a human who has checked the
corresponding physical reality:

* `calibration.yaml` → `apriltag_grid.verified_by_user` — you measured the board
* `calibration.yaml` → `handeye.verified_by_user` — `mode` matches your actual setup
* `safety.yaml` → `workspace.verified_by_user` — the box is right for your cell
* `safety.yaml` → `joint_limits.verified_by_user` — the limits are right for your cell

The workspace box and joint limits that ship here are conservative guesses, not measurements
of your cell. `gamepad.require_deadman` must never be set `false`.

---

## Conventions

A UR TCP pose is `[x, y, z, rx, ry, rz]` where `(rx, ry, rz)` is a **rotation vector**
(axis-angle): direction is the rotation axis, magnitude is the angle in radians. These are
**not** roll/pitch/yaw and must never be fed to an Euler-angle routine. Use
`calibration_utils.rotvec_to_matrix()` / `matrix_to_rotvec()`. This convention matches
OpenCV's `rvec`, so UR poses and `cv2.Rodrigues` interoperate directly.
`rotvec_to_rpy_deg()` exists for display only. Every stored pose repeats this in a
`rotation_representation` field, so a reader six months from now cannot mistake it.

The **board frame** is OpenCV's `cv2.aruco.GridBoard` frame: origin at the top-left corner
of the first tag, +X right along a row, +Y **down** along a column, +Z out of the printed
face away from the viewer.

Every waypoint stores the robot's **actual measured** state (`getActualQ()`,
`getActualTCPPose()`), never the commanded target, and only after the arm has been verified
stationary and allowed to settle.

All result files are written atomically, via a temporary file and `os.replace`. A
half-written observation is worse than a missing one. Images are written before their
metadata, because the metadata is the commit point every loader keys off.

---

## Source map

| Module | Responsibility |
|---|---|
| `calibration_utils.py` | Paths, atomic YAML/CSV I/O, rotation conversions, config loading |
| `camera_interface.py` | RealSense and v4l2 capture, warmup, flushing, serial resolution |
| `apriltag_detector.py` | Grid detection, PnP pose, per-frame quality rating |
| `intrinsic_calibration.py` | Intrinsic solve, error statistics, view-diversity tracking |
| `robot_interface.py` | RTDE, with reading and commanding hard-separated |
| `safety.py` | The envelope: interlocks, workspace box, joint limits, preflight |
| `jog_controller.py` | Step-size and frame state for a single jog command |
| `input_devices.py` | Keyboard and gamepad → jog commands; the gamepad deadman |
| `collection_session.py` | The interactive loop shared by the three teleop scripts |
| `session_setup.py` | Startup contract: connect, check, confirm, arm |
| `waypoint_recorder.py` | The all-or-nothing record procedure and the on-disk format |
| `pose_diversity.py` | Live diversity analysis and the next-joint suggestion |
| `waypoint_quality.py` | Per-waypoint scoring, outlier flagging, best-N selection |
| `handeye_calibration.py` | The hand-eye solve, method cross-check, hold-out validation |
| `ui_overlay.py` | The preview overlay drawing helpers |

---

## Relationship to `~/leader_arm`

That project is the existing, safety-reviewed UR teleoperation code on this machine
(Dynamixel leader arm → UR follower over `ur_rtde`, URSim only). This project **reuses its
approach** — same `ur_rtde` library, same state-read calls, same discipline of stopping the
robot in a `finally` block — but is a separate implementation, because calibration needs
discrete low-speed jogging rather than 500 Hz `servoJ` streaming.

**Nothing in `~/leader_arm` is modified or replaced by this project.**
