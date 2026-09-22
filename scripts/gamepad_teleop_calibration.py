#!/usr/bin/env python3
"""Xbox/gamepad jog of the UR12e while watching the AprilTag board.

    python scripts/gamepad_teleop_calibration.py --camera camera_1
    python scripts/gamepad_teleop_calibration.py --probe

LB is a DEADMAN: it must be held continuously for the robot to move, and a
stale or disconnected controller reads as released.

Controller button and axis indices vary between drivers. Run --probe, press
each control, and correct config/safety.yaml -> gamepad.mapping to match.
"""
import argparse
import sys
import time

import _bootstrap  # noqa: F401
import _session_main as session_main


def probe() -> int:
    """Print live axis/button state so the mapping can be corrected."""
    from input_devices import GamepadInput
    from jog_controller import JogState
    from safety import load_envelope

    envelope = load_envelope()
    try:
        device = GamepadInput(JogState(envelope.config), envelope.gamepad).open()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Gamepad: {device.describe()}")
    print()
    print("Press each control and note the index that changes.")
    print("Put those numbers into config/safety.yaml -> gamepad.mapping.")
    print("Ctrl+C to stop.")
    print()
    try:
        while True:
            lines = device.probe_lines()
            print("\033[K" + lines[0])
            print("\033[K" + lines[1])
            print("\033[K" + lines[2])
            print("\033[3A", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n" * 3 + "Stopped.")
    finally:
        device.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Gamepad teleoperation for calibration, with live "
                    "AprilTag preview and a mandatory deadman.")
    parser.add_argument("--probe", action="store_true",
                        help="show live axis/button indices and exit "
                             "(no camera, no robot)")
    session_main.add_common_arguments(parser)
    parser.add_argument("--no-record", action="store_true",
                        help="framing only; disable waypoint recording")

    # --probe must work without --camera, which is otherwise required.
    if argv is None:
        argv = sys.argv[1:]
    if "--probe" in argv:
        return probe()

    args = parser.parse_args(argv)
    return session_main.run(args, "GAMEPAD TELEOP", "gamepad",
                            allow_recording=not args.no_record)


if __name__ == "__main__":
    sys.exit(main())
