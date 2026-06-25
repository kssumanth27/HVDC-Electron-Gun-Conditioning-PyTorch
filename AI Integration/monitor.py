

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from p4p.client.thread import Context


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# This CSV is used only by the client.
# The client reads a voltage value, sends it with put(), and waits for the server update.
VOLTAGE_CSV = r"C:\Users\skantamne\Downloads\PhD EGun\Data\EPICS_test\processed_polgun_voltage_11042020_hz_combined_uptoMax_plus10.csv"
VOLTAGE_COLUMN = "glassmanDataXfer:hvPsVoltageMeasM"

# This helper PV is used to know when the server has finished publishing a row.
ROW_PV_NAME = "example:row"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_voltage_values(raw_path: str, column_name: str) -> list[float]:
    """
    Read the voltage CSV column and return all usable values as floats.
    Blank cells are skipped.
    """
    path = Path(str(raw_path).strip().strip('"').strip("'"))
    values: list[float] = []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")

        headers = [h.strip() if h is not None else h for h in reader.fieldnames]
        reader.fieldnames = headers

        if column_name not in headers:
            raise ValueError(
                f"Column '{column_name}' not found in {path}. Available: {headers}"
            )

        for row in reader:
            cell = row.get(column_name, "")
            cell = "" if cell is None else str(cell).strip()

            # Skip blanks so the client can still use a partially clean file.
            if cell:
                values.append(float(cell))

    if not values:
        raise ValueError(f"No numeric values found in column '{column_name}' of {path}")

    return values


# ---------------------------------------------------------------------------
# Monitor callbacks
# ---------------------------------------------------------------------------

def make_print_callback(name: str) -> Callable:
    """
    Build a simple callback that prints each incoming monitor update.
    """
    def callback(value):
        raw_value = getattr(value, "value", value)
        timestamp = getattr(value, "timestamp", None)

        print(f"{name}: {raw_value}")

        if timestamp is not None:
            print("Updated:", datetime.fromtimestamp(timestamp))

        print("---")

    return callback


def make_row_callback(sync_state: dict, step_event: threading.Event) -> Callable:
    """
    Build a callback that watches example:row and wakes the client up
    when the server has published the row we were waiting for.
    """
    def callback(value):
        row_value = int(float(getattr(value, "value", value)))
        timestamp = getattr(value, "timestamp", None)

        print(f"{ROW_PV_NAME}: {row_value}")

        if timestamp is not None:
            print("Updated:", datetime.fromtimestamp(timestamp))

        print("---")

        with sync_state["lock"]:
            expected_row = sync_state["expected_row"]

        # Only release the waiting client when the server has reached the row
        # that matches the most recent voltage put.
        if expected_row > 0 and row_value >= expected_row:
            step_event.set()

    return callback


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    # Load the voltage sequence that the client will send to the server.
    voltages = load_voltage_values(VOLTAGE_CSV, VOLTAGE_COLUMN)

    # Create a PVA client context.
    ctxt = Context("pva")

    # This event is used to block until the server finishes publishing each step.
    step_event = threading.Event()

    # Shared state used by the row callback and the main loop.
    sync_state = {
        "lock": threading.Lock(),
        "expected_row": 0,   # the row we are waiting for right now
    }

    # Create monitors for the three diagnostics, the voltage PV, and the row PV.
    # The print callbacks are for visibility in the console.
    monitors = [
        ctxt.monitor("example:current", make_print_callback("current")),
        ctxt.monitor("example:pressure", make_print_callback("pressure")),
        ctxt.monitor("example:radiation", make_print_callback("radiation")),
        ctxt.monitor("example:voltage", make_print_callback("voltage")),
        ctxt.monitor(ROW_PV_NAME, make_row_callback(sync_state, step_event)),
    ]

    print("Client monitors are active.")
    print("The client will:")
    print("  1. put a voltage")
    print("  2. wait until the server publishes the matching diagnostic row")
    print("  3. move to the next voltage")
    print()

    for step, voltage in enumerate(voltages, start=1):
        # Tell the row callback which server row we expect next.
        with sync_state["lock"]:
            sync_state["expected_row"] = step

        # Clear the event before sending the next voltage.
        step_event.clear()

        # Send the new voltage to the server.
        ctxt.put("example:voltage", voltage)
        print(f"PUT voltage step {step} -> {voltage}")

        # Wait until the server publishes the matching diagnostic row.
        # This is the synchronization point that prevents the client from running ahead.
        if not step_event.wait(timeout=10.0):
            raise TimeoutError(
                f"Timed out waiting for server row {step} after putting voltage {voltage}"
            )

        print(f"Step {step} completed.")
        print("=====")

        # A tiny pause is optional; it just makes console output easier to read.
        time.sleep(2)

    input("Press Enter to quit...")


if __name__ == "__main__":
    main()
