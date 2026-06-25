from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from p4p.client.thread import Context


PRINT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# This CSV is used only by the client.
# The client reads a voltage value, sends it with put(), and waits for the server update.
VOLTAGE_CSV = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\polgun v8 until max conditioning\v8 spikes cleaned until max\test poster\processed_polgun_voltage_11042020_hz_combined_uptoMax_plus10.csv"
VOLTAGE_COLUMN = "glassmanDataXfer:hvPsVoltageMeasM"

# This helper PV is used to know when the server has finished publishing a row.
ROW_PV_NAME = "example:row"


# ---------------------------------------------------------------------------
# Live monitor CSV output
# ---------------------------------------------------------------------------

# Put your desired output folder path here.
# Example:
# OUTPUT_FOLDER = r"C:\Users\suman\Downloads\monitor_output"
OUTPUT_FOLDER = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\EPICS_testing\For_environment\CSV generated"

CSV_HEADERS = [
    "Server Time",
    "Voltage put",
    "Voltage monitor",
    "Current monitor",
    "Pressure Monitor",
    "Radiation Monitor",
]

CSV_LOCK = threading.Lock()
CSV_ROWS = {}
CSV_PATH = None


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
# Live monitor CSV helpers
# ---------------------------------------------------------------------------

def extract_scalar(value) -> float:
    """
    Extract the numeric scalar from a P4P monitor value.
    """
    try:
        return float(value["value"])
    except Exception:
        pass

    try:
        return float(getattr(value, "value"))
    except Exception:
        pass

    return float(value)


def extract_timestamp(value):
    """
    Extract the server-side PV timestamp from a P4P monitor value.
    """
    return getattr(value, "timestamp", None)


def format_timestamp(ts: float) -> str:
    """
    Use the same timestamp style already used in the code.
    """
    return str(datetime.fromtimestamp(ts))


def initialize_live_csv() -> None:
    """
    Create the live monitor CSV file.

    File name format:
    Monitor_data_(today's date)_(time first created).csv
    """
    global CSV_PATH

    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    created_time = datetime.now()
    filename = f"Monitor_data_{created_time:%Y-%m-%d}_{created_time:%H-%M-%S}.csv"
    CSV_PATH = output_folder / filename

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

    with PRINT_LOCK:
        print(f"Live monitor CSV created: {CSV_PATH}", flush=True)


def write_live_csv_snapshot() -> None:
    """
    Rewrite the CSV file with the latest monitor data.

    This allows rows to be updated as voltage/current/pressure/radiation
    monitor callbacks arrive asynchronously.
    """
    if CSV_PATH is None:
        return

    sorted_keys = sorted(CSV_ROWS.keys())

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

        for key in sorted_keys:
            writer.writerow(CSV_ROWS[key])


def update_live_csv(
    *,
    server_timestamp: float,
    monitor_name: str,
    monitor_value: float,
    sync_state: dict,
) -> None:
    """
    Update the live CSV using server timestamp as the row key.

    All monitor values with the same server post timestamp go into the same row.
    The CSV starts only from client step 2.
    """
    if server_timestamp is None:
        return

    with sync_state["lock"]:
        current_step = sync_state["current_step"]
        voltage_put = sync_state["current_voltage_put"]

    # Start CSV logging only from client step 2.
    if current_step < 2:
        return

    column_map = {
        "voltage": "Voltage monitor",
        "current": "Current monitor",
        "pressure": "Pressure Monitor",
        "radiation": "Radiation Monitor",
    }

    if monitor_name not in column_map:
        return

    server_time = format_timestamp(server_timestamp)

    # Use the raw server timestamp as the grouping key.
    # Same server post time = same CSV row.
    row_key = server_timestamp

    with CSV_LOCK:
        if row_key not in CSV_ROWS:
            CSV_ROWS[row_key] = {
                "Server Time": server_time,
                "Voltage put": voltage_put,
                "Voltage monitor": "",
                "Current monitor": "",
                "Pressure Monitor": "",
                "Radiation Monitor": "",
            }

        CSV_ROWS[row_key]["Voltage put"] = voltage_put
        CSV_ROWS[row_key][column_map[monitor_name]] = monitor_value

        write_live_csv_snapshot()


# ---------------------------------------------------------------------------
# Monitor callbacks
# ---------------------------------------------------------------------------

def make_print_callback(name: str, sync_state: dict) -> Callable:
    """
    Build a callback that prints each incoming monitor update cleanly
    and writes monitor values into the live CSV.

    PRINT_LOCK prevents different monitor callbacks from mixing their print lines.
    """
    def callback(value):
        monitor_receive_ts = time.time()
        raw_value = extract_scalar(value)
        pv_post_timestamp = extract_timestamp(value)

        update_live_csv(
            server_timestamp=pv_post_timestamp,
            monitor_name=name,
            monitor_value=raw_value,
            sync_state=sync_state,
        )

        with PRINT_LOCK:
            print()
            print("-" * 60)
            print(f"MONITOR RECEIVED {name}")
            print(f"{name} value              : {raw_value}")
            print(f"Monitor receive time   : {datetime.fromtimestamp(monitor_receive_ts)}")

            if pv_post_timestamp is not None:
                print(f"Server PV post time    : {datetime.fromtimestamp(pv_post_timestamp)}")
                print(f"Approx monitor latency : {monitor_receive_ts - pv_post_timestamp:.6f} seconds")

            print("-" * 60, flush=True)

    return callback


def make_row_callback(sync_state: dict, step_event: threading.Event) -> Callable:
    """
    Build a callback that watches example:row and wakes the client up
    when the server has published the row we were waiting for.
    """
    def callback(value):
        monitor_receive_ts = time.time()
        row_value = int(extract_scalar(value))
        pv_post_timestamp = extract_timestamp(value)

        with PRINT_LOCK:
            print()
            print("-" * 60)
            print(f"MONITOR RECEIVED {ROW_PV_NAME}")
            print(f"{ROW_PV_NAME} value       : {row_value}")
            print(f"Monitor receive time   : {datetime.fromtimestamp(monitor_receive_ts)}")

            if pv_post_timestamp is not None:
                print(f"Server PV post time    : {datetime.fromtimestamp(pv_post_timestamp)}")
                print(f"Approx monitor latency : {monitor_receive_ts - pv_post_timestamp:.6f} seconds")

            print("-" * 60, flush=True)

        with sync_state["lock"]:
            expected_row = sync_state["expected_row"]

        if expected_row > 0 and row_value >= expected_row:
            step_event.set()

    return callback


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    # Load the voltage sequence that the client will send to the server.
    voltages = load_voltage_values(VOLTAGE_CSV, VOLTAGE_COLUMN)

    # Create the live monitor CSV file.
    initialize_live_csv()

    # Create a PVA client context.
    ctxt = Context("pva")

    # This event is used to block until the server finishes publishing each step.
    step_event = threading.Event()

    # Shared state used by the row callback, monitor callbacks, and the main loop.
    sync_state = {
        "lock": threading.Lock(),
        "expected_row": 0,          # the row we are waiting for right now
        "current_step": 0,          # the current client step
        "current_voltage_put": "",  # the voltage sent during the current client step
    }

    # Create monitors for the three diagnostics, the voltage PV, and the row PV.
    # The print callbacks are for visibility in the console.
    monitors = [
        ctxt.monitor("example:current", make_print_callback("current", sync_state)),
        ctxt.monitor("example:pressure", make_print_callback("pressure", sync_state)),
        ctxt.monitor("example:radiation", make_print_callback("radiation", sync_state)),
        ctxt.monitor("example:voltage", make_print_callback("voltage", sync_state)),
        ctxt.monitor(ROW_PV_NAME, make_row_callback(sync_state, step_event)),
    ]

    print("Client monitors are active.")
    print("The client will:")
    print("  1. put a voltage")
    print("  2. wait until the server publishes the matching diagnostic row")
    print("  3. move to the next voltage")
    print()

    for step, voltage in enumerate(voltages, start=1):
        with sync_state["lock"]:
            sync_state["expected_row"] = step
            sync_state["current_step"] = step
            sync_state["current_voltage_put"] = voltage

        step_event.clear()

        client_put_ts = time.time()

        with PRINT_LOCK:
            print()
            print("=" * 60)
            print(f"CLIENT STEP {step}")
            print(f"Client PUT start time : {datetime.fromtimestamp(client_put_ts)}")
            print(f"PUT voltage           : {voltage}")
            print("=" * 60, flush=True)

        ctxt.put("example:voltage", voltage)

        client_put_done_ts = time.time()

        with PRINT_LOCK:
            print(f"Client PUT done time  : {datetime.fromtimestamp(client_put_done_ts)}")
            print(f"PUT call duration     : {client_put_done_ts - client_put_ts:.6f} seconds", flush=True)

        if not step_event.wait(timeout=10.0):
            raise TimeoutError(
                f"Timed out waiting for server row {step} after putting voltage {voltage}"
            )

        print(f"Step {step} completed.")
        print("=====", flush=True)

        time.sleep(2)

    input("Press Enter to quit...")


if __name__ == "__main__":
    main()
