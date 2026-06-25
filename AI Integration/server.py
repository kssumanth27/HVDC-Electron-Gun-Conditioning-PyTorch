
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Dict, List

from p4p.nt import NTScalar
from p4p.server import Server
from p4p.server.thread import SharedPV


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# All server PV values will be posted as doubles ("d").
# This makes the server side consistent and avoids type mismatches.
NT_DOUBLE = NTScalar("d")

# These are the three diagnostic PVs that the server owns.
# The server will read these CSV columns and publish one row at a time.
PV_CONFIG = {
    "example:current": {
        "file": r"C:\Users\skantamne\Downloads\PhD EGun\Data\EPICS_test\processed_polgun_voltage_11042020_hz_combined_uptoMax_plus10.csv",
        "column": "GunCurrent.Avg",
    },
    "example:pressure": {
        "file": r"C:\Users\skantamne\Downloads\PhD EGun\Data\EPICS_test\processed_polgun_voltage_11042020_hz_combined_uptoMax_plus10.csv",
        "column": "peg-BL-cc:pressureM",
    },
    "example:radiation": {
        "file": r"C:\Users\skantamne\Downloads\PhD EGun\Data\EPICS_test\processed_polgun_voltage_11042020_hz_combined_uptoMax_plus10.csv",
        "column": "RadiationTotal",
    },
}

# This PV is only a synchronization helper.
# It lets the client know which diagnostic row the server has just published.
ROW_PV_NAME = "example:row"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_values(raw_path: str, column_name: str) -> list[float]:
    """
    Read one numeric column from a CSV file and return it as a list of floats.

    The function:
    - cleans up the path string
    - verifies the file exists
    - verifies the header exists
    - verifies the requested column exists
    - converts every non-empty cell to float
    """
    path = Path(str(raw_path).strip().strip('"').strip("'"))

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    values: list[float] = []

    # Use utf-8-sig so files with BOM still parse correctly.
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")

        # Strip spaces from header names so accidental whitespace does not break lookup.
        headers = [h.strip() if h is not None else h for h in reader.fieldnames]
        reader.fieldnames = headers

        if column_name not in headers:
            raise ValueError(
                f"Column '{column_name}' not found in {path}. "
                f"Available columns: {headers}"
            )

        for row_num, row in enumerate(reader, start=2):
            # Read the cell, remove whitespace, and reject blanks.
            cell = row.get(column_name, "")
            cell = "" if cell is None else str(cell).strip()

            if cell == "":
                raise ValueError(
                    f"{path}: empty cell in column '{column_name}' at row {row_num}"
                )

            values.append(float(cell))

    if not values:
        raise ValueError(f"No numeric values found in column '{column_name}' of {path}")

    return values


# ---------------------------------------------------------------------------
# SharedPV helpers
# ---------------------------------------------------------------------------

def make_scalar_pv(initial_value: float) -> SharedPV:
    """
    Create a writable/readable scalar PV with an initial numeric value.
    """
    return SharedPV(
        nt=NT_DOUBLE,
        initial=NT_DOUBLE.wrap(initial_value, timestamp=time.time()),
    )


class VoltageAdvanceHandler:
    """
    Handle client writes to example:voltage.

    Every time the client puts a new voltage:
    1. Store that voltage.
    2. Publish the voltage PV.
    3. Publish the next diagnostic row from the CSV files.
    4. Advance the server row index by exactly one.

    This is the key change that removes autonomous server timing.
    """

    def __init__(self, state: dict, pvs: dict[str, SharedPV], series: dict[str, List[float]]):
        self.state = state
        self.pvs = pvs
        self.series = series

    def put(self, pv, op):
        """
        Called by P4P whenever a client performs ctxt.put("example:voltage", value).
        """
        try:
            # op.value() is the new value the client requested.
            new_voltage = float(op.value())
        except Exception as exc:
            # Reject the write if the value cannot be interpreted as a float.
            op.done(error=f"Invalid voltage value: {exc}")
            return

        # Protect the shared row counter and shared voltage state.
        with self.state["lock"]:
            row_index = self.state["row_index"]

            # Stop cleanly if the CSV has no more rows.
            if row_index >= self.state["row_count"]:
                op.done(error="End of diagnostic CSV reached")
                return

            # One timestamp for the entire update batch keeps the monitors aligned.
            ts = time.time()

            # Store the latest voltage so the server can report it.
            self.state["voltage"] = new_voltage

            # Publish the voltage PV so the client monitor sees the new setpoint.
            pv.post(NT_DOUBLE.wrap(new_voltage, timestamp=ts))

            # Publish the diagnostics for the current row.
            current_value = self.series["example:current"][row_index]
            pressure_value = self.series["example:pressure"][row_index]
            radiation_value = self.series["example:radiation"][row_index]

            self.pvs["example:current"].post(NT_DOUBLE.wrap(current_value, timestamp=ts))
            self.pvs["example:pressure"].post(NT_DOUBLE.wrap(pressure_value, timestamp=ts))
            self.pvs["example:radiation"].post(NT_DOUBLE.wrap(radiation_value, timestamp=ts))

            



            # Debug output
            print()
            print("=" * 60)
            print(f"SERVER STEP {row_index}")
            print(f"Voltage PUT received : {new_voltage}")
            print(f"Current published    : {current_value}")
            print(f"Pressure published   : {pressure_value}")
            print(f"Radiation published  : {radiation_value}")
            print("=" * 60)
            print()

            # Advance the server row counter by exactly one.
            self.state["row_index"] = row_index + 1

            # Publish the row number as a helper PV so the client knows the update is complete.
            # This is only for synchronization and debugging.
            self.pvs[ROW_PV_NAME].post(NT_DOUBLE.wrap(float(self.state["row_index"]), timestamp=ts))

        # Tell P4P that the write completed successfully.
        op.done()


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    # Load the three diagnostic CSV series into memory.
    series = {
        pv_name: load_values(info["file"], info["column"])
        for pv_name, info in PV_CONFIG.items()
    }

    # Check that the three diagnostic files line up row-for-row.
    lengths = {pv_name: len(values) for pv_name, values in series.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"CSV files must have the same number of rows. Got: {lengths}")

    row_count = next(iter(lengths.values()))

    # Shared state used by the voltage put handler.
    state = {
        "lock": threading.Lock(),   # protects row_index and voltage
        "row_index": 0,             # next diagnostic row to publish
        "row_count": row_count,     # total number of rows available
        "voltage": 0.0,             # latest voltage written by the client
    }

    # Build the PV dictionary first.
    # The handler is added to the writable voltage PV after the dict exists,
    # so the handler can post to the other PVs.
    pvs: dict[str, SharedPV] = {}

    # The diagnostics start with row 0 so the PVs exist immediately.
    pvs["example:current"] = make_scalar_pv(series["example:current"][0])
    pvs["example:pressure"] = make_scalar_pv(series["example:pressure"][0])
    pvs["example:radiation"] = make_scalar_pv(series["example:radiation"][0])

    # Helper PV for synchronization/debugging.
    pvs[ROW_PV_NAME] = make_scalar_pv(0.0)

    # Create the writable voltage PV.
    # This PV will call VoltageAdvanceHandler.put() every time the client writes to it.
    pvs["example:voltage"] = SharedPV(
        nt=NT_DOUBLE,
        initial=NT_DOUBLE.wrap(0.0, timestamp=time.time()),
        handler=VoltageAdvanceHandler(state, pvs, series),
    )

    print("PVA server starting.")
    print("Server behavior:")
    print("  - waits for client put() to example:voltage")
    print("  - publishes one diagnostic row per voltage put")
    print("  - advances only when the client requests the next voltage")
    print()

    # Start the server and block forever.
    Server.forever(providers=[pvs])


if __name__ == "__main__":
    main()
