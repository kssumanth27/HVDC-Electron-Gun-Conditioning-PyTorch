from __future__ import annotations

import csv
import threading
import time
from pathlib import Path


from p4p.nt import NTScalar
from p4p.server import Server
from p4p.server.thread import SharedPV


UPDATE_SECONDS = 0.1

PV_CONFIG = {
    "example:current": {
        "file": r"c:\Users\skantamne\Downloads\PhD EGun\Data\testing\processed_polgun_voltage_11042020_hz_combined_uptoMax_plus10.csv",
        "column": "GunCurrent.Avg",
    },
    "example:pressure": {
        "file": r"c:\Users\skantamne\Downloads\PhD EGun\Data\testing\processed_polgun_voltage_11042020_hz_combined_uptoMax_plus10.csv",
        "column": "peg-BL-cc:pressureM",
    },
    "example:radiation": {
        "file": r"c:\Users\skantamne\Downloads\PhD EGun\Data\testing\processed_polgun_voltage_11042020_hz_combined_uptoMax_plus10.csv",
        "column":"RadiationTotal",
    },
    "example:voltage": {
        "file": r"c:\Users\skantamne\Downloads\PhD EGun\Data\testing\processed_polgun_voltage_11042020_hz_combined_uptoMax_plus10.csv",
        "column": "glassmanDataXfer:hvPsVoltageMeasM",
    },
}



from pathlib import Path
import csv

def load_values(raw_path, column_name: str) -> list[float]:
    path = Path(str(raw_path).strip().strip('"').strip("'"))

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    values: list[float] = []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")

        headers = [h.strip() if h is not None else h for h in reader.fieldnames]
        reader.fieldnames = headers

        print("\nFILE:", path)
        print("HEADERS:", headers)
        print("COLUMN REQUESTED:", column_name)

        if column_name not in headers:
            raise ValueError(
                f"Column '{column_name}' not found in {path}. "
                f"Available columns: {headers}"
            )

        for row_num, row in enumerate(reader, start=2):  # row 1 is header
            cell = row.get(column_name, "")
            cell = "" if cell is None else str(cell).strip()

            if cell == "":
                raise ValueError(f"{path}: empty cell in column '{column_name}' at row {row_num}")

            try:
                values.append(float(cell))
            except ValueError:
                raise ValueError(
                    f"{path}: non-numeric value in column '{column_name}' at row {row_num}: {cell!r}"
                )

    print(f"Parsed {len(values)} numeric rows from {path}")
    return values


def make_pv(initial_value: float) -> SharedPV:
    nt = NTScalar("d")
    return SharedPV(
        nt=nt,
        initial=nt.wrap(initial_value, timestamp=time.time()),
    )


def start_server(pvs: dict[str, SharedPV]) -> None:
    Server.forever(providers=[pvs])


def main() -> None:
    series = {
        pv_name: load_values(info["file"], info["column"])
        for pv_name, info in PV_CONFIG.items()
    }

    lengths = {pv_name: len(values) for pv_name, values in series.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"CSV files must have the same number of rows. Got: {lengths}")
    
#     lengths = {pv_name: len(values) for pv_name, values in series.items()}
# if len(set(lengths.values())) != 1:
#     raise ValueError(f"CSV files must have the same number of rows. Got: {lengths}")

    pvs = {
        pv_name: make_pv(values[0])
        for pv_name, values in series.items()
    }

    threading.Thread(target=start_server, args=(pvs,), daemon=True).start()

    print("PVA server started.")
    for pv_name, info in PV_CONFIG.items():
        print(f"{pv_name} <- {info['file']} [{info['column']}]")
    print(f"Update rate: every {UPDATE_SECONDS} seconds")
    print()

    row_count = lengths[next(iter(lengths))]
    for idx in range(row_count):
        ts = time.time()

        for pv_name, pv in pvs.items():
            value = series[pv_name][idx]
            pv.post(value, timestamp=ts)
            print(f"{pv_name}: {value}")

        print("---")
        time.sleep(UPDATE_SECONDS)

    print("Done — reached the end of the CSV files. Stop the program.")


if __name__ == "__main__":
    main()