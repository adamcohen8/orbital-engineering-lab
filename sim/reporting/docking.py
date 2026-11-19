"""Write docking truth samples to the run review database."""

from __future__ import annotations

import json

_SCALARS = (
    "gap_m", "lateral_m", "angle_rad", "normal_speed_m_s", "speed_m_s", "rate_rad_s",
    "normal_load_n", "interface_force_n", "body_torque_n_m",
)
_VECTORS = (
    "offset_receiver_m", "rotation_error_eci_rad", "force_on_probe_eci_n", "site_loads_n",
    "probe_torque_about_com_eci_n_m", "receiver_torque_about_com_eci_n_m",
)


def insert_docking_samples(connection, samples):
    fields = [*(f"{name} REAL NOT NULL" for name in _SCALARS),
              *(f"{name}_json TEXT NOT NULL" for name in _VECTORS)]
    connection.execute(
        "CREATE TABLE docking_samples (time_s REAL PRIMARY KEY, phase TEXT NOT NULL, "
        + ", ".join(fields) + ")"
    )
    placeholders = ", ".join("?" for _ in range(2 + len(fields)))
    connection.executemany(
        f"INSERT INTO docking_samples VALUES ({placeholders})",
        ((row["time_s"], row["phase"], *(row[key] for key in _SCALARS),
          *(json.dumps(row[key], allow_nan=False) for key in _VECTORS)) for row in samples),
    )
