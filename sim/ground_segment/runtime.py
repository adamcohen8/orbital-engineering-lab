"""Ground knowledge and a separate simulation-to-measurement adapter."""

import heapq
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from sim.core.models import StateBelief
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.ground_stations import (
    _ground_measurement_physical_validity,
    evaluate_ground_station_access,
    evaluate_ground_station_measurements,
)

from .config import normalize_config
from .estimation import GroundOrbitEstimator


class GroundSegment:
    """Receive packets and predict ground knowledge without access to simulation truth.

    Packet times are seconds from the scenario epoch. Tracking rows use the
    existing station measurement schema; telemetry is a mapping of measured
    resource channels. advance() accepts only already-received packets.
    """

    def __init__(
        self, config, *, stations, object_ids, mu_km3_s2=EARTH_MU_KM3_S2, jd_utc_start=None, frame_context=None
    ):
        self.config = normalize_config(config)
        self.stations = {s.id: asdict(s) for s in stations}
        self.objects = set(object_ids)
        if set(self.config["priors"]) - self.objects:
            raise ValueError("ground prior references unknown object")
        self.estimator = GroundOrbitEstimator(
            mu_km3_s2=mu_km3_s2,
            process_noise_diag=self.config["process_noise_diag"],
            prediction_step_s=self.config["prediction_step_s"],
            jd_utc_start=jd_utc_start,
            frame_context=frame_context,
        )
        self.posterior = {
            oid: StateBelief(
                np.array(p["state"], dtype=float), np.array(p["covariance"], dtype=float), float(p["epoch_s"])
            )
            for oid, p in self.config["priors"].items()
        }
        self.predicted = {}
        self.last_tracking = {}
        self.tracking = {}
        self.last_assimilated = {}
        self.telemetry = {}
        self.time_s = -np.inf
        self.history = []

    def _validate_packet(self, packet, now):
        required = {"station_id", "object_id", "time_s", "received_time_s", "kind", "data"}
        if set(packet) != required:
            raise ValueError("ground packet fields do not match the packet contract")
        if packet["station_id"] not in self.stations or packet["object_id"] not in self.objects:
            raise ValueError("ground packet references unknown station or object")
        t, received = float(packet["time_s"]), float(packet["received_time_s"])
        if not np.isfinite(t) or not np.isfinite(received) or not 0 <= t <= received <= now:
            raise ValueError("ground packet requires 0 <= measurement time <= reception time <= current time")
        d = packet["data"]
        if not isinstance(d, dict):
            raise ValueError("ground packet data must be a mapping")
        if packet["kind"] == "tracking":
            if set(d) != {"time_s", "components", "vector", "sigma"} or d["time_s"] != t:
                raise ValueError("ground tracking payload must contain timestamp, components, vector and sigma")
            c = d["components"]
            if (
                not isinstance(c, list)
                or not c
                or len(set(c)) != len(c)
                or set(c) - {"azimuth_deg", "elevation_deg", "range_km", "range_rate_km_s"}
            ):
                raise ValueError("unsupported ground tracking components")
            z, sig = np.array(d["vector"], dtype=float), np.array(d["sigma"], dtype=float)
            if (
                z.shape != (len(c),)
                or sig.shape != z.shape
                or not np.all(np.isfinite(z))
                or not np.all(np.isfinite(sig))
                or np.any(sig < 0)
            ):
                raise ValueError("invalid ground tracking values or sigmas")
            physically_valid, validity_reason = _ground_measurement_physical_validity(
                {name: value for name, value in zip(c, z.tolist(), strict=True)}
            )
            if not physically_valid:
                raise ValueError(
                    f"invalid ground tracking physical domain ({validity_reason})"
                )
        elif packet["kind"] == "telemetry":
            if any(
                not isinstance(k, str) or not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v)
                for k, v in d.items()
            ):
                raise ValueError("ground telemetry requires finite scalar channels")
        else:
            raise ValueError("unknown ground packet kind")

    def advance(self, time_s, *, packets=(), contacts=None):
        now = float(time_s)
        if not np.isfinite(now) or now < 0 or now <= self.time_s:
            raise ValueError("ground time must be finite, nonnegative and strictly increasing")
        packets = deepcopy(list(packets))
        for p in packets:
            self._validate_packet(p, now)
        audits = []
        received_channels = set()
        for p in sorted(packets, key=lambda p: (p["received_time_s"], p["time_s"], p["station_id"])):
            oid, epoch = p["object_id"], p["time_s"]
            status = "accepted"
            if p["kind"] == "tracking":
                belief = self.posterior.get(oid)
                previous = self.last_tracking.get((oid, p["station_id"]), -np.inf)
                if epoch <= previous:
                    status = "duplicate_or_out_of_order"
                elif belief is None:
                    status = "no_prior"
                    self.last_tracking[(oid, p["station_id"])] = epoch
                elif epoch < belief.last_update_t_s:
                    status = "older_than_posterior"
                else:
                    self.posterior[oid] = self.estimator.update(belief, p["data"], self.stations[p["station_id"]])
                    self.predicted.pop(oid, None)
                    self.last_assimilated[oid] = {
                        "measurement_time_s": epoch,
                        "received_time_s": p["received_time_s"],
                        "station_id": p["station_id"],
                    }
                    self.last_tracking[(oid, p["station_id"])] = epoch
            else:
                held = self.telemetry.setdefault(oid, {})
                accepted = False
                for channel, value in p["data"].items():
                    if channel not in held or epoch > held[channel]["measurement_time_s"]:
                        held[channel] = {
                            "value": value,
                            "measurement_time_s": epoch,
                            "received_time_s": p["received_time_s"],
                            "station_id": p["station_id"],
                        }
                        received_channels.add((oid, channel))
                        accepted = True
                if not accepted:
                    status = "duplicate_or_out_of_order"
            if p["kind"] == "tracking" and status in {"accepted", "no_prior"}:
                self.tracking.setdefault(oid, {})[p["station_id"]] = {
                    "measurement_time_s": epoch,
                    "received_time_s": p["received_time_s"],
                    "data": deepcopy(p["data"]),
                    "disposition": status,
                }
            audits.append({**p, "disposition": status})
        states = {}
        for oid in sorted(self.objects):
            orbit = None
            if oid in self.posterior:
                b = self.estimator.predict(self.predicted.get(oid, self.posterior[oid]), now)
                self.predicted[oid] = b
                epoch = self.posterior[oid].last_update_t_s
                orbit = {
                    "state": b.state.tolist(),
                    "covariance": b.covariance.tolist(),
                    "epoch_s": now,
                    "posterior_epoch_s": epoch,
                    "age_s": now - epoch,
                    "status": "estimated" if now == epoch else "predicted",
                    "stale": now - epoch > self.config["stale_after_s"],
                    "frame": "eci",
                    "units": "km, km/s",
                    "prediction_model": "two_body",
                }
            telemetry = deepcopy(self.telemetry.get(oid, {}))
            for channel, row in telemetry.items():
                row.update(
                    age_s=now - row["measurement_time_s"],
                    stale=now - row["measurement_time_s"] > self.config["stale_after_s"],
                    status="received" if (oid, channel) in received_channels else "held",
                )
            if orbit is not None:
                orbit["source"] = "tracking" if oid in self.last_assimilated else "configured_prior"
                orbit["last_measurement"] = deepcopy(self.last_assimilated.get(oid))
            tracking = deepcopy(self.tracking.get(oid, {}))
            for row in tracking.values():
                row.update(
                    age_s=now - row["measurement_time_s"],
                    stale=now - row["measurement_time_s"] > self.config["stale_after_s"],
                )
            states[oid] = {"orbit": orbit, "telemetry": telemetry, "tracking": tracking}
        snapshot = {"time_s": now, "contacts": deepcopy(contacts or {}), "objects": states, "packets": audits}
        self.time_s = now
        self.history.append(snapshot)
        return deepcopy(snapshot)

    def snapshot(self, time_s=None):
        if not self.history:
            return {}
        if time_s is None:
            return deepcopy(self.history[-1])
        # Histories have one row per simulation sample, including between receptions.
        from bisect import bisect_left

        i = bisect_left(self.history, time_s - 1e-9, key=lambda row: row["time_s"])
        return (
            deepcopy(self.history[i])
            if i < len(self.history) and abs(self.history[i]["time_s"] - time_s) < 1e-9
            else {}
        )

    def retain_from(self, time_s):
        self.history = [r for r in self.history if r["time_s"] >= time_s]


class SimulatedGroundSegment:
    """Truth-facing sensor/contact adapter; the receiver sees only delivered data.

    Access is geometric; service outages optionally suppress tracking/downlink/
    uplink independently. Latency applies after acquisition, so in-flight data
    can arrive after loss of contact. There is no onboard store-and-forward.
    """

    CHANNELS = (
        "temperature_k",
        "battery_soc",
        "battery_energy_wh",
        "solar_generation_w",
        "load_demand_w",
        "load_served_w",
        "unmet_load_w",
        "battery_charge_w",
        "battery_discharge_w",
    )

    def __init__(
        self, config, *, stations, object_ids, jd_utc_start=None, frame_context=None, mu_km3_s2=EARTH_MU_KM3_S2
    ):
        self.ground = GroundSegment(
            config,
            stations=stations,
            object_ids=object_ids,
            jd_utc_start=jd_utc_start,
            frame_context=frame_context,
            mu_km3_s2=mu_km3_s2,
        )
        self.stations = stations
        self.jd = jd_utc_start
        self.frames = frame_context
        self.streams = {}
        self.pending = []
        self.sequence = 0
        self.last_telemetry = -np.inf

    def _queue(self, sid, oid, kind, time_s, data):
        arrival = time_s + self.ground.config["latency_s"]
        packet = dict(station_id=sid, object_id=oid, kind=kind, time_s=time_s, received_time_s=arrival, data=data)
        heapq.heappush(self.pending, (arrival, self.sequence, packet))
        self.sequence += 1

    def step(self, time_s, *, truth, resources=None):
        if time_s <= self.ground.time_s:
            raise ValueError("ground simulation time must increase")
        histories = {
            oid: np.asarray(state, dtype=float).reshape(1, -1)
            for oid, state in truth.items()
            if oid in self.ground.objects and np.all(np.isfinite(np.asarray(state)[:6]))
        }
        access, _ = evaluate_ground_station_access(
            ground_stations=self.stations,
            t_s=np.array([time_s]),
            truth_hist=histories,
            jd_utc_start=self.jd,
            frame_context=self.frames,
        )
        contacts = {}
        tracking_stations = []
        for station in self.stations:
            disabled = set()
            for outage in self.ground.config["outages"]:
                if outage["station_id"] == station.id and outage["start_s"] <= time_s < outage["end_s"]:
                    disabled.update(outage["services"])
            targets = {}
            for oid in sorted(self.ground.objects):
                row = access.get(station.id, {}).get("targets", {}).get(oid)
                visible = bool(row and row["access"][0])
                targets[oid] = {s: visible and s not in disabled for s in ("tracking", "downlink", "uplink")}
                targets[oid]["access_reason"] = row["reason"][0] if row else "unavailable"
                targets[oid]["disabled_services"] = sorted(disabled)
            contacts[station.id] = targets
            if "tracking" not in disabled:
                tracking_stations.append(station)
        measured = evaluate_ground_station_measurements(
            ground_stations=tracking_stations,
            t_s=np.array([time_s]),
            truth_hist=histories,
            jd_utc_start=self.jd,
            frame_context=self.frames,
            stream_state=self.streams,
        )
        for sid, station in measured.items():
            for oid, target in station["targets"].items():
                for row in target["measurements"]:
                    # The producer retains every raw Gaussian draw for audit
                    # evidence, but only an explicitly usable row may enter
                    # the receiver packet queue.
                    if row.get("valid", True) is not True:
                        continue
                    self._queue(
                        sid, oid, "tracking", time_s, {k: row[k] for k in ("time_s", "components", "vector", "sigma")}
                    )
        if time_s - self.last_telemetry >= self.ground.config["cadence_s"] - 1e-9:
            for sid, targets in contacts.items():
                for oid, services in targets.items():
                    if services["downlink"]:
                        values = (resources or {}).get(oid, {})
                        data = {
                            key: float(values[key])
                            for key in self.CHANNELS
                            if key in values and np.isfinite(values[key])
                        }
                        if data:
                            self._queue(sid, oid, "telemetry", time_s, data)
            self.last_telemetry = time_s
        packets = []
        while self.pending and self.pending[0][0] <= time_s:
            packets.append(heapq.heappop(self.pending)[2])
        return self.ground.advance(time_s, packets=packets, contacts=contacts)
