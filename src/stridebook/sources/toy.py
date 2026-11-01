"""Synthetic runs in Stridelab's shapes, for building the dashboard before integration.

The physiology is simple but has the effects the analyses look for:
- HR lags pace changes, rising faster than it falls.
- HR drifts up with time (more in the heat) and falls with fitness.
- Step length shrinks and vertical oscillation and stance time grow with fatigue on long runs.

Records come one per second, like a Coros watch. Laps follow the workout steps, or an autolap
every N metres on steady runs. Like Coros files, laps carry no warmup/active/rest labels;
deriving those is the analysis layer's job.
"""

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import pandas as pd

from stridebook.domain import BlockPace, ParsedRun, RunCategory, SegmentRole

W, A, R, C = SegmentRole.WARMUP, SegmentRole.ACTIVE, SegmentRole.REST, SegmentRole.COOLDOWN
S = SegmentRole.STEADY

# Model constants
HR_FLOOR = 62.0  # bpm the steady-state model gives at zero speed
HR_PER_SPEED = 26.0  # bpm per m/s at fitness 1.0
HR_CEILING = 192.0
TAU_HR_UP = 35.0  # s
TAU_HR_DOWN = 60.0  # s
TAU_HR_SLOW = 600.0  # s, slow component
HARD_SPEED = 3.2  # m/s at fitness 1.0; faster running builds the slow component
TAU_SPEED = 4.0  # s to settle into a new pace
RUNNING_SPEED = 1.8  # m/s; slower is walking
ROUTE_CENTER = (55.70, 12.57)  # arbitrary; the toy route is a loop around it
UTC_OFFSET = timedelta(hours=1)

# Demo workouts: name -> label for the import page
WORKOUTS = {
    "easy": "Easy 50 min",
    "intervals_5x6min": "Intervals 5 × 6 min",
    "intervals_6x1000m": "Intervals 6 × 1000 m",
    "tempo_2x15min": "Tempo 2 × 15 min",
    "tempo_25min": "Tempo 25 min",
    "long_run": "Long run 24 km",
    "long_run_mp": "Long run 24 km, 2 × 5 km at MP",
    "long_run_hmp": "Long run 24 km, 3 × 3 km at HMP",
    "race_10k": "Race 10 km",
    "race_half": "Race half marathon",
}


@dataclass(frozen=True)
class Step:
    role: SegmentRole  # what the step really is; not written to the laps
    speed: float  # target, m/s
    duration_s: float | None = None
    distance_m: float | None = None


@dataclass(frozen=True)
class Workout:
    name: str
    category: RunCategory
    steps: tuple[Step, ...]
    auto_lap_m: float | None = None  # a lap every N metres, like a watch's autolap
    fatigue: float = 0.0  # 0 = none, 1 = the end of a hard long run
    loop_m: float = 2500.0  # route perimeter
    hills_m: float = 12.0  # altitude amplitude
    block_pace: BlockPace | None = None  # what a long run's pace blocks are; not in the file


# Long-run pace blocks: name -> (blocks, km each, speed at fitness 1.0, pace)
LONG_RUN_BLOCKS = {
    "long_run_mp": (2, 5, 3.62, BlockPace.MARATHON),
    "long_run_hmp": (3, 3, 3.85, BlockPace.HALF_MARATHON),
}
LONG_RUN_SPEED = 3.15  # m/s at fitness 1.0


def workout(
    name: str,
    fitness: float = 1.0,
    *,
    minutes: float = 50,
    distance_km: float = 24,
    burn_in_km: int = 3,
) -> Workout:
    """A workout from the catalogue. Fitness scales every target speed (1.0 = baseline)."""
    f = fitness
    warmup = Step(W, 2.95 * f, duration_s=15 * 60)
    cooldown = Step(C, 2.85 * f, duration_s=10 * 60)
    match name:
        case "easy":
            return Workout(
                name, RunCategory.EASY, (Step(A, 3.0 * f, minutes * 60),), auto_lap_m=1000
            )
        case "intervals_5x6min":
            reps = _reps(5, Step(A, 4.05 * f, duration_s=360), Step(R, 1.3, duration_s=90))
            return Workout(name, RunCategory.INTERVAL, (warmup, *reps, cooldown))
        case "intervals_6x1000m":
            reps = _reps(6, Step(A, 4.3 * f, distance_m=1000), Step(R, 0.6, duration_s=150))
            return Workout(
                name, RunCategory.INTERVAL, (warmup, *reps, cooldown), loop_m=400, hills_m=0
            )
        case "tempo_2x15min":
            reps = _reps(2, Step(A, 3.85 * f, duration_s=900), Step(R, 2.5 * f, duration_s=180))
            return Workout(name, RunCategory.TEMPO, (warmup, *reps, cooldown))
        case "tempo_25min":
            block = Step(A, 3.8 * f, duration_s=1500)
            return Workout(name, RunCategory.TEMPO, (warmup, block, cooldown))
        case "long_run" | "long_run_mp" | "long_run_hmp":
            steady = LONG_RUN_SPEED * f
            # The burn-in: a lap per km, each a little faster as the runner settles in
            burn_in = [
                Step(W, steady * x, distance_m=1000) for x in np.linspace(0.9, 0.97, burn_in_km)
            ]
            rest_km = distance_km - burn_in_km
            n, block_km, speed, pace = LONG_RUN_BLOCKS.get(name, (0, 0, 0, None))
            # The runner laps at each block's start and end; steady running is split evenly
            # before, between and after the blocks
            gap_m = (rest_km - n * block_km) * 1000 / (n + 1)
            main = [Step(S, steady, distance_m=gap_m)]
            for _ in range(n):
                main += [Step(A, speed * f, distance_m=block_km * 1000), main[0]]
            return Workout(
                name,
                RunCategory.LONG_RUN,
                (*burn_in, *main),
                auto_lap_m=1000,
                fatigue=min(1, distance_km / 30),
                block_pace=pace,
            )
        case "race_10k":
            step = Step(A, 4.1 * f, distance_m=10_000)
            return Workout(name, RunCategory.RACE, (step,), auto_lap_m=1000, fatigue=0.3)
        case "race_half":
            step = Step(A, 3.85 * f, distance_m=21_097.5)
            return Workout(name, RunCategory.RACE, (step,), auto_lap_m=1000, fatigue=0.6)
    raise KeyError(f"unknown workout {name!r}; choose from {', '.join(WORKOUTS)}")


def _reps(n: int, work: Step, rest: Step) -> list[Step]:
    steps = []
    for i in range(n):
        steps.append(work)
        if i < n - 1:
            steps.append(rest)
    return steps


def generate(
    name: str,
    *,
    start: datetime,
    seed: int = 0,
    fitness: float = 1.0,
    heat: bool = False,
    **kwargs,
) -> ParsedRun:
    """A synthetic run of the named workout, starting at `start` (tz-aware UTC)."""
    return simulate(
        workout(name, fitness, **kwargs), start=start, seed=seed, fitness=fitness, heat=heat
    )


def simulate(
    w: Workout, *, start: datetime, seed: int = 0, fitness: float = 1.0, heat: bool = False
) -> ParsedRun:
    rng = np.random.default_rng(seed)
    speed, step_no, step_dist = _speed_profile(w, rng)
    n = len(speed)
    t = np.arange(n, dtype=float)
    distance = np.concatenate([[0.0], np.cumsum(speed[:-1])])
    phi = w.fatigue * (t / n) ** 2  # fatigue, growing towards the end

    # Heart rate: first-order response to a speed-driven target, plus drift with time, plus a
    # slow component that builds up during hard running (rep-on-rep rise, elevated recovery)
    drift_per_h = rng.uniform(3, 6) * (1.8 if heat else 1.0)
    target = HR_FLOOR + HR_PER_SPEED * speed / fitness + drift_per_h * t / 3600 + (3 if heat else 0)
    hard = np.maximum(0.0, speed / fitness - HARD_SPEED) * HR_PER_SPEED
    hr = np.empty(n)
    h, slow = 95.0, 0.0  # a warmed-up resting HR at the start
    for i in range(n):
        slow += (hard[i] - slow) / TAU_HR_SLOW
        tgt = min(target[i] + 0.5 * slow, HR_CEILING)
        h += (tgt - h) / (TAU_HR_UP if tgt > h else TAU_HR_DOWN)
        hr[i] = h
    hr = np.round(hr + _noise(rng, n, 1.2))

    # Running dynamics; a watch writes 0 while standing
    running = speed > RUNNING_SPEED
    moving = speed > 0.3
    spm = np.where(running, 150 + 9.0 * speed, np.where(moving, 100 + 12 * speed, 0.0))
    spm = np.where(moving, spm - 1.5 * phi + _noise(rng, n, 1.2), 0.0)
    step_length = np.where(moving, speed / np.maximum(spm, 1) * 60 * 1000, 0.0)
    vo = np.where(running, 62 + 7.5 * speed + 9 * phi, 35.0) + _noise(rng, n, 1.5)
    vo = np.where(moving, vo, 0.0)
    stance = np.where(running, 330 - 25 * speed + 15 * phi, 380.0) + _noise(rng, n, 4)
    stance = np.where(moving, stance, 0.0)

    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    altitude = (
        40
        + w.hills_m * np.sin(2 * np.pi * distance / 3200 + p1)
        + 0.4 * w.hills_m * np.sin(2 * np.pi * distance / 900 + p2)
    )
    lat, lon = _route(distance, w.loop_m)

    index = pd.date_range(start, periods=n, freq="1s", tz="UTC", unit="us", name="timestamp")
    records = pd.DataFrame(
        {
            "distance": distance,
            "heart_rate": hr,
            "speed": np.round(speed, 3),
            "altitude": np.round(altitude, 1),
            "cadence": np.round(spm / 2),  # FIT running cadence is one foot, whole rpm
            "stance_time": np.round(stance, 1),
            "vertical_oscillation": np.round(vo, 1),
            "vertical_ratio": np.round(
                np.where(step_length > 0, 100 * vo / np.maximum(step_length, 1), 0.0), 2
            ),
            "step_length": np.round(step_length),
            "position_lat": lat,
            "position_long": lon,
        },
        index=index,
    )
    lap_key = step_no * 10_000 + ((step_dist // w.auto_lap_m).astype(int) if w.auto_lap_m else 0)
    lap_index = pd.factorize(lap_key)[0]
    records["elapsed_s"] = t
    records["lap_index"] = pd.array(lap_index, dtype="Int64")
    records.attrs["units"] = {
        "distance": "m",
        "heart_rate": "bpm",
        "speed": "m/s",
        "altitude": "m",
        "cadence": "rpm",
        "stance_time": "ms",
        "vertical_oscillation": "mm",
        "vertical_ratio": "percent",
        "step_length": "mm",
        "position_lat": "deg",
        "position_long": "deg",
        "elapsed_s": "s",
    }
    laps = _laps(records)
    return ParsedRun(records=records, laps=laps, summary=_summary(records, laps, start))


def _speed_profile(w: Workout, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Per-second speed, step number and distance into the step."""
    total = sum(s.duration_s or s.distance_m / s.speed for s in w.steps)
    speeds, steps, dists = [], [], []
    v_prev, t0 = 0.0, 0
    for i, s in enumerate(w.steps):
        n_max = int(s.duration_s or s.distance_m / s.speed * 1.25 + 60)
        tt = np.arange(n_max)
        tgt = s.speed * (1 - 0.05 * w.fatigue * ((t0 + tt) / total) ** 2)
        v = tgt + (v_prev - tgt) * np.exp(-tt / TAU_SPEED)
        v = np.clip(v + _noise(rng, n_max, 0.05 if s.speed > RUNNING_SPEED else 0.08), 0, None)
        cum = np.cumsum(v)
        n = int(np.searchsorted(cum, s.distance_m)) + 1 if s.distance_m else int(s.duration_s)
        speeds.append(v[:n])
        steps.append(np.full(n, i))
        dists.append(np.concatenate([[0.0], cum[: n - 1]]))
        v_prev, t0 = v[n - 1], t0 + n
    return np.concatenate(speeds), np.concatenate(steps), np.concatenate(dists)


def _noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """Smooth Gaussian noise with standard deviation sigma (a 10 s moving sum)."""
    k = 10
    return np.convolve(rng.standard_normal(n + k - 1), np.ones(k) / np.sqrt(k), "valid") * sigma


def _route(distance: np.ndarray, loop_m: float) -> tuple[np.ndarray, np.ndarray]:
    """A wobbly elliptical loop, repeated, in degrees."""
    r = loop_m / (2 * np.pi)
    theta = 2 * np.pi * distance / loop_m
    x = 1.25 * r * np.cos(theta) + 0.08 * r * np.sin(3 * theta)
    y = 0.75 * r * np.sin(theta)
    lat0, lon0 = ROUTE_CENTER
    lat = lat0 + y / 111_320
    lon = lon0 + x / (111_320 * np.cos(np.radians(lat0)))
    return np.round(lat, 7), np.round(lon, 7)


def _laps(records: pd.DataFrame) -> pd.DataFrame:
    """One row per lap with Stridelab's lap columns. Laps tile the timeline."""
    rows = []
    groups = list(records.groupby("lap_index", sort=True))
    for k, (lap, g) in enumerate(groups):
        nxt = groups[k + 1][1] if k + 1 < len(groups) else None
        end = nxt.index[0] if nxt is not None else g.index[-1]
        end_dist = nxt["distance"].iloc[0] if nxt is not None else g["distance"].iloc[-1]
        secs = (end - g.index[0]).total_seconds()
        dist = end_dist - g["distance"].iloc[0]
        moving = g[g["cadence"] > 0]
        climb = g["altitude"].diff()
        rows.append(
            {
                "message_index": int(lap),
                "timestamp": end,
                "start_time": g.index[0],
                "total_timer_time": secs,
                "total_elapsed_time": secs,
                "total_distance": round(dist, 2),
                "sport": "running",
                "sub_sport": "generic",
                "max_heart_rate": int(g["heart_rate"].max()),
                "min_heart_rate": int(g["heart_rate"].min()),
                "avg_heart_rate": round(g["heart_rate"].mean()),
                "max_speed": g["speed"].max(),
                "avg_speed": round(dist / secs, 3) if secs else 0.0,
                "avg_cadence": round(moving["cadence"].mean()) if len(moving) else 0,
                "avg_step_length": round(moving["step_length"].mean(), 1) if len(moving) else 0.0,
                "max_cadence": int(g["cadence"].max()),
                "total_descent": round(-climb[climb < 0].sum()),
                "total_ascent": round(climb[climb > 0].sum()),
                "avg_stance_time": round(moving["stance_time"].mean(), 1) if len(moving) else 0.0,
                "avg_vertical_oscillation": round(moving["vertical_oscillation"].mean(), 1)
                if len(moving)
                else 0.0,
                "avg_vertical_ratio": round(moving["vertical_ratio"].mean(), 2)
                if len(moving)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _summary(records: pd.DataFrame, laps: pd.DataFrame, start: datetime) -> dict:
    """Stridelab's Activity.summary() keys."""
    elapsed = float(laps["total_elapsed_time"].sum())
    distance = float(laps["total_distance"].sum())
    return {
        "device": "Toy watch",
        "manufacturer": "development",
        "sport": "running",
        "sub_sport": "generic",
        "start": start,
        "utc_offset": UTC_OFFSET,
        "distance_m": round(distance, 2),
        "timer_time_s": elapsed,
        "elapsed_time_s": elapsed,
        "avg_heart_rate_bpm": round(records["heart_rate"].mean()),
        "max_heart_rate_bpm": int(records["heart_rate"].max()),
        "avg_speed_m_per_s": round(distance / elapsed, 3),
        "max_speed_m_per_s": float(records["speed"].max()),
        "avg_cadence_rpm": round(records.loc[records["cadence"] > 0, "cadence"].mean()),
        "avg_power_w": None,
        "total_ascent_m": int(laps["total_ascent"].sum()),
        "total_descent_m": int(laps["total_descent"].sum()),
        "total_calories_kcal": round(distance / 1000 * 70),
        "records": len(records),
        "laps": len(laps),
        "dev_fields": [],
        "warnings": [],
    }


class ToySource:
    """Stands in for Stridelab: any uploaded file becomes a synthetic run chosen from its bytes."""

    def parse(self, data: bytes, name: str) -> ParsedRun:
        seed = int.from_bytes(hashlib.sha256(data).digest()[:4], "little")
        workout_name = list(WORKOUTS)[seed % len(WORKOUTS)]
        start = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(hours=2)
        return generate(workout_name, start=start, seed=seed)


@dataclass
class DemoRun:
    parsed: ParsedRun
    workout: Workout
    tags: list[str]
    seed: int


def demo_history(weeks: int = 12, *, end: date | None = None, seed: int = 0) -> list[DemoRun]:
    """A training block ending on `end` (default today): five runs a week, getting fitter.

    Tue intervals (5x6min / 6x1000m alternating), Wed easy, Thu tempo (2x15 / 25 min),
    Sat easy, Sun long run growing in distance, with a 2-4 km burn-in and in turn no pace
    blocks, 2 x 5 km at marathon pace or 3 x 3 km at half marathon pace. A 10k replaces the
    middle week's long run and a half marathon the last one's.
    """
    rng = np.random.default_rng(seed)
    end = end or datetime.now(UTC).date()
    first_monday = end - timedelta(days=end.weekday(), weeks=weeks - 1)
    runs = []
    for wk in range(weeks):
        fitness = 1 + 0.004 * wk
        sunday = ("long_run", "long_run_mp", "long_run_hmp")[wk % 3]
        sunday_kwargs = {"distance_km": min(30, 18 + 1.0 * wk), "burn_in_km": 2 + wk % 3}
        if wk == weeks // 2:
            sunday, sunday_kwargs = "race_10k", {}
        elif wk == weeks - 1 and weeks >= 8:
            sunday, sunday_kwargs = "race_half", {}
        plan = [
            (1, "intervals_5x6min" if wk % 2 == 0 else "intervals_6x1000m", {}),
            (2, "easy", {"minutes": float(rng.choice([40, 50, 60]))}),
            (3, "tempo_25min" if wk % 2 == 0 else "tempo_2x15min", {}),
            (5, "easy", {"minutes": float(rng.choice([35, 45]))}),
            (6, sunday, sunday_kwargs),
        ]
        for day, name, kwargs in plan:
            d = first_monday + timedelta(weeks=wk, days=day)
            if d > end:
                continue
            run_seed = int(rng.integers(2**31))
            heat = bool(rng.random() < 0.15)
            start = datetime.combine(d, time(6, 0), tzinfo=UTC) + timedelta(
                minutes=int(rng.integers(0, 90))
            )
            w = workout(name, fitness * rng.normal(1, 0.005), **kwargs)
            parsed = simulate(w, start=start, seed=run_seed, fitness=fitness, heat=heat)
            tags = (["track"] if name == "intervals_6x1000m" else []) + (["heat"] if heat else [])
            runs.append(DemoRun(parsed, w, tags, run_seed))
    return runs
