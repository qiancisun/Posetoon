import numpy as np

JERK_THRESHOLD_DEG = 8.0
STANCE_VELOCITY_PCTL = 35


def angular_velocity(angles_rad):
    unwrapped = np.unwrap(np.asarray(angles_rad, dtype=float))
    return np.degrees(np.diff(unwrapped))


def smoothness(angles_rad, jerk_threshold_deg=JERK_THRESHOLD_DEG):
    vel = angular_velocity(angles_rad)
    accel = np.diff(vel) if vel.size > 1 else np.array([0.0])
    unwrapped_deg = np.degrees(np.unwrap(np.asarray(angles_rad, dtype=float)))
    return {
        "jerky_frames": int(np.sum(np.abs(vel) > jerk_threshold_deg)),
        "jerky_fraction": float(np.mean(np.abs(vel) > jerk_threshold_deg)),
        "vel_abs_mean_deg": float(np.mean(np.abs(vel))),
        "vel_abs_max_deg": float(np.max(np.abs(vel))) if vel.size else 0.0,
        "accel_rms_deg": float(np.sqrt(np.mean(accel ** 2))),
        "range_deg": float(np.ptp(unwrapped_deg)),
    }


def phase_lag(series_a, series_b, max_lag=None):
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]

    if max_lag is None:
        max_lag = n // 2

    def _corr(x, y):
        if x.size < 2:
            return -np.inf
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
        return float(np.sum(x * y) / denom) if denom > 1e-12 else -np.inf

    best_lag, best_corr = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = _corr(a[:n - lag], b[lag:])
        else:
            corr = _corr(a[-lag:], b[:n + lag])
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag, best_corr


def detect_stance(paw_positions, percentile=STANCE_VELOCITY_PCTL):
    pos = np.asarray(paw_positions, dtype=float)
    speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    return speed <= np.percentile(speed, percentile)


def foot_slide(char_paw_positions, stance_mask):
    pos = np.asarray(char_paw_positions, dtype=float)
    mask = np.asarray(stance_mask, dtype=bool)

    slides = []
    start = None
    for i, in_stance in enumerate(mask):
        if in_stance and start is None:
            start = i
        elif not in_stance and start is not None:
            if i - start >= 2:
                run = pos[start:i]
                slides.append(float(np.max(np.linalg.norm(run - run.mean(axis=0), axis=1))))
            start = None
    if start is not None and len(mask) - start >= 2:
        run = pos[start:]
        slides.append(float(np.max(np.linalg.norm(run - run.mean(axis=0), axis=1))))

    if not slides:
        return {"mean_slide_px": 0.0, "max_slide_px": 0.0, "n_stance_runs": 0}
    return {
        "mean_slide_px": float(np.mean(slides)),
        "max_slide_px": float(np.max(slides)),
        "n_stance_runs": len(slides),
    }


def limit_saturation(angles_rad, lo_rad, hi_rad, tol_rad=np.radians(1.0)):
    a = np.asarray(angles_rad, dtype=float)
    at_limit = (a <= lo_rad + tol_rad) | (a >= hi_rad - tol_rad)
    return float(np.mean(at_limit))


def reprojection_error(char_positions, tracked_positions, scores=None,
                        score_threshold=0.3, spine_lengths=None):
    char = np.asarray(char_positions, dtype=float)
    trk = np.asarray(tracked_positions, dtype=float)
    err = np.linalg.norm(char - trk, axis=1)

    valid = np.ones(err.shape, dtype=bool)
    if scores is not None:
        valid &= np.asarray(scores, dtype=float) >= score_threshold
    if spine_lengths is not None:
        sl = np.asarray(spine_lengths, dtype=float)
        valid &= sl > 1e-6
        err = np.where(sl > 1e-6, err / np.maximum(sl, 1e-6), np.nan)

    if not np.any(valid):
        return {"mean": float("nan"), "median": float("nan"),
                "p90": float("nan"), "n_valid": 0}
    e = err[valid]
    return {
        "mean": float(np.mean(e)),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "n_valid": int(valid.sum()),
    }


def bone_length_consistency(pos_a, pos_b, scores_a=None, scores_b=None,
                             score_threshold=0.3):
    a = np.asarray(pos_a, dtype=float)
    b = np.asarray(pos_b, dtype=float)
    lengths = np.linalg.norm(a - b, axis=1)

    valid = np.ones(lengths.shape, dtype=bool)
    if scores_a is not None:
        valid &= np.asarray(scores_a, dtype=float) >= score_threshold
    if scores_b is not None:
        valid &= np.asarray(scores_b, dtype=float) >= score_threshold

    if valid.sum() < 2:
        return {"cv": float("nan"), "mean_px": float("nan"), "n_valid": int(valid.sum())}
    L = lengths[valid]
    return {
        "cv": float(np.std(L) / (np.mean(L) + 1e-9)),
        "mean_px": float(np.mean(L)),
        "n_valid": int(valid.sum()),
    }


MEASURE_FIELDS = ["lower", "upper", "head", "hum", "rad", "fem", "tib"]


def proportion_distance(measure_a, measure_b, reference=200.0):
    diffs = [abs(measure_a[k] - measure_b[k]) for k in MEASURE_FIELDS]
    return {
        "mean_abs_norm": float(np.mean(diffs) / reference),
        "max_abs_norm": float(np.max(diffs) / reference),
        "per_field_norm": {k: float(abs(measure_a[k] - measure_b[k]) / reference)
                            for k in MEASURE_FIELDS},
    }


def _self_test():
    rng = np.random.default_rng(0)
    n = 279

    print("=== smoothness: smooth vs jittery ===")
    t = np.linspace(0, 8 * np.pi, n)
    smooth_sig = 0.6 * np.sin(t)
    jittery_sig = smooth_sig + rng.normal(0, 0.25, n)
    s_smooth = smoothness(smooth_sig)
    s_jitter = smoothness(jittery_sig)
    print(f"  smooth : jerky={s_smooth['jerky_frames']:3}  "
          f"accel_rms={s_smooth['accel_rms_deg']:6.2f}  range={s_smooth['range_deg']:6.1f}")
    print(f"  jittery: jerky={s_jitter['jerky_frames']:3}  "
          f"accel_rms={s_jitter['accel_rms_deg']:6.2f}  range={s_jitter['range_deg']:6.1f}")
    assert s_jitter["jerky_frames"] > s_smooth["jerky_frames"]

    print("\n=== smoothness: over-smoothing is visible via range ===")
    mushy = 0.05 * np.sin(t)
    s_mushy = smoothness(mushy)
    print(f"  mushy  : jerky={s_mushy['jerky_frames']:3}  "
          f"range={s_mushy['range_deg']:6.1f}  <- low jerk BUT collapsed range")
    assert s_mushy["range_deg"] < s_smooth["range_deg"]

    print("\n=== phase_lag: recovers a known shift ===")
    for true_lag in [9, -21, -19]:
        base = np.sin(np.linspace(0, 6 * np.pi, n))
        shifted = np.roll(base, true_lag)
        found, corr = phase_lag(base, shifted, max_lag=60)
        print(f"  injected {true_lag:+3}  ->  recovered {found:+3}  (corr {corr:.3f})")

    print("\n=== foot_slide: planted vs sliding paw ===")
    paw_tracked = np.zeros((n, 2))
    paw_tracked[:, 0] = np.where(np.arange(n) % 40 < 20, 0.0, 1.0).cumsum() * 0.5
    stance = detect_stance(paw_tracked)
    planted = np.zeros((n, 2))
    sliding = np.column_stack([np.arange(n) * 0.4, np.zeros(n)])
    print(f"  stance frames detected: {int(stance.sum())}/{n}")
    print(f"  planted paw: {foot_slide(planted, stance)}")
    print(f"  sliding paw: {foot_slide(sliding, stance)}")
    assert foot_slide(sliding, stance)["mean_slide_px"] > \
           foot_slide(planted, stance)["mean_slide_px"]

    print("\n=== reprojection_error: low-confidence frames excluded ===")
    char = rng.normal(0, 1, (n, 2))
    trk = char + rng.normal(0, 2, (n, 2))
    scores = np.where(np.arange(n) < 30, 0.1, 0.9)
    r = reprojection_error(char, trk, scores)
    print(f"  n_valid={r['n_valid']} (expected {n - 30}), mean={r['mean']:.2f}")
    assert r["n_valid"] == n - 30

    print("\n=== proportion_distance ===")
    m1 = {"lower": 93.5, "upper": 93.5, "head": 93.2, "hum": 47.0,
          "rad": 34.6, "fem": 48.5, "tib": 44.4}
    m2 = {"lower": 100.0, "upper": 100.0, "head": 76.1, "hum": 41.8,
          "rad": 32.5, "fem": 47.5, "tib": 40.0}
    d = proportion_distance(m1, m2)
    print(f"  measured vs small template: mean={d['mean_abs_norm']:.4f}  "
          f"max={d['max_abs_norm']:.4f}")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
