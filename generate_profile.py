import argparse
import numpy as np
import pandas as pd


def _build_base_profile(time, dt, seed=None):
    rng = np.random.default_rng(seed)
    n_steps = len(time)
    power = np.zeros(n_steps)

    segments = [
        (0, 1, 0),
        (1, 50, 150),
        (50, 100, 500),
        (100, 150, 1200),
        (150, 200, 2500),
        (200, 250, 1800),
        (250, 300, 300),
        (300, 350, 3200),
        (350, 400, 1000),
        (400, 450, 700),
        (450, 500, 2200),
    ]

    for start, end, val in segments:
        power[int(start / dt):int(end / dt)] = val

    ramp1_start = int(500 / dt)
    ramp1_end = int(550 / dt)
    power[ramp1_start:ramp1_end] = np.linspace(800, 2800, ramp1_end - ramp1_start)

    power[ramp1_end:int(600 / dt)] = 3500

    ramp2_start = int(600 / dt)
    ramp2_end = int(650 / dt)
    power[ramp2_start:ramp2_end] = np.linspace(3500, 1500, ramp2_end - ramp2_start)

    tail_segments = [
        (650, 700, 1400),
        (700, 750, 600),
        (750, 800, 2600),
        (800, 850, 1900),
        (850, 900, 400),
        (900, 950, 1100),
        (950, 1000, 200)
    ]

    for start, end, val in tail_segments:
        power[int(start / dt):int(end / dt)] = val

    power[-int(1 / dt):] = 0

    noise_std = 50
    noise = rng.normal(0, noise_std, n_steps)
    noise += 20 * np.sin(2 * np.pi * 1.0 * time)
    noise += 10 * np.sin(2 * np.pi * 10.0 * time)

    total_power = np.maximum(power + noise, 0)
    return total_power


def generate_profile(seed=None):
    duration = 1000
    dt = 0.01
    n_steps = int(duration / dt)
    time = np.linspace(0, duration, n_steps)

    power = _build_base_profile(time, dt, seed=seed)
    df = pd.DataFrame({"time": time, "power": power})
    df.to_csv("lv_power_demand.csv", index=False)
    print(f"Generated lv_power_demand.csv with {n_steps} steps.")


def generate_profile_variant(seed=None, filename="lv_power_demand_variant.csv"):
    duration = 1000
    dt = 0.01
    n_steps = int(duration / dt)
    time = np.linspace(0, duration, n_steps)

    base_power = _build_base_profile(time, dt, seed=seed)

    rng = np.random.default_rng(seed if seed is not None else None)
    scale = rng.uniform(0.8, 1.2)
    drift = rng.normal(0, 150, n_steps)
    variant_power = np.maximum(base_power * scale + drift, 0)

    df_variant = pd.DataFrame({"time": time, "power": variant_power})
    df_variant.to_csv(filename, index=False)
    print(f"Generated {filename} with {n_steps} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LV demand scenarios.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    parser.add_argument(
        "--variant-filename",
        type=str,
        default="lv_power_demand_variant.csv",
        help="Filename for the second scenario CSV.",
    )
    args = parser.parse_args()

    generate_profile(seed=args.seed)
    generate_profile_variant(seed=args.seed, filename=args.variant_filename)
