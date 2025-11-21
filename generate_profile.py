import numpy as np
import pandas as pd

def generate_profile():
    # Parameters
    duration = 1000  # seconds
    dt = 0.01        # seconds
    n_steps = int(duration / dt)
    time = np.linspace(0, duration, n_steps)
    
    # Base profile (steps)
    power = np.zeros(n_steps)
    
    # Define some segments
    # 0-100s: Low load (approx 200W)
    power[0:int(100/dt)] = 200
    
    # 100-400s: High load (approx 2000W)
    power[int(100/dt):int(400/dt)] = 2000
    
    # 400-600s: Medium load (approx 1000W)
    power[int(400/dt):int(600/dt)] = 1000
    
    # 600-800s: Variable load (ramp)
    ramp = np.linspace(1000, 3000, int(200/dt))
    power[int(600/dt):int(800/dt)] = ramp
    
    # 800-1000s: Low load
    power[int(800/dt):] = 200
    
    # Add noise
    # White noise
    noise_std = 50 # Standard deviation of noise
    noise = np.random.normal(0, noise_std, n_steps)
    
    # Add some higher frequency sine waves
    noise += 20 * np.sin(2 * np.pi * 1.0 * time) # 1 Hz
    noise += 10 * np.sin(2 * np.pi * 10.0 * time) # 10 Hz
    
    total_power = power + noise
    
    # Ensure non-negative power (optional, but realistic for consumption)
    total_power = np.maximum(total_power, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'power': total_power
    })
    
    # Save to CSV
    df.to_csv('lv_power_demand.csv', index=False)
    print(f"Generated lv_power_demand.csv with {n_steps} steps.")

if __name__ == "__main__":
    generate_profile()
