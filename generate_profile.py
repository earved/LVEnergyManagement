import numpy as np
import pandas as pd

def generate_profile():
    # Parameters
    duration = 1000  # seconds
    dt = 0.01        # seconds
    n_steps = int(duration / dt)
    time = np.linspace(0, duration, n_steps)
    
    # Base profile (steps) - More diverse with shorter durations
    power = np.zeros(n_steps)
    
    # Define multiple segments with varying power levels
    # First 100 sampling points: 0W (only noise will be added later)
    power[0:100] = 0
    
    # 100-150 sampling points: Very low load (approx 150W)
    power[100:int(50/dt)] = 150
    
    # 50-100s: Low-medium load (approx 500W)
    power[int(50/dt):int(100/dt)] = 500
    
    # 100-150s: Medium load (approx 1200W)
    power[int(100/dt):int(150/dt)] = 1200
    
    # 150-200s: High load (approx 2500W)
    power[int(150/dt):int(200/dt)] = 2500
    
    # 200-250s: Medium-high load (approx 1800W)
    power[int(200/dt):int(250/dt)] = 1800
    
    # 250-300s: Low load (approx 300W)
    power[int(250/dt):int(300/dt)] = 300
    
    # 300-350s: Very high load (approx 3200W)
    power[int(300/dt):int(350/dt)] = 3200
    
    # 350-400s: Medium load (approx 1000W)
    power[int(350/dt):int(400/dt)] = 1000
    
    # 400-450s: Low-medium load (approx 700W)
    power[int(400/dt):int(450/dt)] = 700
    
    # 450-500s: High load (approx 2200W)
    power[int(450/dt):int(500/dt)] = 2200
    
    # 500-550s: Ramp up
    ramp1 = np.linspace(800, 2800, int(50/dt))
    power[int(500/dt):int(550/dt)] = ramp1
    
    # 550-600s: Peak load (approx 3500W)
    power[int(550/dt):int(600/dt)] = 3500
    
    # 600-650s: Ramp down
    ramp2 = np.linspace(3500, 1500, int(50/dt))
    power[int(600/dt):int(650/dt)] = ramp2
    
    # 650-700s: Medium load (approx 1400W)
    power[int(650/dt):int(700/dt)] = 1400
    
    # 700-750s: Low-medium load (approx 600W)
    power[int(700/dt):int(750/dt)] = 600
    
    # 750-800s: High load (approx 2600W)
    power[int(750/dt):int(800/dt)] = 2600
    
    # 800-850s: Medium-high load (approx 1900W)
    power[int(800/dt):int(850/dt)] = 1900
    
    # 850-900s: Low load (approx 400W)
    power[int(850/dt):int(900/dt)] = 400
    
    # 900-950s: Medium load (approx 1100W)
    power[int(900/dt):int(950/dt)] = 1100
    
    # 950s to last 100 sampling points: Very low load (approx 200W)
    power[int(950/dt):-100] = 200
    
    # Last 100 sampling points: 0W (only noise will be added later)
    power[-100:] = 0
    
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
