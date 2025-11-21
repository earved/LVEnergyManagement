# LV Operating Strategies

This project simulates operating strategies for a Low Voltage (LV) power system consisting of a battery storage system and a DC/DC converter. It aims to analyze the performance, efficiency, and losses of different control strategies under realistic power demand scenarios.

## Features

- **Battery Model**: Simulates a battery with State of Charge (SOC) tracking, open-circuit voltage dependence on SOC, and internal resistance losses.
- **DC/DC Converter Model**: Simulates a converter with efficiency curves based on load power.
- **Operating Strategies**:
    - **Dumb Strategy**: A baseline strategy where the battery handles high-frequency load fluctuations (residuals) while the converter handles the smoothed load demand.
- **Visualization**: Generates detailed plots of power flows, SOC, voltages, and efficiency maps.

## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`

## Usage

1.  Ensure you have the required libraries installed:
    ```bash
    pip install numpy pandas matplotlib
    ```

2.  Run the simulation:
    ```bash
    python3 operating_strategies.py
    ```

3.  The script will:
    - Load the power demand scenario from `lv_power_demand.csv`.
    - Run the "Dumb Strategy" simulation.
    - Display three figures:
        1.  **Simulation Results**: Time-series plots of power, SOC, losses, and voltage.
        2.  **DC/DC Efficiency**: The efficiency curve of the converter with operating points marked.
        3.  **Battery Efficiency Map**: A heatmap of battery efficiency across different power and SOC levels.

## Models

### Battery Model
- **Inputs**: Current or Power.
- **Dynamics**: Updates SOC based on coulomb counting.
- **Losses**: $I^2 R$ losses based on internal resistance.
- **Voltage**: $V_{term} = V_{oc}(SOC) - I \cdot R_{int}$

### DC/DC Converter
- **Efficiency**: Modeled as a parabolic curve peaking at an optimal power level ($P_{opt}$).
- **Losses**: Calculated based on efficiency $\eta$.

## License

This project is for educational and research purposes.
