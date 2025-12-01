"""Simulation models and operating strategies for LV energy management."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

class BattModel:
    """Simple equivalent battery model with linear open-circuit voltage."""

    def __init__(self, U_min, U_max, R_int, capacity_Ah, start_soc=80) -> None:
        """Create a battery model instance.

        Args:
            U_min (float): Open-circuit voltage at 0% SOC in volts.
            U_max (float): Open-circuit voltage at 100% SOC in volts.
            R_int (float): Internal resistance in ohms.
            capacity_Ah (float): Battery capacity in ampere-hours.
            start_soc (float, optional): Initial state of charge in percent.
        """
        self.U_min = U_min
        self.U_max = U_max
        self.R_int = R_int
        self.capacity_C = capacity_Ah * 3600  # Convert Ah to coulombs.
        self.start_soc = start_soc
        self.charge_remaining = self.capacity_C * start_soc/100

    def get_soc(self) -> float:
        """Return the current state of charge in percent."""
        return max(min(self.charge_remaining / self.capacity_C * 100, 100), 0)

    def get_U_oc(self, soc = None) -> float:
        """Compute the open-circuit voltage for a given SOC.

        Args:
            soc (float | None): State of charge in fraction (0-1). Uses current SOC if None.

        Returns:
            float: Open-circuit voltage in volts.
        """
        if soc is None:
            soc = self.get_soc() / 100
        return self.U_min + soc * (self.U_max - self.U_min)

    def get_voltage(self, current=None, power=None, soc=None) -> float:
        """Calculate the terminal voltage under load.

        Args:
            current (float | None): Applied current in amperes. Positive discharges the battery.
            power (float | None): Applied power in watts. Used if current is None.
            soc (float | None): State of charge fraction used for open-circuit voltage.

        Returns:
            float: Terminal voltage in volts.
        """
        I = self._resolve_current(current, power)
        U_oc = self.get_U_oc(soc=soc)
        return max(U_oc - I * self.R_int, 0)

    def _resolve_current(self, current=None, power=None, soc = None) -> float:
        """Resolve the battery current from current or power input.

        Args:
            current (float | None): Current request in amperes.
            power (float | None): Power request in watts.
            soc (float | None): State of charge fraction used for open-circuit voltage.

        Returns:
            float: Battery current in amperes.

        Raises:
            ValueError: If neither current nor power is provided.
        """
        if current is not None:
            return current
        elif power is not None:
            U_oc = self.get_U_oc(soc=soc)
            discriminant = U_oc**2 - 4*self.R_int*power
            if discriminant < 0:
                # Power exceeds maximum possible power; return current at max power point.
                return U_oc / (2 * self.R_int)
            
            I1 = (U_oc - discriminant**0.5)/(2*self.R_int)
            I2 = (U_oc + discriminant**0.5)/(2*self.R_int)
            return I1 if abs(I1) < abs(I2) else I2
        else:
            raise ValueError("Provide either current or power.")

    def update_charge(self, dt, current=None, power=None) -> dict:
        """Advance the battery state for a time step.

        Args:
            dt (float): Time step in seconds.
            current (float | None): Applied current in amperes.
            power (float | None): Applied power in watts.

        Returns:
            dict: Updated telemetry (SOC, voltage, efficiency, current, power, losses).
        """
        I = self._resolve_current(current, power)
        self.charge_remaining -= I * dt  # Negative current increases charge.
        # Clamp charge between empty and full.
        self.charge_remaining = max(min(self.charge_remaining, self.capacity_C), 0)

        U_term = self.get_voltage(current=I)

        return {
            "soc" : self.get_soc(),
            "voltage" : U_term,
            "eta" : self.get_efficiency(current = I),
            "current" : I,
            "power" : I*U_term,
            "losses" : I**2*self.R_int
        }

    def get_efficiency(self, current=None, power=None, soc = None) -> float:
        """Estimate instantaneous charge or discharge efficiency.

        Args:
            current (float | None): Applied current in amperes.
            power (float | None): Applied power in watts.
            soc (float | None): State of charge fraction used for voltage.

        Returns:
            float: Efficiency as a value between 0 and 1.
        """
        I = self._resolve_current(current, power, soc=soc)
        U_oc = self.get_U_oc(soc=soc)
        U_term = self.get_voltage(current=I, soc=soc)
        if I > 0:  # Discharging
            P_load = U_term * I
            P_loss = I**2 * self.R_int
            return P_load / (P_load + P_loss)
        else:  # Charging
            P_in = abs(U_term * I)
            P_loss = I**2 * self.R_int
            return (P_in - P_loss) / P_in if P_in > 0 else 0

    def get_efficiency_curve(self, max_load_power=2000, soc=None):
        """Generate an efficiency curve for a fixed SOC.

        Args:
            max_load_power (float): Upper bound for discharge power sweep in watts.
            soc (float | None): State of charge fraction for the curve.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Power, efficiency, and voltage samples.
        """
        p_values = np.linspace(0,max_load_power,200)
        eta_values =[self.get_efficiency(power=p, soc=soc) for p in p_values]
        V_values = [self.get_voltage(power=p, soc=soc) for p in p_values]
        return p_values, eta_values, V_values

    def get_losses_for_power_dem_vec(self, power_vec=None, current_vec=None, soc=None):
        """Compute resistive losses for a vector of power demands.

        Args:
            power_vec (np.ndarray | None): Load powers in watts.
            current_vec (np.ndarray | None): Optional currents that override power.
            soc (float | None): State of charge fraction used for voltage estimation.

        Returns:
            list[float]: Power loss values in watts.
        """
        eta_values =[self.get_efficiency(power=p, soc=soc) for p in power_vec]
        i_values = [self._resolve_current(power=p, soc=soc) for p in power_vec]
        V_values = [self.get_voltage(power=p, soc=soc) for p in power_vec]
        p_loss_values = [i**2*self.R_int for i in i_values]
        return p_loss_values
       
    def plot_efficiency_map(self, power_vec_operation=None, soc_vec_operation=None, power_limits=None):
        """Plot the battery efficiency map with optional operating points.

        Args:
            power_vec_operation (Iterable[float] | None): Historical power samples.
            soc_vec_operation (Iterable[float] | None): Corresponding SOC samples.
            power_limits (tuple[float, float] | None): Optional fixed min/max power axis limits.

        Returns:
            matplotlib.figure.Figure: Figure handle for the contour plot.
        """
        soc_range = range(101)
        
        # Determine power range.
        if power_limits is not None:
            min_p, max_p = power_limits
        elif power_vec_operation is not None and len(power_vec_operation) > 0:
            min_p_op = min(power_vec_operation)
            max_p_op = max(power_vec_operation)
            margin = (max_p_op - min_p_op) * 0.1 if max_p_op != min_p_op else 100
            min_p = min_p_op - margin
            max_p = max_p_op + margin
        else:
            max_theoretical = self.U_max * self.capacity_C / 3600
            min_p = -max_theoretical
            max_p = max_theoretical
        
        # Create grid for contour plot.
        p_grid = np.linspace(min_p, max_p, 200)
        soc_grid = np.array(soc_range)
        P, S = np.meshgrid(p_grid, soc_grid)
        
        # Calculate efficiency for the grid.
        Z = np.zeros_like(P)
        for i, soc in enumerate(soc_grid):
            for j, p in enumerate(p_grid):
                Z[i, j] = self.get_efficiency(power=p, soc=soc)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Heatmap.
        cax = ax.imshow(Z, aspect='auto', origin='lower', 
                           extent=[min_p, max_p, 0, 100], cmap='viridis')
        fig.colorbar(cax, label="Efficiency [0-1]")
        
        # Contour lines.
        contours = ax.contour(P, S, Z, levels=10, colors='white', alpha=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

        # Operating points.
        if power_vec_operation is not None and soc_vec_operation is not None:
            ax.scatter(power_vec_operation, soc_vec_operation, c='red', s=10, alpha=0.5, label="Operating Points")
            ax.legend()

        ax.set_title("Battery Efficiency Map")
        ax.set_xlabel("Power [W] (Negative = Charging)")
        ax.set_ylabel("SOC [%]")
        ax.set_xlim(min_p, max_p)
        return fig


class LVDCDC:
    """Physics-based DC/DC converter model with simple loss formulation."""

    def __init__(self, hv_voltage, p_fixed=15.0, k_sw=0.1, k_cond=0.002, max_power=4000, max_power_rate=500.0) -> None:
        """Create a converter instance.

        Args:
            hv_voltage (float): High-voltage bus value in volts.
            p_fixed (float): Constant loss term in watts.
            k_sw (float): Coefficient for switching losses per ampere.
            k_cond (float): Coefficient for conduction losses per ampere squared.
            max_power (float): Maximum LV-side power in watts.
            max_power_rate (float): Maximum rate of change of power in watts per second.
        """
        self.hv_voltage = hv_voltage
        self.p_fixed = p_fixed
        self.k_sw = k_sw
        self.k_cond = k_cond
        self.max_power = max_power
        self.max_power_rate = max_power_rate

    def calculate_losses(self, current) -> float:
        """Return total losses for a given LV-side current.

        Args:
            current (float): LV-side current in amperes.

        Returns:
            float: Losses in watts.
        """
        return self.p_fixed + self.k_sw * abs(current) + self.k_cond * current**2

    def limit_power(self, target_power, current_power, dt) -> float:
        """Rate-limit the converter power command.

        Args:
            target_power (float): Desired LV-side power in watts.
            current_power (float): Current LV-side power in watts.
            dt (float): Time step in seconds.

        Returns:
            float: Rate-limited power command in watts.
        """
        delta_p = target_power - current_power
        max_delta = self.max_power_rate * dt
        
        if abs(delta_p) <= max_delta:
            return target_power
        else:
            return current_power + np.sign(delta_p) * max_delta

    def compute(self, lv_voltage, current=None, power=None) -> dict:
        """Evaluate converter performance for the supplied load.

        Args:
            lv_voltage (float): LV-side voltage in volts.
            current (float | None): LV-side current in amperes.
            power (float | None): LV-side power in watts.

        Returns:
            dict: Dictionary containing LV voltage, power, efficiency, input power/current, and losses.

        Raises:
            ValueError: If neither current nor power is supplied.
        """
        if current is not None:
            i_out = current
            p_out = lv_voltage * current
        elif power is not None:
            p_out = power
            i_out = p_out / lv_voltage if lv_voltage > 0 else 0
        else:
            raise ValueError("Provide either current or power.")

        losses = self.calculate_losses(i_out)
        p_in = p_out + losses
        eta = p_out / p_in if p_in > 0 else 0.0
        i_in = p_in / self.hv_voltage if self.hv_voltage > 0 else float('inf')

        return {
            "lv_voltage": lv_voltage,
            "p_out": p_out,
            "efficiency": eta,
            "p_in": p_in,
            "i_in": i_in,
            "losses": losses
        }

    def plot_efficiency_curve(self, p_vec, eta_vec, voltage=12.0, limits=None):
        """Plot simulated efficiency points against a reference curve.

        Args:
            p_vec (Iterable[float]): Power samples in watts.
            eta_vec (Iterable[float]): Efficiency samples (0-1).
            voltage (float): Reference LV voltage for the analytic curve.
            limits (dict | None): Optional axis limit overrides with keys
                "power", "efficiency", and "hist" (tuples of (min, max)).

        Returns:
            matplotlib.figure.Figure: Figure handle for the plot.
        """
        p_vec = np.asarray(list(p_vec)) if p_vec is not None else np.array([])
        eta_vec = np.asarray(list(eta_vec)) if eta_vec is not None else np.array([])

        # Convert efficiency to percentage.
        eta_percent = eta_vec * 100 if eta_vec.size > 0 else np.array([])

        # Calculate reference curve.
        p_ref = np.linspace(0, self.max_power, 100)
        eta_ref = []
        for p in p_ref:
            i = p / voltage if voltage > 0 else 0
            loss = self.calculate_losses(i)
            p_in = p + loss
            eta = p / p_in if p_in > 0 else 0
            eta_ref.append(eta * 100)

        fig, ax_eff = plt.subplots(figsize=(10, 6))
        ax_eff.plot(p_ref, eta_ref, label=f"Reference Curve (@{voltage}V)", linewidth=2, linestyle='--')
        if p_vec.size > 0 and eta_percent.size > 0:
            ax_eff.scatter(p_vec, eta_percent, color='red', alpha=0.5, label="Operating Points", s=10)

        ax_eff.set_title("DC/DC Converter Efficiency vs. Load Power")
        ax_eff.set_xlabel("Load Power [W]")
        ax_eff.set_ylabel("Efficiency [%]")
        ax_eff.grid(True, linestyle='--', alpha=0.7)

        hist_handles = []
        if p_vec.size > 0:
            hist_min = min(0.0, float(np.min(p_vec)))
            hist_max = max(self.max_power, float(np.max(p_vec)))
            if hist_max - hist_min < 1e-6:
                hist_max = hist_min + 1.0
            bins = np.linspace(hist_min, hist_max, 40)
            hist_counts, bin_edges = np.histogram(p_vec, bins=bins)
            total_samples = hist_counts.sum()
            if total_samples > 0:
                hist_percent = hist_counts / total_samples * 100.0
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                bar_widths = np.diff(bin_edges)
                ax_hist = ax_eff.twinx()
                bars = ax_hist.bar(
                    bin_centers,
                    hist_percent,
                    width=bar_widths,
                    color='tab:blue',
                    alpha=0.3,
                    label="Operating Time Share",
                    align='center'
                )
                ax_hist.set_ylabel("Time Share [%]")
                if limits and "hist" in limits:
                    ax_hist.set_ylim(*limits["hist"])
                else:
                    ax_hist.set_ylim(0, max(hist_percent) * 1.2 if hist_percent.size > 0 else 1.0)
                hist_handles.append(bars)

        if limits:
            if "power" in limits:
                ax_eff.set_xlim(*limits["power"])
            if "efficiency" in limits:
                ax_eff.set_ylim(*limits["efficiency"])

        handles_eff, labels_eff = ax_eff.get_legend_handles_labels()
        handles_hist, labels_hist = (hist_handles, [h.get_label() for h in hist_handles]) if hist_handles else ([], [])
        if handles_hist:
            ax_eff.legend(handles_eff + handles_hist, labels_eff + labels_hist, loc='best')
        else:
            ax_eff.legend(loc='best')

        fig.tight_layout()
        return fig

def plot_results(results, plotstring, axis_limits=None):
    """Render standard plots for converter and battery telemetry.

    Args:
        results (dict): Dictionary containing time, load, battery, and converter traces.
        plotstring (str): Title prefix used for the plot window.
        axis_limits (dict | None): Optional axis limits with keys
            "time", "power", "soc", "losses", "voltage", "energy".

    Returns:
        matplotlib.figure.Figure: Figure handle for the combined plot.
    """
   
    batt_result_dict = results["batt_result_dict"]
    conv_result_dict = results["conv_result_dict"]


    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(plotstring, fontsize=16)
    
    axs[0].plot(results["time"], results["load_dem"], label="LV Power Demand", alpha=0.7)
    axs[0].plot(results["time"], conv_result_dict["p_out"], label="DC/DC Power", linestyle='--')
    axs[0].plot(results["time"], batt_result_dict["power"], label="Battery Power", linestyle=':')
    axs[0].set_ylabel("Power [W]")
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    if axis_limits and "power" in axis_limits:
        axs[0].set_ylim(*axis_limits["power"])

    axs[1].plot(results["time"], batt_result_dict["soc"], label="Battery SOC", color='green')
    axs[1].set_ylabel("SOC [%]")
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    if axis_limits and "soc" in axis_limits:
        axs[1].set_ylim(*axis_limits["soc"])

    axs[2].plot(results["time"], batt_result_dict["losses"], label="Battery Losses")
    axs[2].plot(results["time"], conv_result_dict["losses"], label="DC/DC Losses")
    axs[2].set_ylabel("Power Loss [W]")
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.5)
    if axis_limits and "losses" in axis_limits:
        axs[2].set_ylim(*axis_limits["losses"])

    axs[3].plot(results["time"], batt_result_dict["voltage"], label="Battery Voltage", color='orange')
    axs[3].set_ylabel("Voltage [V]")
    axs[3].legend(loc='upper right')
    axs[3].grid(True, linestyle='--', alpha=0.5)
    if axis_limits and "voltage" in axis_limits:
        axs[3].set_ylim(*axis_limits["voltage"])

    # Calculate cumulative energy losses in watt-hours.
    time = results["time"]
    dt = time[1] - time[0] if len(time) > 1 else 0
    batt_loss_wh = np.cumsum(batt_result_dict["losses"]) * dt / 3600
    conv_loss_wh = np.cumsum(conv_result_dict["losses"]) * dt / 3600
    total_loss_wh = batt_loss_wh + conv_loss_wh

    axs[4].plot(time, batt_loss_wh, label="Cum. Battery Loss")
    axs[4].plot(time, conv_loss_wh, label="Cum. DC/DC Loss")
    axs[4].plot(time, total_loss_wh, label="Cum. Total Loss", linestyle='--', color='black')
    axs[4].set_ylabel("Energy Loss [Wh]")
    axs[4].set_xlabel("Time [s]")
    axs[4].legend(loc='upper right')
    axs[4].grid(True, linestyle='--', alpha=0.5)
    if axis_limits and "energy" in axis_limits:
        axs[4].set_ylim(*axis_limits["energy"])

    if axis_limits and "time" in axis_limits:
        for ax in axs:
            ax.set_xlim(*axis_limits["time"])

    plt.tight_layout()
    return fig

def get_load_scenario(ind=1):
    """Load LV demand scenario data from disk.

    Args:
        ind (int): Scenario index to read.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Time vector, raw demand, and smoothed demand.

    Raises:
        ValueError: If the requested scenario index is unavailable.
    """
    match ind:
        case 1:
            scenario = pd.read_csv("lv_power_demand.csv")
        case 2:
            pass
        case _:
            raise ValueError("Scenario index not available")
       
    power_dem = scenario["power"]
    power_dem_smoothed = power_dem.rolling(window=4).mean()
    power_dem_smoothed[power_dem_smoothed.isna()] = 0
    return scenario["time"].to_numpy(), power_dem.to_numpy(), power_dem_smoothed.to_numpy()


def compute_converter_efficiency_info(converter, lv_voltage_estimate=12.0, samples=400):
    """Precompute converter efficiency characteristics.

    Args:
        converter (LVDCDC): Converter model instance.
        lv_voltage_estimate (float): Representative LV voltage in volts.
        samples (int): Number of sample points across the power range.

    Returns:
        dict: Sample arrays and derived metrics (peak efficiency power, minimum efficient power (90%), peak eta).
    """
    lv_voltage = max(lv_voltage_estimate, 1.0)
    p_samples = np.linspace(0, converter.max_power, samples)
    eta_samples = []
    for p in p_samples:
        i = p / lv_voltage if lv_voltage > 0 else 0
        losses = converter.calculate_losses(i)
        p_in = p + losses
        eta_samples.append(p / p_in if p_in > 0 else 0.0)

    eta_samples = np.array(eta_samples)
    if len(eta_samples) == 0:
        return {
            "p_samples": np.array([]),
            "eta_samples": np.array([]),
            "p_peak": 0.0,
            "p_min_eff": 0.0,
            "eta_peak": 0.0
        }

    peak_idx = int(np.argmax(eta_samples))
    p_peak = float(p_samples[peak_idx])
    eta_peak = float(eta_samples[peak_idx])

    eff_threshold = eta_peak * 0.9
    above_threshold = np.where(eta_samples >= eff_threshold)[0]
    if above_threshold.size > 0:
        p_min_eff = float(p_samples[above_threshold[0]])
    else:
        p_min_eff = p_peak

    return {
        "p_samples": p_samples,
        "eta_samples": eta_samples,
        "p_peak": p_peak,
        "p_min_eff": max(p_min_eff, 0.0),
        "eta_peak": eta_peak
    }
       
def strategy_ECMS(battery, converter, time_vec, load_dem_vec, plot_figures=True):
    """
    This function implements an ECMS controller that optimizes power split between
    a battery and DC/DC converter to minimize equivalent fuel consumption while
    maintaining charge-sustaining operation (returning to initial SOC).
    The algorithm uses bisection search to find the optimal equivalence factor (s_opt)
    that balances battery usage with converter efficiency over the entire driving cycle.
        dict | None: Results dictionary with the following keys if convergence succeeds:
            - time (np.ndarray): Time vector
            - load_dem (np.ndarray): Original load demand
            - load_smoothed (list): Converter output power
            - batt_result_dict (dict): Battery telemetry (voltage, current, soc, power, etc.)
            - conv_result_dict (dict): Converter telemetry (efficiency, losses, etc.)
            Returns None if optimization fails.
    Variables:
        target_soc (float): Target state of charge to return to (initial SOC), in %.
        s_min (float): Lower bound for equivalence factor in bisection search.
        s_max (float): Upper bound for equivalence factor in bisection search.
        s_opt (float): Current equivalence factor candidate value.
        tolerance (float): SOC convergence tolerance in %.
        max_iter (int): Maximum number of bisection iterations allowed.
        num_candidates (int): Number of power setpoints evaluated per timestep.
        voltage_epsilon (float): Minimum voltage threshold to prevent division by zero.
        soc_gain (float): Proportional gain for dynamic lambda adjustment based on SOC error.
        lambda_cap (float): Maximum absolute value for lambda to prevent extreme behavior.
        battery_loss_weight (float): Weight for battery resistive losses in cost function.
        low_power_penalty_weight (float): Penalty weight to discourage inefficient low-power operation.
        peak_penalty_weight (float): Penalty weight to encourage operation near peak efficiency point.
        conv_eff_info (dict): Converter efficiency characteristics from analysis.
        p_min_eff (float): Minimum power for 90% of peak converter efficiency.
        preferred_power (float): Power setpoint at peak converter efficiency.
        p_active_floor (float): Lower bound for active power candidates.
        active_candidates (np.ndarray): Power candidate values above minimum efficiency threshold.
        p_dcdc_candidates_template (np.ndarray): Template array of converter power candidates including zero.
        best_results (tuple | None): Best simulation results meeting SOC tolerance.
        best_soc_diff (float): Smallest SOC deviation achieved (currently unused).
    Notes:
        - The cost function includes battery losses, converter losses, equivalence cost,
          and penalties for inefficient operating regions.
        - Bisection search adjusts s_opt to achieve charge-sustaining operation.
        - Each iteration runs a full simulation with a fresh battery state.

    Args:
        battery (BattModel): Battery model instance used for energy balancing.
        converter (LVDCDC): Converter supplying the LV bus.
        time_vec (np.ndarray): Simulation time vector in seconds.
    load_dem_vec (np.ndarray): LV load demand trace in watts.
    plot_figures (bool): Whether to generate plots internally (default True).

    Returns:
        dict | None: Results dictionary with telemetry if convergence succeeds, else None.
    """

    target_soc = battery.get_soc()  # Target is to return to start SOC.
    s_min = 0.0
    s_max = 2.0
    s_opt = 1.0
    s_min_diff = None
    s_max_diff = None
    tolerance = 0.1 # SOC tolerance in %
    max_iter = 15
    num_candidates = 200 # Number of power candidates to evaluate.
    voltage_epsilon = 1e-3 # Minimum voltage to avoid division by zero.
    soc_gain = 1.5
    lambda_cap = 3.0 # Max lambda value to avoid extreme behavior.
    battery_loss_weight = 1.0
    low_power_penalty_weight = converter.p_fixed * 5.0
    peak_penalty_weight = converter.p_fixed * 3.0

    conv_eff_info = compute_converter_efficiency_info(
        converter,
        lv_voltage_estimate=battery.get_voltage(current=0) or 12.0,
        samples=400
    )
    p_min_eff = conv_eff_info["p_min_eff"] # lower power threshold for 90% efficiency (90% of peak efficiency)
    preferred_power = conv_eff_info["p_peak"] if conv_eff_info["p_peak"] > 0 else converter.max_power * 0.6
    p_active_floor = min(max(preferred_power * 0.9, p_min_eff), converter.max_power)
    preferred_power = min(preferred_power, converter.max_power)

    if converter.max_power > 0:
        active_candidates = np.linspace(p_active_floor, converter.max_power, max(num_candidates - 1, 1))
    else:
        active_candidates = np.zeros(max(num_candidates - 1, 1))

    p_dcdc_candidates_template = np.unique(np.concatenate(([0.0], active_candidates)))
    
    best_results = None
    best_soc_diff = float('inf')

    print("--- Starting ECMS Optimization ---")

    for i in range(max_iter):
        # Reset components for each iteration to the same initial state.
        # The simplest approach is to instantiate a fresh temporary battery copy.
        temp_battery = BattModel(U_min=battery.U_min, U_max=battery.U_max, R_int=battery.R_int, 
                                 capacity_Ah=battery.capacity_C/3600, start_soc=target_soc)
        
        batt_result_dict, conv_result_dict = init_results_dicts()
        N = len(time_vec)
        dt = time_vec[1] - time_vec[0]
        
        # Simulation loop over all samples.
        for ind in range(N):
            P_load = load_dem_vec[ind]
            
            p_dcdc_candidates = p_dcdc_candidates_template
            p_batt_candidates = P_load - p_dcdc_candidates

            soc_fraction = temp_battery.get_soc() / 100.0
            U_oc = temp_battery.get_U_oc(soc=soc_fraction)
            R_int = temp_battery.R_int
            soc_error = (temp_battery.get_soc() - target_soc) / 100.0
            lambda_dynamic = np.clip(s_opt - soc_gain * soc_error, -lambda_cap, lambda_cap)

            discriminant = U_oc**2 - 4 * R_int * p_batt_candidates
            valid_mask = discriminant >= 0
            sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0))
            denom = 2 * R_int if R_int != 0 else np.finfo(float).eps

            with np.errstate(divide='ignore', invalid='ignore'):
                I1 = (U_oc - sqrt_discriminant) / denom
                I2 = (U_oc + sqrt_discriminant) / denom

            i_candidates = np.where(np.abs(I1) < np.abs(I2), I1, I2)
            i_candidates = np.where(valid_mask, i_candidates, np.nan)

            u_term_candidates = U_oc - i_candidates * R_int
            u_term_candidates = np.clip(u_term_candidates, 0, None)
            lv_voltage_candidates = np.maximum(u_term_candidates, voltage_epsilon)

            p_loss_batt = i_candidates**2 * R_int
            i_dcdc = p_dcdc_candidates / lv_voltage_candidates
            p_loss_conv = converter.p_fixed + converter.k_sw * np.abs(i_dcdc) + converter.k_cond * i_dcdc**2
            battery_equiv_cost = lambda_dynamic * p_batt_candidates
            battery_loss_cost = battery_loss_weight * p_loss_batt
            converter_low_power_penalty = np.where(
                (p_dcdc_candidates > 0) & (p_dcdc_candidates < p_min_eff),
                low_power_penalty_weight * (p_min_eff - p_dcdc_candidates) / max(p_min_eff, 1.0),
                0.0
            )
            converter_peak_penalty = np.where(
                p_dcdc_candidates > 0,
                peak_penalty_weight * ((p_dcdc_candidates - preferred_power) / max(preferred_power, 1.0))**2,
                0.0
            )
            """
            total_cost = (
                battery_loss_cost
                + p_loss_conv
                + battery_equiv_cost
                + converter_low_power_penalty
                + converter_peak_penalty
            )
            """
            total_cost = lambda_dynamic * (p_loss_batt + p_batt_candidates) + (p_loss_conv + p_dcdc_candidates)

            total_cost = np.where(valid_mask, total_cost, np.inf)

            if np.all(~np.isfinite(total_cost)):
                best_p_dcdc = float(np.clip(P_load, 0, converter.max_power))
            else:
                best_idx = int(np.argmin(total_cost))
                best_p_dcdc = float(p_dcdc_candidates[best_idx])
            
            p_batt_opt = P_load - best_p_dcdc
            
            # Apply to components.
            curr_batt_dict = temp_battery.update_charge(power=p_batt_opt, dt=dt)
            curr_conv_dict = converter.compute(lv_voltage=curr_batt_dict["voltage"], power=best_p_dcdc)
            
            # Store results
            for key, val in curr_batt_dict.items():
                batt_result_dict[key].append(val)
            for key, val in curr_conv_dict.items():
                conv_result_dict[key].append(val)
        
    # Check final SOC.
        final_soc = temp_battery.get_soc()
        soc_diff = final_soc - target_soc
        
        print(f"Iter {i}: s={s_opt:.4f}, Final SOC={final_soc:.2f}%, Diff={soc_diff:.2f}%")
        
        if abs(soc_diff) < tolerance:
            print("Converged!")
            best_results = (batt_result_dict, conv_result_dict)
            break
        
        # Update s bounds and use linear interpolation on SOC difference to find next candidate.
        if soc_diff > 0:  # SOC too high, encourage battery discharge -> reduce s
            s_max = s_opt
            s_max_diff = soc_diff
        else:  # SOC too low, encourage charging -> increase s
            s_min = s_opt
            s_min_diff = soc_diff

        if s_min_diff is not None and s_max_diff is not None:
            denom = s_max_diff - s_min_diff
            if abs(denom) > 1e-9:
                s_candidate = s_min - s_min_diff * (s_max - s_min) / denom
            else:
                s_candidate = (s_min + s_max) / 2
        else:
            s_candidate = (s_min + s_max) / 2

        s_opt = float(np.clip(s_candidate, min(s_min, s_max), max(s_min, s_max)))
        
        if i == max_iter - 1:
            print("Max iterations reached.")
            best_results = (batt_result_dict, conv_result_dict)

    # Plotting.
    if best_results:
        batt_res, conv_res = best_results
        results = {
            "time" : time_vec,
            "load_dem" : load_dem_vec,
            "load_smoothed" : conv_res["p_out"], 
            "batt_result_dict" : batt_res,
            "conv_result_dict" : conv_res
        }
        results["s_opt"] = s_opt
        if plot_figures:
            plot_results(results, f'ECMS Strategy (s={s_opt:.4f})')
            converter.plot_efficiency_curve(p_vec=conv_res["p_out"], eta_vec=conv_res["efficiency"])
            battery.plot_efficiency_map(power_vec_operation=batt_res["power"], soc_vec_operation=batt_res["soc"])

        return results

    return None

def strategy_base(battery, converter, time_vec, load_dem_vec, plot_figures=True):
    """Baseline strategy that rate-limits the converter and lets the battery fill gaps.

    Args:
        battery (BattModel): Battery model instance.
        converter (LVDCDC): Converter model instance.
        time_vec (np.ndarray): Simulation time vector in seconds.
    load_dem_vec (np.ndarray): LV load demand trace in watts.
    plot_figures (bool): Whether to generate plots internally (default True).

    Returns:
        dict: Results dictionary with telemetry for plotting.
    """
    batt_result_dict, conv_result_dict = init_results_dicts()

    N = len(time_vec)
    dt = time_vec[1] - time_vec[0]
    
    current_dcdc_power = 0.0

    # Perform simulation.
    for ind in range(N):
        load_power = load_dem_vec[ind]
        target_power = load_power

        # Slightly increase target_power if SOC drops below start value and decrease if above via linear dependency.
        soc_deviation = battery.get_soc() - battery.start_soc
        adjustment_factor = 1 - 0.1 * soc_deviation
        target_power *= adjustment_factor
        
        # Limit DC/DC power rate.
        current_dcdc_power = converter.limit_power(target_power, current_dcdc_power, dt)
        
        # Battery covers the remainder relative to actual load.
        batt_power = load_power - current_dcdc_power

        # Battery results.
        curr_batt_dict = battery.update_charge(power = batt_power, dt=dt)
        # Update battery values of current timestep.
        for key, val in curr_batt_dict.items():
            batt_result_dict[key].append(val)

        # Converter.
        curr_conv_dict = converter.compute(lv_voltage = curr_batt_dict["voltage"], power = current_dcdc_power)
        # Update converter values of current timestamp.
        for key, val in curr_conv_dict.items():
            conv_result_dict[key].append(val)

    results = {
        "time" : time_vec,
        "load_dem" : load_dem_vec,
        "load_smoothed" : conv_result_dict["p_out"],  # Use actual DC/DC power for plotting comparison.
        "batt_result_dict" : batt_result_dict,
        "conv_result_dict" : conv_result_dict
    }
    if plot_figures:
        plot_results(results, 'Base Strategy')
        converter.plot_efficiency_curve(p_vec=conv_result_dict["p_out"], eta_vec=conv_result_dict["efficiency"])
        battery.plot_efficiency_map(power_vec_operation=batt_result_dict["power"], soc_vec_operation=batt_result_dict["soc"])

    return results

def init_results_dicts():
    """Create empty telemetry containers for battery and converter results.

    Returns:
        tuple[dict, dict]: Battery results dictionary and converter results dictionary.
    """
    batt_result_dict = {
        "soc" : [],
        "voltage" : [],
        "eta" : [],
        "current": [],
        "power" : [],
        "losses" : []
    }
    conv_result_dict = {
        "lv_voltage": [],
        "p_out": [],
        "efficiency": [],
        "p_in": [],
        "i_in": [],
        "losses": []
    }
    return batt_result_dict, conv_result_dict


def compute_energy_metrics(results):
    """Calculate aggregate energy demand and losses for a strategy run.

    Args:
        results (dict): Strategy results dictionary.

    Returns:
        dict: Load energy and loss figures in watt-hours.
    """
    time = results["time"]
    if len(time) < 2:
        return {
            "load_energy_wh": 0.0,
            "battery_losses_wh": 0.0,
            "converter_losses_wh": 0.0,
            "total_losses_wh": 0.0,
        }

    dt = float(np.mean(np.diff(time)))
    load_power = results["load_dem"]
    batt_losses = np.array(results["batt_result_dict"]["losses"])
    conv_losses = np.array(results["conv_result_dict"]["losses"])

    load_energy_wh = np.sum(load_power) * dt / 3600.0
    battery_losses_wh = np.sum(batt_losses) * dt / 3600.0
    converter_losses_wh = np.sum(conv_losses) * dt / 3600.0

    return {
        "load_energy_wh": load_energy_wh,
        "battery_losses_wh": battery_losses_wh,
        "converter_losses_wh": converter_losses_wh,
        "total_losses_wh": battery_losses_wh + converter_losses_wh,
    }


def _pad_limits(min_val, max_val, pad_ratio=0.05, min_span=1e-3):
    if min_val is None or max_val is None:
        return None
    span = max(max_val - min_val, min_span)
    pad = span * pad_ratio
    return (min_val - pad, max_val + pad)


def compute_timeseries_axis_limits(results_list):
    """Derive shared axis limits for time-series plots across strategies."""
    time_min = float('inf')
    time_max = float('-inf')
    power_vals = []
    soc_vals = []
    loss_vals = []
    voltage_vals = []
    energy_vals = []

    for res in results_list:
        if not res:
            continue
        time = np.asarray(res.get("time", []))
        if time.size == 0:
            continue
        time_min = min(time_min, float(time[0]))
        time_max = max(time_max, float(time[-1]))

        load = np.asarray(res.get("load_dem", []))
        conv_power = np.asarray(res["conv_result_dict"].get("p_out", []))
        batt_power = np.asarray(res["batt_result_dict"].get("power", []))
        if load.size:
            power_vals.extend([float(load.min()), float(load.max())])
        if conv_power.size:
            power_vals.extend([float(conv_power.min()), float(conv_power.max())])
        if batt_power.size:
            power_vals.extend([float(batt_power.min()), float(batt_power.max())])

        soc = np.asarray(res["batt_result_dict"].get("soc", []))
        if soc.size:
            soc_vals.extend([float(soc.min()), float(soc.max())])

        batt_losses = np.asarray(res["batt_result_dict"].get("losses", []))
        conv_losses = np.asarray(res["conv_result_dict"].get("losses", []))
        if batt_losses.size:
            loss_vals.extend([float(batt_losses.min()), float(batt_losses.max())])
        if conv_losses.size:
            loss_vals.extend([float(conv_losses.min()), float(conv_losses.max())])

        voltage = np.asarray(res["batt_result_dict"].get("voltage", []))
        if voltage.size:
            voltage_vals.extend([float(voltage.min()), float(voltage.max())])

        # Energy traces
        if time.size >= 2:
            dt = float(np.mean(np.diff(time)))
            batt_loss_wh = np.cumsum(batt_losses) * dt / 3600 if batt_losses.size else np.array([])
            conv_loss_wh = np.cumsum(conv_losses) * dt / 3600 if conv_losses.size else np.array([])
            total_loss_wh = batt_loss_wh + conv_loss_wh if batt_loss_wh.size and conv_loss_wh.size else np.array([])
            for arr in (batt_loss_wh, conv_loss_wh, total_loss_wh):
                if arr.size:
                    energy_vals.extend([float(arr.min()), float(arr.max())])

    limits = {}
    if time_min < time_max:
        limits["time"] = (time_min, time_max)
    if power_vals:
        limits["power"] = _pad_limits(min(power_vals), max(power_vals))
    if soc_vals:
        limits["soc"] = _pad_limits(min(soc_vals), max(soc_vals))
    if loss_vals:
        limits["losses"] = _pad_limits(min(loss_vals), max(loss_vals))
    if voltage_vals:
        limits["voltage"] = _pad_limits(min(voltage_vals), max(voltage_vals))
    if energy_vals:
        limits["energy"] = _pad_limits(min(energy_vals), max(energy_vals))

    return limits


def compute_converter_axis_limits(results_list):
    """Derive shared axis limits for converter efficiency plots."""
    power_vals = []
    for res in results_list:
        if not res:
            continue
        p_out = np.asarray(res["conv_result_dict"].get("p_out", []))
        if p_out.size:
            power_vals.extend([float(p_out.min()), float(p_out.max())])

    limits = {
        "efficiency": (0.0, 100.0),
        "hist": (0.0, 100.0)
    }
    if power_vals:
        limits["power"] = _pad_limits(min(power_vals + [0.0]), max(power_vals + [0.0]))
    else:
        limits["power"] = (0.0, 1.0)
    return limits


def compute_battery_power_limits(results_list):
    """Return shared power limits for battery efficiency maps."""
    power_vals = []
    for res in results_list:
        if not res:
            continue
        batt_power = np.asarray(res["batt_result_dict"].get("power", []))
        if batt_power.size:
            power_vals.extend([float(batt_power.min()), float(batt_power.max())])
    if power_vals:
        return _pad_limits(min(power_vals), max(power_vals))
    return None

def main():
    # Create components.
    battery = BattModel(U_min=10.5, U_max=12.6, R_int=0.005, capacity_Ah=70, start_soc=80)
    converter = LVDCDC(hv_voltage=400, p_fixed=25.0, k_sw=0.1, k_cond=0.003, max_power=4000, max_power_rate=500.0)
    
    # Fetch scenario.
    time_vec, load_dem_vec, _ = get_load_scenario(1)

    base_results = strategy_base(
        battery = copy.deepcopy(battery),
        converter = copy.deepcopy(converter),
        time_vec = time_vec,
        load_dem_vec = load_dem_vec,
        plot_figures = False
        )
    ecms_results = strategy_ECMS(
        battery = copy.deepcopy(battery),
        converter = copy.deepcopy(converter),
        time_vec = time_vec,
        load_dem_vec = load_dem_vec,
        plot_figures = False
        )

    if base_results and ecms_results:
        base_metrics = compute_energy_metrics(base_results)
        ecms_metrics = compute_energy_metrics(ecms_results)

        print("\n=== Strategy Energy Summary ===")
        print(
            "Base strategy: Load energy = {:.2f} Wh, Total losses = {:.2f} Wh".format(
                base_metrics["load_energy_wh"], base_metrics["total_losses_wh"]
            )
        )
        print(
            "ECMS strategy: Load energy = {:.2f} Wh, Total losses = {:.2f} Wh".format(
                ecms_metrics["load_energy_wh"], ecms_metrics["total_losses_wh"]
            )
        )

        loss_savings = base_metrics["total_losses_wh"] - ecms_metrics["total_losses_wh"]
        print("ECMS total-loss savings vs. base: {:.2f} Wh".format(loss_savings))

        # percentage savings
        if base_metrics["total_losses_wh"] > 0:
            percent_savings = (loss_savings / base_metrics["total_losses_wh"]) * 100.0
            print("ECMS percentage savings vs. base: {:.2f} %".format(percent_savings))

    # Derive shared axis limits for plotting and render figures for each strategy.
    results_list = [res for res in (base_results, ecms_results) if res]
    axis_limits = compute_timeseries_axis_limits(results_list)
    converter_limits = compute_converter_axis_limits(results_list)
    battery_power_limits = compute_battery_power_limits(results_list)

    labeled_results = []
    if base_results:
        labeled_results.append(("Base Strategy", base_results))
    if ecms_results:
        s_val = ecms_results.get("s_opt", None)
        if s_val is not None:
            label = f"ECMS Strategy (s={s_val:.4f})"
        else:
            label = "ECMS Strategy"
        labeled_results.append((label, ecms_results))

    for label, res in labeled_results:
        plot_results(res, label, axis_limits=axis_limits)
        converter.plot_efficiency_curve(
            p_vec=res["conv_result_dict"]["p_out"],
            eta_vec=res["conv_result_dict"]["efficiency"],
            limits=converter_limits
        )
        battery.plot_efficiency_map(
            power_vec_operation=res["batt_result_dict"]["power"],
            soc_vec_operation=res["batt_result_dict"]["soc"],
            power_limits=battery_power_limits
        )

    plt.show()

    return base_results, ecms_results

if __name__ == "__main__":
    main()

