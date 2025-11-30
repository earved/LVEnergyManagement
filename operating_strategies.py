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
       
    def plot_efficiency_map(self, power_vec_operation=None, soc_vec_operation=None):
        """Plot the battery efficiency map with optional operating points.

        Args:
            power_vec_operation (Iterable[float] | None): Historical power samples.
            soc_vec_operation (Iterable[float] | None): Corresponding SOC samples.

        Returns:
            matplotlib.figure.Figure: Figure handle for the contour plot.
        """
        soc_range = range(101)
        
        # Determine power range.
        if power_vec_operation is not None and len(power_vec_operation) > 0:
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

    def plot_efficiency_curve(self, p_vec, eta_vec, voltage=12.0):
        """Plot simulated efficiency points against a reference curve.

        Args:
            p_vec (Iterable[float]): Power samples in watts.
            eta_vec (Iterable[float]): Efficiency samples (0-1).
            voltage (float): Reference LV voltage for the analytic curve.

        Returns:
            matplotlib.figure.Figure: Figure handle for the plot.
        """
        # Convert efficiency to percentage.
        eta_percent = [e * 100 for e in eta_vec]

        # Calculate reference curve.
        p_ref = np.linspace(0, self.max_power, 100)
        eta_ref = []
        for p in p_ref:
            i = p / voltage if voltage > 0 else 0
            loss = self.calculate_losses(i)
            p_in = p + loss
            eta = p / p_in if p_in > 0 else 0
            eta_ref.append(eta * 100)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(p_ref, eta_ref, label=f"Reference Curve (@{voltage}V)", linewidth=2, linestyle='--')
        plt.scatter(p_vec, eta_percent, color='red', alpha=0.5, label="Operating Points", s=10)
        plt.title("DC/DC Converter Efficiency vs. Load Power")
        plt.xlabel("Load Power [W]")
        plt.ylabel("Efficiency [%]")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        return fig

def plot_results(results, plotstring):
    """Render standard plots for converter and battery telemetry.

    Args:
        results (dict): Dictionary containing time, load, battery, and converter traces.
        plotstring (str): Title prefix used for the plot window.

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

    axs[1].plot(results["time"], batt_result_dict["soc"], label="Battery SOC", color='green')
    axs[1].set_ylabel("SOC [%]")
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    axs[2].plot(results["time"], batt_result_dict["losses"], label="Battery Losses")
    axs[2].plot(results["time"], conv_result_dict["losses"], label="DC/DC Losses")
    axs[2].set_ylabel("Power Loss [W]")
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    axs[3].plot(results["time"], batt_result_dict["voltage"], label="Battery Voltage", color='orange')
    axs[3].set_ylabel("Voltage [V]")
    axs[3].legend(loc='upper right')
    axs[3].grid(True, linestyle='--', alpha=0.5)

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
        dict: Sample arrays and derived metrics (peak power, minimum efficient power, peak eta).
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
       
def strategy_ECMS(battery, converter, time_vec, load_dem_vec):
    """Run an equivalent consumption minimization strategy (ECMS).

    Args:
        battery (BattModel): Battery model instance used for energy balancing.
        converter (LVDCDC): Converter supplying the LV bus.
        time_vec (np.ndarray): Simulation time vector in seconds.
        load_dem_vec (np.ndarray): LV load demand trace in watts.

    Returns:
        dict | None: Results dictionary with telemetry if convergence succeeds, else None.
    """

    target_soc = battery.get_soc()  # Target is to return to start SOC.
    s_min = 0.0
    s_max = 2.0
    s_opt = 0.5
    tolerance = 0.1 # SOC tolerance in %
    max_iter = 15
    num_candidates = 200
    voltage_epsilon = 1e-3
    soc_gain = 1.5
    lambda_cap = 3.0
    battery_loss_weight = 1.0
    low_power_penalty_weight = converter.p_fixed * 5.0
    peak_penalty_weight = converter.p_fixed * 3.0

    conv_eff_info = compute_converter_efficiency_info(
        converter,
        lv_voltage_estimate=battery.get_voltage(current=0) or 12.0,
        samples=400
    )
    p_min_eff = conv_eff_info["p_min_eff"]
    preferred_power = conv_eff_info["p_peak"] if conv_eff_info["p_peak"] > 0 else converter.max_power * 0.6
    p_active_floor = min(max(preferred_power * 0.9, p_min_eff), converter.max_power)
    preferred_power = min(preferred_power, converter.max_power)

    if num_candidates < 2:
        num_candidates = 2

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
            total_cost = (
                battery_loss_cost
                + p_loss_conv
                + battery_equiv_cost
                + converter_low_power_penalty
                + converter_peak_penalty
            )
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
        
        # Update s based on whether the battery finished above or below target SOC.
        
        if soc_diff > 0:  # SOC too high, encourage battery discharge -> reduce s
            s_max = s_opt
        else:  # SOC too low, encourage charging -> increase s
            s_min = s_opt

        s_opt = (s_min + s_max) / 2
        
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
        fig1 = plot_results(results, f'ECMS Strategy (s={s_opt:.4f})')
        fig2 = converter.plot_efficiency_curve(p_vec=conv_res["p_out"], eta_vec=conv_res["efficiency"])
        fig3 = battery.plot_efficiency_map(power_vec_operation=batt_res["power"], soc_vec_operation=batt_res["soc"])

        return results

    return None

def strategy_dumb(battery, converter, time_vec, load_dem_vec):
    """Baseline strategy that rate-limits the converter and lets the battery fill gaps.

    Args:
        battery (BattModel): Battery model instance.
        converter (LVDCDC): Converter model instance.
        time_vec (np.ndarray): Simulation time vector in seconds.
        load_dem_vec (np.ndarray): LV load demand trace in watts.

    Returns:
        dict: Results dictionary with telemetry for plotting.
    """
    batt_result_dict, conv_result_dict = init_results_dicts()

    N = len(time_vec)
    dt = time_vec[1] - time_vec[0]
    
    current_dcdc_power = 0.0

    # Perform simulation.
    for ind in range(N):
        target_power = load_dem_vec[ind]
        
        # Limit DC/DC power rate.
        current_dcdc_power = converter.limit_power(target_power, current_dcdc_power, dt)
        
        # Battery covers the remainder.
        batt_power = target_power - current_dcdc_power

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
    fig1 = plot_results(results, 'Dumb Strategy (Rate Limited)')
    fig2 = converter.plot_efficiency_curve(p_vec=conv_result_dict["p_out"], eta_vec=conv_result_dict["efficiency"])
    fig3 = battery.plot_efficiency_map(power_vec_operation=batt_result_dict["power"], soc_vec_operation=batt_result_dict["soc"])

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

if __name__ == "__main__":
    # Create components.
    battery = BattModel(U_min=10.5, U_max=12.6, R_int=0.005, capacity_Ah=70, start_soc=80)
    converter = LVDCDC(hv_voltage=400, p_fixed=25.0, k_sw=0.1, k_cond=0.005, max_power=4000, max_power_rate=500.0)
    
    # Fetch scenario.
    time_vec, load_dem_vec, _ = get_load_scenario(1)

    strategy_dumb(
        battery = copy.deepcopy(battery),
        converter = copy.deepcopy(converter),
        time_vec = time_vec,
        load_dem_vec = load_dem_vec
        )
    if 1:
        strategy_ECMS(
            battery = copy.deepcopy(battery),
            converter = copy.deepcopy(converter),
            time_vec = time_vec,
            load_dem_vec = load_dem_vec
            )

    plt.show()