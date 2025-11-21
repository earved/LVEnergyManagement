
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class BattModel:
    def __init__(self, U_min, U_max, R_int, capacity_Ah, start_soc=80) -> None:
        """
        Initialize the battery model.
        :param U_min: Minimum open-circuit voltage [V] at 0% SOC
        :param U_max: Maximum open-circuit voltage [V] at 100% SOC
        :param R_int: Internal resistance [Ohm]
        :param capacity_Ah: Battery capacity [Ah]
        """
        self.U_min = U_min
        self.U_max = U_max
        self.R_int = R_int
        self.capacity_C = capacity_Ah * 3600  # Convert Ah to Coulomb
        self.charge_remaining = self.capacity_C * start_soc/100

    def get_soc(self) -> float:
        """Return State of Charge (SOC) in percent."""
        return max(min(self.charge_remaining / self.capacity_C * 100, 100), 0)

    def get_U_oc(self, soc = None) -> float:
        """Open-circuit voltage as a function of SOC (linear approximation)."""
        if soc is None:
            soc = self.get_soc() / 100
        return self.U_min + soc * (self.U_max - self.U_min)

    def get_voltage(self, current=None, power=None, soc=None) -> float:
        """
        Calculate terminal voltage under load.
        Positive current = discharge, negative current = charge.
        :param current: Current [A]
        :param power: Power [W]
        :return: Voltage [V]
        """
        I = self._resolve_current(current, power)
        U_oc = self.get_U_oc(soc=soc)
        return max(U_oc - I * self.R_int, 0)

    def _resolve_current(self, current=None, power=None, soc = None) -> float:
        """Resolve current from either current or power input."""
        if current is not None:
            return current
        elif power is not None:
            U_oc = self.get_U_oc(soc=soc)
            discriminant = U_oc**2 - 4*self.R_int*power
            if discriminant < 0:
                # Power exceeds maximum possible power
                # Return current at max power point (U_oc / 2R)
                return U_oc / (2 * self.R_int)
            
            I1 = (U_oc - discriminant**0.5)/(2*self.R_int)
            I2 = (U_oc + discriminant**0.5)/(2*self.R_int)
            return I1 if abs(I1) < abs(I2) else I2
        else:
            raise ValueError("Provide either current or power.")

    def update_charge(self, dt, current=None, power=None) -> dict:
        """
        Update battery charge for time dt.
        Positive current = discharge, negative current = charge.
        :param dt: Time [s]
        :param current: Current [A]
        :param power: Power [W]
        """
        I = self._resolve_current(current, power)
        self.charge_remaining -= I * dt  # Negative I increases charge
        # Clamp charge between 0 and capacity
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
        """
        Calculate efficiency for charging or discharging.
        For discharge: η = P_load / (P_load + losses)
        For charge: η = energy stored / energy supplied
        :param current: Current [A]
        :param power: Power [W]
        :return: Efficiency [0..1]
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
        """
        Calculate efficiency curve for fixed SOC
        """
        p_values = np.linspace(0,max_load_power,200)
        eta_values =[self.get_efficiency(power=p, soc=soc) for p in p_values]
        V_values = [self.get_voltage(power=p, soc=soc) for p in p_values]
        return p_values, eta_values, V_values

    def get_losses_for_power_dem_vec(self, power_vec=None, current_vec=None, soc=None):
        eta_values =[self.get_efficiency(power=p, soc=soc) for p in power_vec]
        i_values = [self._resolve_current(power=p, soc=soc) for p in power_vec]
        V_values = [self.get_voltage(power=p, soc=soc) for p in power_vec]
        p_loss_values = [i**2*self.R_int for i in i_values]
        return p_loss_values
       
    def plot_efficiency_map(self, power_vec_operation=None, soc_vec_operation=None):
        soc_range = range(101)
        
        # Determine power range
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
        
        # Create grid for contour plot
        p_grid = np.linspace(min_p, max_p, 200)
        soc_grid = np.array(soc_range)
        P, S = np.meshgrid(p_grid, soc_grid)
        
        # Calculate efficiency for the grid
        Z = np.zeros_like(P)
        for i, soc in enumerate(soc_grid):
            for j, p in enumerate(p_grid):
                Z[i, j] = self.get_efficiency(power=p, soc=soc)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Heatmap
        cax = ax.imshow(Z, aspect='auto', origin='lower', 
                           extent=[min_p, max_p, 0, 100], cmap='viridis')
        fig.colorbar(cax, label="Efficiency [0-1]")
        
        # Contour lines
        contours = ax.contour(P, S, Z, levels=10, colors='white', alpha=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

        # Operating points
        if power_vec_operation is not None and soc_vec_operation is not None:
            ax.scatter(power_vec_operation, soc_vec_operation, c='red', s=10, alpha=0.5, label="Operating Points")
            ax.legend()

        ax.set_title("Battery Efficiency Map")
        ax.set_xlabel("Power [W] (Negative = Charging)")
        ax.set_ylabel("SOC [%]")
        return fig


class LVDCDC:
    def __init__(self, hv_voltage, p_fixed=15.0, k_sw=0.1, k_cond=0.002, max_power=4000, max_power_rate=500.0) -> None:
        """
        Initialize DC/DC converter model with physics-based loss model.
        :param hv_voltage: High-voltage side voltage [V]
        :param p_fixed: Constant losses [W]
        :param k_sw: Switching loss coefficient [W/A]
        :param k_cond: Conduction loss coefficient [W/A^2]
        :param max_power: Maximum power [W]
        :param max_power_rate: Maximum power change rate [W/s]
        """
        self.hv_voltage = hv_voltage
        self.p_fixed = p_fixed
        self.k_sw = k_sw
        self.k_cond = k_cond
        self.max_power = max_power
        self.max_power_rate = max_power_rate

    def calculate_losses(self, current) -> float:
        """
        Calculate total losses based on current.
        P_loss = P_fixed + k_sw * |I| + k_cond * I^2
        """
        return self.p_fixed + self.k_sw * abs(current) + self.k_cond * current**2

    def limit_power(self, target_power, current_power, dt) -> float:
        """
        Calculate the next power output limited by max_power_rate.
        """
        delta_p = target_power - current_power
        max_delta = self.max_power_rate * dt
        
        if abs(delta_p) <= max_delta:
            return target_power
        else:
            return current_power + np.sign(delta_p) * max_delta

    def compute(self, lv_voltage, current=None, power=None) -> dict:
        """
        Compute converter performance for given LV load and LV voltage.
        :param lv_voltage: LV side voltage [V]
        :param current: LV side current [A]
        :param power: LV side power [W]
        :return: dict with efficiency, input power, input current, losses
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
        """
        Plot efficiency vs load power using actual simulation results.
        Also plots a reference curve at fixed voltage.
        """
        # Convert efficiency to percentage
        eta_percent = [e * 100 for e in eta_vec]

        # Calculate reference curve
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
    """results = {
        "time" : time_vec,
        "load_dem" : load_dem_vec,
        "load_smoothed" : load_dem_vec_smoothed,
        "batt_result_dict" : batt_result_dict,
        "conv_result_dict" : conv_result_dict
    }
    """
   
    batt_result_dict = results["batt_result_dict"]
    conv_result_dict = results["conv_result_dict"]


    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
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
    axs[3].set_xlabel("Time [s]")
    axs[3].legend(loc='upper right')
    axs[3].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig

def get_load_scenario(ind=1):
    """
    return LV power demand over time
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
       
def strategy_ECMS():
    # STRATEGY 2
    # 1. get current load demand
    # 2. calculate losses for batt and DCDC for all power-splits and use ECMS to determine load share
    # 3. get battery and DCDC values
    # 4. do same plotting
    pass

def strategy_dumb(battery, converter, time_vec, load_dem_vec):
    # STRATEGY 1 (dumb with rate limiting)
    # 1. get current load demand
    # 2. determine DCDC power with rate limiting
    # 3. determine Battery power as residual
    # 4. plot(losses, voltage, currents) over time
    batt_result_dict, conv_result_dict = init_results_dicts()

    N = len(time_vec)
    dt = time_vec[1] - time_vec[0]
    
    current_dcdc_power = 0.0

    # perform simulation
    for ind in range(N):
        target_power = load_dem_vec[ind]
        
        # Limit DCDC power rate
        current_dcdc_power = converter.limit_power(target_power, current_dcdc_power, dt)
        
        # Battery covers the rest
        batt_power = target_power - current_dcdc_power

        # battery results
        curr_batt_dict = battery.update_charge(power = batt_power, dt=dt)
        # update battey values of current timestep
        for key, val in curr_batt_dict.items():
            batt_result_dict[key].append(val)

        # converter
        curr_conv_dict = converter.compute(lv_voltage = curr_batt_dict["voltage"], power = current_dcdc_power)
        # update converter values of current timestamp
        for key, val in curr_conv_dict.items():
            conv_result_dict[key].append(val)

    results = {
        "time" : time_vec,
        "load_dem" : load_dem_vec,
        "load_smoothed" : conv_result_dict["p_out"], # Use actual DCDC power for plotting comparison
        "batt_result_dict" : batt_result_dict,
        "conv_result_dict" : conv_result_dict
    }
    fig1 = plot_results(results, 'Dumb Strategy (Rate Limited)')
    fig2 = converter.plot_efficiency_curve(p_vec=conv_result_dict["p_out"], eta_vec=conv_result_dict["efficiency"])
    fig3 = battery.plot_efficiency_map(power_vec_operation=batt_result_dict["power"], soc_vec_operation=batt_result_dict["soc"])


def init_results_dicts():
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
    # create components
    battery = BattModel(U_min=10.5, U_max=12.6, R_int=0.005, capacity_Ah=70, start_soc=80)
    converter = LVDCDC(hv_voltage=400, p_fixed=15.0, k_sw=0.1, k_cond=0.002, max_power=4000, max_power_rate=500.0)
    
    # fetch scneario
    time_vec, load_dem_vec, _ = get_load_scenario(1)

    strategy_dumb(
        battery = battery,
        converter = converter,
        time_vec = time_vec,
        load_dem_vec = load_dem_vec
        )

    plt.show()