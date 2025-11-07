import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
MAX_WATER_LEVEL_M_3: float = 100
WATER_LOSS_RATE_M_3_PER_SEC: float = 0.1  # Added some water loss for realism
WATER_FILL_RATE_M_3_PER_SEC: float = 0.534
DT_STEP_S: float = 0.2
SIMULATION_DURATION_S: float = 120  # Extended for better visualization

# Control parameters
SETPOINT_M_3: float = 75  # Target water level
DEADBAND_M_3: float = 5   # Hysteresis band to prevent chattering


# for more realism add random disturbances to inflow and outflow of water

class OnOffController:
    def __init__(self, setpoint: float, deadband: float):
        self.setpoint = setpoint
        self.deadband = deadband
        self.control_output = 0

    def update(self, current_level: float) -> int:
        """
        Update controller output based on current water level
        Returns 1 (on) or 0 (off)
        """
        if current_level <= (self.setpoint - self.deadband/2):
            self.control_output = 1  # Turn on
        elif current_level >= (self.setpoint + self.deadband/2):
            self.control_output = 0  # Turn off
        # Otherwise maintain current state (hysteresis)

        return self.control_output


class WaterReservoir:
    def __init__(self, initial_volume: float = 0, max_capacity: float = MAX_WATER_LEVEL_M_3):
        self.volume = initial_volume
        self.max_capacity = max_capacity

    def update(self, control_signal: int, dt: float) -> float:
        """
        Update reservoir water level based on control signal
        """
        # Water inflow (only when controller is on)
        inflow = WATER_FILL_RATE_M_3_PER_SEC * control_signal * dt

        # Water outflow (constant loss)
        outflow = WATER_LOSS_RATE_M_3_PER_SEC * dt

        # Update volume
        self.volume += inflow - outflow

        # Handle overflow
        if self.volume > self.max_capacity:
            overflow = self.volume - self.max_capacity
            self.volume = self.max_capacity
        else:
            overflow = 0

        # Prevent negative volume
        if self.volume < 0:
            self.volume = 0

        return overflow


def run_simulation():
    """Run the water reservoir simulation with on/off control"""

    # Initialize system
    controller = OnOffController(SETPOINT_M_3, DEADBAND_M_3)
    reservoir = WaterReservoir(initial_volume=50)  # Start at 50% capacity

    # Simulation arrays
    time_steps = int(SIMULATION_DURATION_S / DT_STEP_S)
    time_array = np.linspace(0, SIMULATION_DURATION_S, time_steps)
    water_levels = np.zeros(time_steps)
    control_outputs = np.zeros(time_steps)
    overflow_amounts = np.zeros(time_steps)

    # Run simulation
    for i in range(time_steps):
        # Update controller
        control_signal = controller.update(reservoir.volume)

        # Update reservoir
        overflow = reservoir.update(control_signal, DT_STEP_S)

        # Store results
        water_levels[i] = reservoir.volume
        control_outputs[i] = control_signal
        overflow_amounts[i] = overflow

    return time_array, water_levels, control_outputs, overflow_amounts


def plot_results(time_array, water_levels, control_outputs, overflow_amounts):
    """Create nice plots of simulation results"""

    # Set up the figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Water Reservoir On/Off Control Simulation', fontsize=16, fontweight='bold')

    # Plot 1: Water level
    ax1.plot(time_array, water_levels, 'b-', linewidth=2, label='Water Level')
    ax1.axhline(y=SETPOINT_M_3, color='r', linestyle='--', alpha=0.7, label=f'Setpoint ({SETPOINT_M_3} m³)')
    ax1.axhline(y=SETPOINT_M_3 + DEADBAND_M_3/2, color='orange', linestyle=':', alpha=0.7, label='Upper Threshold')
    ax1.axhline(y=SETPOINT_M_3 - DEADBAND_M_3/2, color='orange', linestyle=':', alpha=0.7, label='Lower Threshold')
    ax1.fill_between(time_array,
                     SETPOINT_M_3 - DEADBAND_M_3/2,
                     SETPOINT_M_3 + DEADBAND_M_3/2,
                     alpha=0.2, color='yellow', label='Deadband')
    ax1.set_ylabel('Water Level (m³)')
    ax1.set_title('Water Level vs Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, MAX_WATER_LEVEL_M_3 * 1.1)

    # Plot 2: Control output
    ax2.fill_between(time_array, 0, control_outputs, step='post', alpha=0.7, color='green', label='Pump Status')
    ax2.plot(time_array, control_outputs, 'g-', linewidth=1, drawstyle='steps-post')
    ax2.set_ylabel('Control Output (0=Off, 1=On)')
    ax2.set_title('Pump Control Signal')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Overflow (if any)
    if np.any(overflow_amounts > 0):
        ax3.fill_between(time_array, 0, overflow_amounts, alpha=0.7, color='red', label='Overflow')
        ax3.plot(time_array, overflow_amounts, 'r-', linewidth=1)
        ax3.set_ylabel('Overflow (m³/s)')
        ax3.set_title('Water Overflow')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No Overflow Occurred', transform=ax3.transAxes,
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax3.set_ylabel('Overflow (m³/s)')
        ax3.set_title('Water Overflow')

    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

    # Print some statistics
    print("\n=== Simulation Results ===")
    print(f"Initial water level: {water_levels[0]:.1f} m³")
    print(f"Final water level: {water_levels[-1]:.1f} m³")
    print(f"Average water level: {np.mean(water_levels):.1f} m³")
    print(f"Pump on time: {np.sum(control_outputs) * DT_STEP_S:.1f} seconds ({np.mean(control_outputs)*100:.1f}% duty cycle)")
    if np.any(overflow_amounts > 0):
        print(f"Total overflow: {np.sum(overflow_amounts) * DT_STEP_S:.2f} m³")
    else:
        print("No overflow occurred")


def main():
    """Main simulation function"""
    print("Starting water reservoir on/off control simulation...")

    # Run simulation
    time_array, water_levels, control_outputs, overflow_amounts = run_simulation()

    # Plot results
    plot_results(time_array, water_levels, control_outputs, overflow_amounts)

    print("Simulation complete!")


if __name__ == "__main__":
    main()