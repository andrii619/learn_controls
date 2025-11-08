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
DEADBAND_M_3: float = 0   # Hysteresis band to prevent chattering

# Realistic delay and noise parameters
SENSOR_DELAY_S: float = 0.3  # ADC conversion + signal filtering delay
SENSOR_JITTER_S: float = 0.1  # Random variation in sensor timing (std dev)
SENSOR_NOISE_M_3: float = 0.5  # Measurement noise (std dev)
CONTROLLER_DELAY_S: float = 0.15  # RTOS task scheduling + computation delay
CONTROLLER_JITTER_S: float = 0.05  # Variation in controller execution time
ACTUATOR_DELAY_S: float = 0.25  # DAC output + physical actuator response
ACTUATOR_JITTER_S: float = 0.08  # Variation in actuator response
FLOW_DISTURBANCE_PERCENT: float = 5.0  # Random variation in flow rates (%)


class DelayBuffer:
    """
    Simulates signal propagation delay in a real system.
    Stores historical values and returns delayed values with optional jitter.
    """
    def __init__(self, delay_s: float, jitter_s: float, dt: float, initial_value: float = 0):
        """
        Args:
            delay_s: Base delay in seconds
            jitter_s: Random jitter standard deviation in seconds
            dt: Simulation timestep
            initial_value: Initial value to fill the buffer
        """
        self.base_delay = delay_s
        self.jitter = jitter_s
        self.dt = dt
        # Buffer size: enough to store delay + 3*jitter worth of samples
        buffer_time = delay_s + 3 * jitter_s
        self.buffer_size = max(1, int(np.ceil(buffer_time / dt)))
        # Initialize buffer with initial value
        self.buffer = [initial_value] * self.buffer_size
        self.current_index = 0

    def push(self, value: float) -> None:
        """Add a new value to the buffer (most recent)"""
        self.buffer[self.current_index] = value
        self.current_index = (self.current_index + 1) % self.buffer_size

    def get_delayed(self) -> float:
        """Get the delayed value with jitter"""
        # Add random jitter to the delay
        actual_delay = max(0, self.base_delay + np.random.normal(0, self.jitter))
        # Calculate how many steps back to look
        steps_back = int(np.round(actual_delay / self.dt))
        steps_back = min(steps_back, self.buffer_size - 1)
        # Get the delayed value
        delayed_index = (self.current_index - steps_back - 1) % self.buffer_size
        return self.buffer[delayed_index]

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
        # Add random disturbances to flow rates (simulating real-world variations)
        inflow_disturbance = 1.0 + np.random.normal(0, FLOW_DISTURBANCE_PERCENT / 100)
        outflow_disturbance = 1.0 + np.random.normal(0, FLOW_DISTURBANCE_PERCENT / 100)

        # Water inflow (only when controller is on) with disturbance
        inflow = WATER_FILL_RATE_M_3_PER_SEC * control_signal * dt * inflow_disturbance

        # Water outflow (constant loss) with disturbance
        outflow = WATER_LOSS_RATE_M_3_PER_SEC * dt * outflow_disturbance

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

    # Initialize delay buffers for realistic signal propagation
    # These simulate the delays in a real embedded system
    sensor_buffer = DelayBuffer(SENSOR_DELAY_S, SENSOR_JITTER_S, DT_STEP_S, initial_value=50)
    controller_buffer = DelayBuffer(CONTROLLER_DELAY_S, CONTROLLER_JITTER_S, DT_STEP_S, initial_value=0)
    actuator_buffer = DelayBuffer(ACTUATOR_DELAY_S, ACTUATOR_JITTER_S, DT_STEP_S, initial_value=0)

    # Simulation arrays
    time_steps = int(SIMULATION_DURATION_S / DT_STEP_S)
    time_array = np.linspace(0, SIMULATION_DURATION_S, time_steps)
    water_levels = np.zeros(time_steps)  # Actual water level
    measured_levels = np.zeros(time_steps)  # What the sensor reads
    control_outputs = np.zeros(time_steps)  # Controller output
    actuator_states = np.zeros(time_steps)  # Actual actuator state
    overflow_amounts = np.zeros(time_steps)

    # Run simulation
    for i in range(time_steps):
        # STEP 1: Sensor measures actual water level (with delay and noise)
        sensor_buffer.push(reservoir.volume)
        measured_level = sensor_buffer.get_delayed()
        # Add sensor noise (measurement uncertainty)
        measured_level += np.random.normal(0, SENSOR_NOISE_M_3)

        # STEP 2: Controller processes measurement (with computational delay)
        control_signal = controller.update(measured_level)
        controller_buffer.push(control_signal)
        delayed_control = controller_buffer.get_delayed()

        # STEP 3: Actuator responds to control signal (with physical delay)
        actuator_buffer.push(delayed_control)
        actual_actuator_state = int(actuator_buffer.get_delayed())

        # STEP 4: Physical system responds to actuator
        overflow = reservoir.update(actual_actuator_state, DT_STEP_S)

        # Store results for visualization
        water_levels[i] = reservoir.volume
        measured_levels[i] = measured_level
        control_outputs[i] = control_signal
        actuator_states[i] = actual_actuator_state
        overflow_amounts[i] = overflow

    return time_array, water_levels, measured_levels, control_outputs, actuator_states, overflow_amounts


def plot_results(time_array, water_levels, measured_levels, control_outputs, actuator_states, overflow_amounts):
    """Create nice plots of simulation results with delay visualization"""

    # Set up the figure with subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('Water Reservoir On/Off Control Simulation with Realistic Delays',
                 fontsize=16, fontweight='bold')

    # Plot 1: Water level (actual vs measured)
    ax1.plot(time_array, water_levels, 'b-', linewidth=2, label='Actual Water Level', alpha=0.8)
    ax1.plot(time_array, measured_levels, 'c--', linewidth=1.5, label='Measured Level (with delay & noise)', alpha=0.7)
    ax1.axhline(y=SETPOINT_M_3, color='r', linestyle='--', alpha=0.7, label=f'Setpoint ({SETPOINT_M_3} m³)')
    if DEADBAND_M_3 > 0:
        ax1.axhline(y=SETPOINT_M_3 + DEADBAND_M_3/2, color='orange', linestyle=':', alpha=0.7, label='Upper Threshold')
        ax1.axhline(y=SETPOINT_M_3 - DEADBAND_M_3/2, color='orange', linestyle=':', alpha=0.7, label='Lower Threshold')
        ax1.fill_between(time_array,
                         SETPOINT_M_3 - DEADBAND_M_3/2,
                         SETPOINT_M_3 + DEADBAND_M_3/2,
                         alpha=0.2, color='yellow', label='Deadband')
    ax1.set_ylabel('Water Level (m³)')
    ax1.set_title('Water Level: Actual vs Measured (showing sensor delay and noise)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_ylim(0, MAX_WATER_LEVEL_M_3 * 1.1)

    # Plot 2: Control signals (controller output vs actual actuator)
    ax2.fill_between(time_array, 0, control_outputs, step='post', alpha=0.5, color='green', label='Controller Output')
    ax2.plot(time_array, control_outputs, 'g-', linewidth=1.5, drawstyle='steps-post', label='Controller Command')
    ax2.plot(time_array, actuator_states, 'r-', linewidth=2, drawstyle='steps-post', label='Actual Actuator State', alpha=0.7)
    ax2.set_ylabel('Control Signal (0=Off, 1=On)')
    ax2.set_title('Control Signals: Controller Output vs Actual Actuator (showing delays)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    # Plot 3: Timing visualization (shows all delays)
    ax3.plot(time_array, water_levels / MAX_WATER_LEVEL_M_3, 'b-', linewidth=1, label='Actual Level (normalized)', alpha=0.6)
    ax3.plot(time_array, measured_levels / MAX_WATER_LEVEL_M_3, 'c--', linewidth=1, label='Measured (sensor delay)', alpha=0.6)
    ax3.plot(time_array, control_outputs, 'g-', linewidth=1, drawstyle='steps-post', label='Controller output', alpha=0.6)
    ax3.plot(time_array, actuator_states, 'r-', linewidth=2, drawstyle='steps-post', label='Actuator state', alpha=0.8)
    ax3.set_ylabel('Normalized Signals')
    ax3.set_title(f'Delay Chain: Total delay ≈ {SENSOR_DELAY_S + CONTROLLER_DELAY_S + ACTUATOR_DELAY_S:.2f}s')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    ax3.set_ylim(-0.1, 1.1)

    # Plot 4: Overflow (if any)
    if np.any(overflow_amounts > 0):
        ax4.fill_between(time_array, 0, overflow_amounts, alpha=0.7, color='red', label='Overflow')
        ax4.plot(time_array, overflow_amounts, 'r-', linewidth=1)
        ax4.set_ylabel('Overflow (m³/s)')
        ax4.set_title('Water Overflow')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Overflow Occurred', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax4.set_ylabel('Overflow (m³/s)')
        ax4.set_title('Water Overflow')

    ax4.set_xlabel('Time (seconds)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

    # Print some statistics
    print("\n=== Simulation Results ===")
    print(f"Initial water level: {water_levels[0]:.1f} m³")
    print(f"Final water level: {water_levels[-1]:.1f} m³")
    print(f"Average water level: {np.mean(water_levels):.1f} m³")
    print(f"Measurement error (RMS): {np.sqrt(np.mean((water_levels - measured_levels)**2)):.2f} m³")
    print(f"\n=== Delay Parameters ===")
    print(f"Sensor delay: {SENSOR_DELAY_S:.3f}s ± {SENSOR_JITTER_S:.3f}s")
    print(f"Controller delay: {CONTROLLER_DELAY_S:.3f}s ± {CONTROLLER_JITTER_S:.3f}s")
    print(f"Actuator delay: {ACTUATOR_DELAY_S:.3f}s ± {ACTUATOR_JITTER_S:.3f}s")
    print(f"Total pipeline delay: ~{SENSOR_DELAY_S + CONTROLLER_DELAY_S + ACTUATOR_DELAY_S:.3f}s")
    print(f"\n=== Control Performance ===")
    print(f"Pump on time: {np.sum(actuator_states) * DT_STEP_S:.1f} seconds ({np.mean(actuator_states)*100:.1f}% duty cycle)")
    print(f"Controller switching events: {np.sum(np.abs(np.diff(control_outputs)))}")
    print(f"Actuator switching events: {np.sum(np.abs(np.diff(actuator_states)))}")
    if np.any(overflow_amounts > 0):
        print(f"Total overflow: {np.sum(overflow_amounts) * DT_STEP_S:.2f} m³")
    else:
        print("No overflow occurred")


def main():
    """Main simulation function"""
    print("Starting water reservoir on/off control simulation...")
    print("This simulation includes realistic delays and disturbances:")
    print(f"  - Sensor measurement delay and noise")
    print(f"  - Controller execution delay")
    print(f"  - Actuator response delay")
    print(f"  - Random flow disturbances")

    # Run simulation
    time_array, water_levels, measured_levels, control_outputs, actuator_states, overflow_amounts = run_simulation()

    # Plot results
    plot_results(time_array, water_levels, measured_levels, control_outputs, actuator_states, overflow_amounts)

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()