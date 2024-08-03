import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def displacement(m, t, N, a, k_max, m_mass, debug=False):
    """
    Compute the displacement u_m(t) for a 1D harmonic crystal.
    """
    k_values = np.linspace(-k_max, k_max, N)
    omega_k = 2 * np.sqrt(np.abs(k_values) / m_mass) * np.sin(np.abs(k_values * a) / 2)
    
    displacement = 0.0
    for k, omega in zip(k_values, omega_k):
        if omega != 0:
            amplitude = np.sqrt(1 / (2 * N * m_mass * omega))
            term = amplitude * np.cos(k * m * a) * np.cos(omega * t)
            displacement += term
            if debug:
                print(f"k={k:.2e}, omega={omega:.2e}, amplitude={amplitude:.2e}, term={term:.2e}")
    
    result = np.real(displacement)
    if debug:
        print(f"m={m}, t={t:.2f}, displacement={result:.2e}")
    return result

def plot_displacement(N, a, k_max, m_mass, t_max, fps, save_interval=1.0):
    atom_positions = np.arange(N) * a
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], 'bo-', lw=1, markersize=3)
    
    ax.set_xlim(0, (N-1) * a)
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Displacement (u)')
    ax.grid(True)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Create output directory if it doesn't exist
    output_dir = 'displacement_frames'
    os.makedirs(output_dir, exist_ok=True)

    # Variables to track the overall min and max displacement
    overall_min = float('inf')
    overall_max = float('-inf')

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        nonlocal overall_min, overall_max
        t = frame / fps
        displacements = [displacement(m, t, N, a, k_max, m_mass, debug=False) for m in range(N)]
        valid_displacements = [d for d in displacements if not np.isnan(d) and not np.isinf(d)]
        
        if valid_displacements:
            frame_min = np.min(valid_displacements)
            frame_max = np.max(valid_displacements)
            overall_min = min(overall_min, frame_min)
            overall_max = max(overall_max, frame_max)
            
            # Dynamically adjust y-axis limits with a larger margin
            y_range = overall_max - overall_min
            y_margin = y_margin_scale * y_range  # Increased from 0.1 to 0.5
            y_min = overall_min - y_margin
            y_max = overall_max + y_margin
            
            # Ensure a minimum vertical range
            if y_max - y_min < 1e-6:  # Adjust this value as needed
                y_center = (y_min + y_max) / 2
                y_min = y_center - 5e-7
                y_max = y_center + 5e-7
            
            ax.set_ylim(y_min, y_max)
            
            print(f"t = {t:.2f}, min = {frame_min:.4e}, max = {frame_max:.4e}")
        else:
            print(f"t = {t:.2f}, all displacements are invalid")
        
        line.set_data(atom_positions, displacements)
        time_text.set_text(f't = {t:.2f}')
        
        # Save frame as PNG at specified intervals
        if frame % int(save_interval * fps) == 0:
            filename = os.path.join(output_dir, f'displacement_t{t:.2f}.png')
            plt.savefig(filename)
            print(f"Saved frame at t = {t:.2f}")
        
        return line, time_text

    frames = int(t_max * fps)
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=1000/fps)
    plt.tight_layout()
    plt.show()

# Parameters
N = 40         # Number of atoms
a = 1.0         # Lattice spacing
k_max = np.pi/a # Maximum wave vector
m_mass = 0.5    # Mass of each atom
t_max = 10.0    # Maximum time for the animation
fps = 30        # Frames per second
save_interval = 0.5  # Save a frame every 0.5 seconds
y_margin_scale = 0.05
# Test the displacement function
print("Testing displacement function:")
test_displacement = displacement(0, 0, N, a, k_max, m_mass)
print(f"Test displacement: {test_displacement}")

# Animate the displacement function and save frames
plot_displacement(N, a, k_max, m_mass, t_max, fps, save_interval)