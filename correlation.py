import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def correlation_function(m, n, t, t_prime, N, a, k_max, m_mass):
    """
    Compute the correlation function <0|u_m(t)u_n(t')|0> for a 1D harmonic crystal.
    """
    k_values = np.linspace(-k_max, k_max, N)
    omega_k = 2 * np.sqrt(np.abs(k_values) / m_mass) * np.sin(k_values * a / 2)
    correlation = 0.0

    for k, omega in zip(k_values, omega_k):
        if omega != 0:
            correlation += (1 / (m_mass * omega)) * np.cos(k * (m - n) * a) * np.cos(omega * (t - t_prime))

    return correlation / N  #

def plot_correlation_function(N, a, k_max, m_mass, t_max, fps, save_interval=1.0):
    distance_range = range(-N//2, N//2)
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], lw=2)
    
    ax.set_xlim(min(distance_range), max(distance_range))
    ax.set_ylim(min_y,max_y)  # Adjust this range based on your values
    ax.set_xlabel('m - n')
    ax.set_ylabel('Correlation')
    ax.grid(True)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Create output directory if it doesn't exist
    output_dir = 'correlation_frames'
    os.makedirs(output_dir, exist_ok=True)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        t = frame / fps
        correlation_values = [
            correlation_function(m, 0, t, 0, N, a, k_max, m_mass)
            for m in distance_range
        ]
        # print(correlation_values)
        line.set_data(distance_range, correlation_values)
        time_text.set_text(f't = {t:.2f}')
        
        # Save frame as PNG at specified intervals
        if png_output and frame % int(save_interval * fps) == 0:
            filename = os.path.join(output_dir, f'correlation_t{t:.2f}.png')
            plt.savefig(filename)
            print(f"Saved frame at t = {t:.2f}")
        
        return line, time_text

    frames = int(t_max * fps)
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=1000/fps)
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
max_y = 1.5e-14
min_y = 1e-14
png_output = False

# Animate the correlation function and save frames
plot_correlation_function(N, a, k_max, m_mass, t_max, fps, save_interval)