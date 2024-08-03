import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
from matplotlib.animation import FuncAnimation

# Constants
N = 200  # Number of lattice points
dx = 1.0  # Lattice spacing
dt = 0.01  # Time step
lambda_ = 1.0  # Coupling constant
num_steps = 2000  # Number of time steps

# Initial conditions
phi = np.zeros(N)
pi = np.zeros(N)

# Function to initialize kinks
def initialize_kinks():
    global phi
    # Two kinks initially placed at different positions
    for i in range(N):
        if i < N // 4:
            phi[i] = np.tanh((i - N // 8) * np.sqrt(lambda_ / 2))
        elif i > 3 * N // 4:
            phi[i] = -np.tanh((i - 7 * N // 8) * np.sqrt(lambda_ / 2))

# Function to compute the next step using finite difference method
def step(phi, pi):
    phi_new = np.zeros_like(phi)
    pi_new = np.zeros_like(pi)
    
    for i in range(1, N-1):
        laplacian = (phi[i+1] - 2 * phi[i] + phi[i-1]) / (dx * dx)
        pi_new[i] = pi[i] + dt * (laplacian - lambda_ * phi[i]**3)
    
    for i in range(N):
        phi_new[i] = phi[i] + dt * pi_new[i]
    
    return phi_new, pi_new

# Initialize kinks
initialize_kinks()

# Prepare for animation
fig, ax = plt.subplots()
line, = ax.plot(phi)
ax.set_xlim(0, N)
ax.set_ylim(-2, 2)

# Function to update the animation
def update(frame):
    global phi, pi
    phi, pi = step(phi, pi)
    line.set_ydata(phi)
    return line,

# Run animation
ani = FuncAnimation(fig, update, frames=num_steps, blit=True, interval=20)
plt.show()
