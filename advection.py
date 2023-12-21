import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib.animation import FuncAnimation

# Parameters
L = 1.0  # Length of the domain
T = 1.0  # Total simulation time
Nx = 64  # Number of spatial points
Nt = 100  # Number of time steps
u = 0.1  # Advection velocity
D = 0.01  # Diffusion coefficient

# Discretization
dx = L / Nx
dt = T / Nt
x = np.linspace(0, L, Nx, endpoint=False)
k = 2 * np.pi / L * np.fft.fftfreq(Nx, d=dx)

# Initial condition
u0 = np.cos(2*x)    

# Pseudo-spectral method
u_hat = np.fft.fft(u0)

# Create a figure and axis for plotting
fig, ax = plt.subplots()
line, = ax.plot(x, u0, label='Initial Condition')
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')
ax.legend()

# Initialize simulation time
current_time = 0.0

# Text object for the title
title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')

# Update function for animation
def update(frame):
    global u, u_hat, current_time
    # Solve advection-diffusion equation in Fourier space
    u_hat = u_hat * np.exp(-1j * k * u * dt) * np.exp(-D * k**2 * dt)

    # Transform back to physical space
    u = ifft(u_hat).real

    # Update the plot data
    line.set_ydata(u)

    # Update the simulation time
    current_time += dt
    title_text.set_text(f'Simulation Time: {current_time:.2f}s')

    return line, title_text

# Create the animation with a shorter pause duration
animation = FuncAnimation(fig, update, frames=Nt, blit=False)

# Display the animation
plt.show()
