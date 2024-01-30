import numpy as np
from scipy.fftpack import fft, ifft
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import matplotlib.pyplot as plt
import timeit
import time


# Parameters
v = 0.1 # Diffusion Term 
v0 = 1 # Comvection Term
t_final = 2
Nt = 201 # Number of Time Points
length = 2*np.pi
Nx = 64 # Number of x points

# Discretise domain
assert(np.mod(Nx,2))==0
t = np.linspace(0,t_final, Nt +1)
xmesh = np.linspace(0, length, Nx+1)
dx = length/Nx
xmesh = xmesh[0:-1]
dt = t[1]-t[0]

def initial_condition(x):
    return np.cos(3*x)
u0 = initial_condition(xmesh)
u0_hat = fft(u0)

# Spectral Differential Matrix
D_1 = np.arange(0, Nx/2+1)
D_2 = np.arange(-Nx/2+1, 0)
D_3 = np.concatenate((D_1, D_2), axis=None)
D = np.diag(1j*D_3)  # Spectral Differentiation Matrix


# Solving the ODE in spectral space
def dudt(t, u):
    du_dt = v * D @ D @ u  # - v0 * D @ u
    return du_dt
sol_spectral = solve_ivp(dudt, t_span=(0, t_final), y0=u0_hat, t_eval=t)
y_spectral = np.real(ifft(sol_spectral.y[:,-1]))

time_spectral = timeit.timeit(lambda: solve_ivp(dudt, t_span=(0, t_final), y0=u0_hat, t_eval=t), number=1)
t0 = time.time()
# Solving 1D Heat equation with Explicit Finite Difference Scheme
y_FD = np.zeros((Nx, Nt))
y_FD[:,0] = u0 
def explicit_FD(y_FD, v, dt, dx, Nt):
    for j in range(1,Nt):
        for i in range(Nx-1):
            y_FD[i, j] = y_FD[i,j-1] + v * dt / dx**2 * (y_FD[i+1,j-1] - 2 * y_FD[i,j-1] + y_FD[i-1,j-1])  # Explicit Finite Difference Technique
            y_FD[0,j] = y_FD[0,j-1] + v * dt / dx**2 * (y_FD[1,j-1] - 2 * y_FD[0,j-1] + y_FD[-1,j-1])  # Explicit Finite Difference Technique
            y_FD[-1,j] = y_FD[-1,j-1] + v * dt / dx**2 * (y_FD[0,j-1] - 2 * y_FD[-1,j-1] + y_FD[-2,j-1])  # Explicit Finite Difference Technique
    return y_FD[:,-1]
y = explicit_FD(y_FD, v, dt, dx, Nt)
t1 = time.time()
total = t1-t0
print(f"Execution time: {time_spectral} seconds")
print('FD time = '+str(total))
plt.plot(xmesh,y_spectral)
plt.plot(xmesh,y)
plt.plot(xmesh, u0, '--')
plt.ylim(-1.1,1.1)
plt.xlim(0, length)
plt.show()