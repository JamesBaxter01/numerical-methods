""" This code uses the second order central difference numerical method to solve the 2D Incompressible Navier-Stokes equations for the 
    Lid Driven Cavity CFD problem. The 2D Navier-Stokes Equations are given by:                                                   
    """ 
        #Momentum Equation:             ∂u/∂t + (u·∇)u = -1/⍴ ∇p + µ∇²u

        #Incompressibility Equation:    ∇·u = 0

""" The divergence of the Momentum Equation is taken, and then the incompressibility constraint is applied. This results in
    the pressure Poisson Equation, and the system of differential equations can be seen below:
    """       
        #Momentum Equation:             ∂u/∂t + (∂u/∂x) + v(∂u/∂y) = -1/⍴(∂p/∂x + ∂p/∂y) + µ(∂²u/∂x² + ∂²u/∂y²)
        #Momentum Equation:             ∂v/∂t + v(∂v/∂x) + v(∂v/∂y) = -1/⍴(∂p/∂y) + µ(∂²v/∂x² + ∂²v/∂y²)
        
        #Pressure Poisson Equation:     ∂²p/∂x² + ∂²p/∂y² = -⍴((∂u/∂x)² + 2(∂u/∂y)(∂v/∂x) + (∂v/∂y)²)

""" A Finite Difference discretisation scheme is used to discretise the momentum equation, and a second order central difference
    approximation is used for the pressure Poisson equation, these approximations are given by:
"""     
        #                               ∂u/∂t = (u[i,j,n+1] - u[i,j,n])/∆t

        #                               ∂u/∂x = (u[i,j,n] - u[i-1,j,n])/∆x
        #                               ∂²u/∂x² = (u[i+1,j,n] - 2u(i,j,n) + u[i-1,j,n])/∆x²

        #                               ∂v/∂y = (v[i,j,n] - v[i,j-1,n])/∆y
        #                               ∂²v/∂y² = (v[i,j+1,n] - 2v(i,j,n) + v[i,j-1,n])/∆y²

        #                               ∂²p/∂x² = (p[i+1,j,n] - 2p[i,j,n] + p[i-1,j,n])/∆x²
        #                               ∂²p/∂y² = (p[i,j-1,n] - 2p[i,j,n] + p[i,j-1,n])/∆y²

""" This code is inspired by https://nbviewer.org/github/barbagroup/CFDPython/blob/master/lessons/14_Step_11.ipynb, and implementation 
    of the methodology has been done by myself for learning purposes.

    Running this code displays a contour plot of the velocity magnitude, however this can be changed, along with all initial conditions.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Domain Discretisation
NODES = 41
DOMAIN_LENGTH = 1.0
dx = (DOMAIN_LENGTH/(NODES-1))
dy = (DOMAIN_LENGTH/(NODES-1))
x = np.linspace(0, DOMAIN_LENGTH, NODES)
y = np.linspace(0, DOMAIN_LENGTH, NODES)
X, Y = np.meshgrid(x, y)

dt = 0.001                      # Time step size
LID_VELOCITY = 1.0             
µ = 0.005                       # Dynamic Viscosity
rho = 1.0                      # Density
nt = 10000                      # Number of Time Steps
T_FINAL = nt*dt
P_ITERATIONS = 50              # Iterations used to solve Pressure Poisson equation, Ideally more than 5

def main():
    
    def pressure_poisson(rho, u, v, p, dx, dy, dt):
        pn = np.empty_like(p)
        
    
        for q in range(P_ITERATIONS):
            pn = p.copy()

            # Discretised Pressure Poisson Equation 
            p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /(2 * (dx**2 + dy**2)) - rho * dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * ((1/dt)*((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx) +(v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy)) - ((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx))**2 - 2*((u[2:, 1:-1]-u[0:-2, 1:-1])/(2*dy) * (v[1:-1, 2:] - v[1:-1, 0:-2])/(2*dx)) - ((v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))**2) 

            p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
            p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
            p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
            p[-1, :] = 0        # p = 0 at y = 2
        
        return p

    def cavity_flow(nt, u, v, dt, dx, dy, p, rho, µ):
        un = np.empty_like(u)
        vn = np.empty_like(v)
    
        for n in range(nt):
            un = u.copy()
            vn = v.copy()
        
            p = pressure_poisson(rho, u, v, p, dx, dy, dt)
        
            u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         µ * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        µ * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            u[0, :]  = 0
            u[:, 0]  = 0
            u[:, -1] = 0
            u[-1, :] = 1    # set velocity on cavity lid equal to 1
            v[0, :]  = 0
            v[-1, :] = 0
            v[:, 0]  = 0
            v[:, -1] = 0
        
        
        return u, v, p
    # Initialising variables
    u = np.zeros((NODES, NODES))   # x component of velocity vector
    v = np.zeros((NODES, NODES))   # y component of velocity vector
    p = np.zeros((NODES, NODES))   # pressure
    
    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, µ)
    Reynolds_Number = (rho*u[-1,:])/µ  # Calculating Reynolds Number

    u_mid = u[21,:]

    

    # Plotting Solution
    fig = plt.figure(figsize=(11,7), dpi=100)
    plt.contourf(X, Y, (np.sqrt(u**2 + v**2)), cmap='rainbow', levels=16)  
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reynolds Number = ' + str(Reynolds_Number[0]) + ' at T = ' + str(T_FINAL) + 's')
    #plt.streamplot(X, Y, u, v)


    #plt.plot(x, u_mid)
    #plt.grid()
    #plt.xlabel('X')
    #plt.ylabel('u velovicty (m/s)')
    #plt.title('Reynolds Number = ' + str(Reynolds_Number[0]) + ' at T = ' + str(T_FINAL) + 's')
    plt.show()


    

if __name__ =='__main__':
    main()
