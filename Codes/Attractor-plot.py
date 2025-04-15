# Lorentz Attractor (plot for three initial condition with slite perturbation) 

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lorentz system parameters (chaotic regime)
sigma = 10.0    # Prandtl number
rho = 28.0      # Rayleigh number
beta = 8.0/3.0  # Geometric parameter

# Define the Lorentz system ODEs
def lorentz_system(state, t):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Time array (long enough to see chaotic behavior)
t = np.linspace(0, 50, 50000)  # Increased resolution

# Initial conditions (slightly perturbed)
initial_conditions = [
    [0.0, 1.0, 1.05],   # Main trajectory
    [0.0, 1.01, 1.05],  # Slightly perturbed
    [0.0, 0.99, 1.05]   # Another perturbation
]

# Create figure
plt.figure(figsize=(12, 10))
ax = plt.axes(projection='3d')

# Plot each trajectory
colors = ['royalblue', 'crimson', 'limegreen']
labels = ['Primary trajectory', 'Perturbation +0.01', 'Perturbation -0.01']

for i, (ic, color, label) in enumerate(zip(initial_conditions, colors, labels)):
    solution = odeint(lorentz_system, ic, t)
    x, y, z = solution.T
    
    # Plot the 3D trajectory
    ax.plot(x, y, z, lw=0.7, color=color, alpha=0.9, label=label)
    
    # Mark initial and final points
    ax.scatter(*ic, color='red', s=80, marker='o', depthshade=False)
    ax.scatter(x[-1], y[-1], z[-1], color='black', s=40, marker='x', depthshade=False)

# Visualization enhancements
ax.set_xlabel('X axis', fontsize=12, labelpad=10)
ax.set_ylabel('Y axis', fontsize=12, labelpad=10)
ax.set_zlabel('Z axis', fontsize=12, labelpad=10)
ax.set_title('3D Phase Portrait of Lorentz Attractor\n' + 
             fr'($\sigma={sigma}$, $\rho={rho}$, $\beta={beta:.2f}$)', 
             fontsize=14, pad=20)


# Add legend and grid
ax.legend(loc='upper right', fontsize=10)
ax.grid(True)
ax.xaxis.set_pane_color((0.98, 0.98, 0.98, 0.9))
ax.yaxis.set_pane_color((0.98, 0.98, 0.98, 0.9))
ax.zaxis.set_pane_color((0.98, 0.98, 0.98, 0.9))

# Save high-quality image
plt.savefig('lorentz_3d.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()
