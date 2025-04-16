import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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
t = np.linspace(0, 50, 5000)  # Reduced resolution for smoother animation

# Initial conditions (slightly perturbed)
initial_conditions = [
    [0.0, 1.0, 1.05],   # Main trajectory
    [0.0, 1.01, 1.05],  # Slightly perturbed
    [0.0, 0.99, 1.05]   # Another perturbation
]

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Colors and labels for each trajectory
colors = ['royalblue', 'crimson', 'limegreen']
labels = ['Primary trajectory', 'Perturbation +0.01', 'Perturbation -0.01']

# Solve the system for each initial condition
solutions = []
for ic in initial_conditions:
    solution = odeint(lorentz_system, ic, t)
    solutions.append(solution)

# Initialize lines and points for each trajectory
lines = []
points = []
for i, (color, label) in enumerate(zip(colors, labels)):
    # Plot the initial empty trajectory line
    line, = ax.plot([], [], [], lw=0.9, color=color, alpha=0.9, label=label)
    lines.append(line)
    
    # Plot the initial position marker
    point, = ax.plot([], [], [], 'o', color=color, markersize=6, alpha=0.9)
    points.append(point)
    
    # Mark initial point
    ax.scatter(*initial_conditions[i], color='red', s=80, marker='o', depthshade=False)

# Set axis limits based on all solutions
all_points = np.concatenate(solutions)
max_val = np.max(np.abs(all_points)) * 1.1
ax.set_xlim(-max_val, max_val)
ax.set_ylim(-max_val, max_val)
ax.set_zlim(0, 2*max_val)

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

# Animation function
def animate(i):
    # i is the frame number (we'll use every 5th point for smoother animation)
    frame = i * 5
    
    for j, (line, point, sol) in enumerate(zip(lines, points, solutions)):
        # Update the trajectory line (show path up to current frame)
        x = sol[:frame, 0]
        y = sol[:frame, 1]
        z = sol[:frame, 2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        
        # Update the current position marker
        if frame > 0:
            point.set_data([sol[frame-1, 0]], [sol[frame-1, 1]])
            point.set_3d_properties([sol[frame-1, 2]])
    
    # Rotate view slightly for better visualization
    ax.view_init(elev=20, azim=i*0.3 % 360)
    
    return lines + points

# Create animation (using every 5th frame for performance)
nframes = len(t) // 5
ani = animation.FuncAnimation(
    fig, animate, frames=nframes, interval=20, blit=False, repeat=True
)

plt.tight_layout()
plt.show()