import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Lorentz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0/3.0

def lorentz_system(state, t):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

# Time array
t = np.linspace(0, 26, 1000)

# Initial conditions
initial_conditions = [
    [-10.594887514965762, -15.398070616041126, 22.893945844825783],   # Main trajectory
    [-10.594887514965762+10**(-4), -15.398070616041126+10**(-4), 22.893945844825783+10**(-4)],  # Slightly perturbed
]

# Solve the system
solutions = [odeint(lorentz_system, ic, t) for ic in initial_conditions]

# Calculate distance between trajectories and its logarithm
distances = np.linalg.norm(solutions[0] - solutions[1], axis=1)
log_distances = np.log(distances + 1e-20)  # Add small constant to avoid log(0)

# Create figure
fig = plt.figure(figsize=(24, 8), facecolor='white')
fig.subplots_adjust(wspace=0.3)

# Main 3D plot
ax1 = fig.add_subplot(121, projection='3d', facecolor='white')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# Log Distance plot
ax2 = fig.add_subplot(122, facecolor='white')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Log(Distance)', fontsize=12)
ax2.set_title('Log Distance Between Trajectories', fontsize=14)

# Colors and labels
#colors = ['#1f77b4', '#ff7f0e']
colors = ['royalblue', 'crimson']
labels = ['Primary', 'Perturbed']

# Initialize plot elements
lines = [ax1.plot([], [], [], lw=1.5, color=color, alpha=0.6, label=label)[0] 
         for color, label in zip(colors, labels)]
points = [ax1.plot([], [], [], 'o', color=color, markersize=6, alpha=0.9)[0] 
          for color in colors]
vector = None
distance_line, = ax2.plot([], [], '-', color='#2ca02c', lw=1.5, alpha=0.8, label='Log Distance')
current_distance_point, = ax2.plot([], [], 'o', color='#d62728', markersize=8, label='Current log distance')

# Set plot limits
all_points = np.concatenate(solutions)
max_val = np.max(np.abs(all_points)) * 1.1
ax1.set_xlim(-max_val, max_val)
ax1.set_ylim(-max_val, max_val)
ax1.set_zlim(0, 2*max_val)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_zlabel('Z', fontsize=12)
ax1.set_title('Lorenz Attractor Trajectories', fontsize=14)
ax1.legend()

ax2.set_xlim(0, t[-1])
ax2.set_ylim(np.min(log_distances)*1.1, np.max(log_distances)*1.1)  # Adjusted for log scale
ax2.legend()

def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
    distance_line.set_data([], [])
    current_distance_point.set_data([], [])
    return lines + points + [distance_line, current_distance_point]

def animate(i):
    global vector
    
    # Update trajectories
    for line, point, sol in zip(lines, points, solutions):
        line.set_data(sol[:i+1, 0], sol[:i+1, 1])
        line.set_3d_properties(sol[:i+1, 2])
        point.set_data([sol[i, 0]], [sol[i, 1]])
        point.set_3d_properties([sol[i, 2]])
    
    # Update difference vector
    pos1 = solutions[0][i]
    pos2 = solutions[1][i]
    diff = pos2 - pos1
    if vector is not None:
        vector.remove()
    vector = ax1.quiver(
        pos1[0], pos1[1], pos1[2],
        diff[0], diff[1], diff[2],
        color='g', arrow_length_ratio=0.2, lw=2, alpha=1
    )
    
    # Update log distance plot
    distance_line.set_data(t[:i+1], log_distances[:i+1])
    current_distance_point.set_data([t[i]], [log_distances[i]])
    
    return lines + points + [distance_line, current_distance_point]

# Create animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(t)//2, interval=40, blit=False, repeat=True)
plt.tight_layout()
writer = animation.PillowWriter(fps=10)
ani.save('Distance-of-two-Traj.gif', writer=writer, dpi=100)
#plt.show()