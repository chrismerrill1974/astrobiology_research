from dimensional_opening import simulate_reactions, DrivingMode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Classic Brusselator parameters: A=1, B=3 with B > 1+A² = 2
result_sim = simulate_reactions(
    reactions=["A -> X", "B + X -> Y + D", "X + X + Y -> X + X + X", "X -> E"],
    rate_constants=[1.0, 1.0, 1.0, 1.0],
    initial_concentrations={'A': 1.0, 'B': 3.0, 'X': 0.5, 'Y': 0.5, 'D': 0.0, 'E': 0.0},
    t_span=(0, 50),
    n_points=5000,
    driving_mode=DrivingMode.CHEMOSTAT,
    chemostat_species={'A': 1.0, 'B': 3.0},
)

# Find X and Y indices
species = ['A', 'B', 'X', 'Y', 'D', 'E']
ix = species.index('X')
iy = species.index('Y')

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(result_sim.time, result_sim.concentrations[:, ix], label='X', linewidth=1)
ax.plot(result_sim.time, result_sim.concentrations[:, iy], label='Y', linewidth=1)
ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Concentration', fontsize=11)
ax.set_title('Brusselator Dynamics')
ax.legend()
plt.tight_layout()
plt.savefig('brusselator_v4.png', dpi=150)
plt.close()
print("Saved brusselator_v4.png")
print(f"X range: {result_sim.concentrations[:, ix].min():.2f} - {result_sim.concentrations[:, ix].max():.2f}")
print(f"Y range: {result_sim.concentrations[:, iy].min():.2f} - {result_sim.concentrations[:, iy].max():.2f}")