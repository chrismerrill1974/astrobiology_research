from dimensional_opening import simulate_reactions, DrivingMode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

result_sim = simulate_reactions(
    reactions=["A -> X", "B + X -> Y + D", "X + X + Y -> X + X + X", "X -> E"],
    rate_constants=[1.0, 1.0, 1.0, 1.0],
    initial_concentrations={'A': 1.0, 'B': 3.0, 'X': 1.0, 'Y': 1.0, 'D': 0.0, 'E': 0.0},
    t_span=(0, 50),
    n_points=2000,
    driving_mode=DrivingMode.CHEMOSTAT,
    chemostat_species={'A': 1.0, 'B': 3.0},
)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(result_sim.time, result_sim.concentrations[:, 2], label='X', linewidth=1.5)
ax.plot(result_sim.time, result_sim.concentrations[:, 3], label='Y', linewidth=1.5)
ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Concentration', fontsize=11)
ax.set_title('Brusselator Limit Cycle (Chemostat)')
ax.legend()
plt.tight_layout()
plt.savefig('brusselator_oscillation_v3.png', dpi=150)
plt.close()
print("Saved brusselator_oscillation_v3.png")