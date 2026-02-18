# Figure 3: Example Brusselator time series showing oscillation
from dimensional_opening import simulate_reactions, DrivingMode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

result_sim = simulate_reactions(
    reactions=["A -> X", "B + X -> Y + D", "X + X + Y -> X + X + X", "X -> E"],
    rate_constants=[1.0, 1.0, 1.0, 1.0],
    initial_concentrations={'A': 1.0, 'B': 3.0, 'X': 1.0, 'Y': 1.0, 'D': 0.0, 'E': 0.0},
    t_span=(0, 100),
    n_points=2000,
    driving_mode=DrivingMode.CSTR,
    cstr_dilution_rate=0.1,
    cstr_feed_concentrations={'A': 1.0, 'B': 3.0},
)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(result_sim.time, result_sim.concentrations[:, 2], label='X', linewidth=1.5)
ax.plot(result_sim.time, result_sim.concentrations[:, 3], label='Y', linewidth=1.5)
ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Concentration', fontsize=11)
ax.set_title('Brusselator Limit Cycle Dynamics')
ax.legend()
plt.tight_layout()
plt.savefig('brusselator_timeseries.png', dpi=150)
plt.close()
print("Saved brusselator_timeseries.png")