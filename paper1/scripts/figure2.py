import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Figure 2: Box plot comparison from Experiment 1
# Re-run experiment 1 to get the data, or use saved result
from dimensional_opening import run_experiment_1
result1 = run_experiment_1(n_networks=30, verbose=False)

from dimensional_opening import QualityFlag
control_etas = [r.eta for r in result1.results[:30] 
                if not r.skipped and r.quality != QualityFlag.FAILED and not np.isnan(r.eta)]
test_etas = [r.eta for r in result1.results[30:] 
             if not r.skipped and r.quality != QualityFlag.FAILED and not np.isnan(r.eta)]

fig, ax = plt.subplots(figsize=(5, 4))
bp = ax.boxplot([control_etas, test_etas], labels=['Control\n(+random)', 'Test\n(+autocatalytic)'])
ax.set_ylabel('η (activation ratio)', fontsize=11)
ax.set_title('Experiment 1: Control vs Test')
ax.axhline(y=0.426, color='gray', linestyle='--', alpha=0.5, label='Pure Brusselator')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('exp1_boxplot.png', dpi=150)
plt.close()
print(f"Saved exp1_boxplot.png")
print(f"Control: n={len(control_etas)}, median={np.median(control_etas):.3f}")
print(f"Test: n={len(test_etas)}, median={np.median(test_etas):.3f}")