import matplotlib.pyplot as plt
import numpy as np

# --- 1. DATA ENTRY ---

# DeepACO with NLS (Intended)
nls_data = [
    [30.64, 28.21, 11.19, 11.09], [28.41, 25.99, 11.13, 11.02],
    [23.96, 21.75, 11.14, 11.03], [20.84, 18.82, 11.12, 10.99],
    [18.43, 16.60, 11.06, 10.93], [16.80, 15.26, 10.98, 10.87],
    [15.50, 14.11, 10.96, 10.85], [15.19, 13.73, 10.93, 10.84],
    [14.97, 13.67, 10.96, 10.86], [14.65, 13.39, 10.96, 10.87],
    [14.53, 13.20, 10.93, 10.85], [14.51, 13.17, 10.94, 10.85],
    [14.48, 13.25, 10.94, 10.85], [14.43, 13.14, 10.93, 10.84],
    [14.40, 13.11, 10.94, 10.86], [14.35, 13.11, 10.92, 10.85],
    [14.44, 13.31, 10.97, 10.87], [14.29, 13.15, 10.93, 10.86],
    [14.30, 13.11, 10.94, 10.86], [14.26, 13.04, 10.92, 10.85],
    [14.26, 13.10, 10.93, 10.85]
]

# DeepACO with Unconstrained 2-OPT (Failure Case)
opt2_unconstrained_data = [
    [30.66, 28.29, 11.27, 11.16], [30.34, 27.91, 11.28, 11.15],
    [30.29, 27.80, 11.28, 11.16], [30.47, 28.06, 11.29, 11.16],
    [30.57, 28.21, 11.28, 11.16], [30.52, 28.10, 11.28, 11.16],
    [29.97, 27.51, 11.27, 11.15], [29.58, 27.13, 11.26, 11.13],
    [29.95, 27.49, 11.26, 11.13], [30.53, 28.06, 11.24, 11.11],
    [31.05, 28.63, 11.24, 11.10], [29.61, 27.13, 11.22, 11.08],
    [28.27, 25.70, 11.21, 11.07], [27.58, 25.06, 11.21, 11.05],
    [28.02, 25.56, 11.20, 11.05], [27.75, 25.38, 11.19, 11.05],
    [27.08, 24.73, 11.20, 11.03], [26.99, 24.60, 11.18, 11.03],
    [26.57, 24.15, 11.17, 11.03], [26.56, 24.21, 11.17, 11.02],
    [26.88, 24.52, 11.18, 11.03]
]

# DeepACO with Limited 2-OPT (N/4) - NEW DATA
opt2_limited_data = [
    [30.64, 28.30, 11.26, 11.16], [28.33, 25.90, 11.28, 11.15],
    [23.86, 21.64, 11.26, 11.12], [19.96, 17.96, 11.21, 11.04],
    [17.69, 15.91, 11.12, 10.96], [15.82, 14.06, 11.00, 10.88],
    [15.68, 13.96, 11.00, 10.88], [15.68, 13.87, 10.97, 10.86],
    [15.63, 13.86, 10.97, 10.86], [15.73, 13.86, 10.97, 10.87],
    [15.55, 13.76, 10.96, 10.87], [15.60, 13.81, 10.97, 10.87],
    [15.83, 13.81, 10.94, 10.85], [15.47, 13.60, 10.95, 10.86],
    [15.66, 13.75, 10.93, 10.86], [15.83, 14.05, 10.97, 10.87],
    [16.02, 13.97, 10.93, 10.85], [16.04, 14.00, 10.92, 10.84],
    [16.04, 13.97, 10.93, 10.84], [15.94, 13.86, 10.93, 10.84],
    [15.99, 13.95, 10.94, 10.85]
]

bench_t1 = 11.2472887
bench_t5 = 11.1011562
epochs = range(len(nls_data))

# Helper to process data
def process(data):
    return (np.array([x[0] for x in data]), # Mean 1
            np.array([x[1] for x in data]), # Best 1
            np.array([x[2] for x in data]), # Mean 5
            np.array([x[3] for x in data])) # Best 5

nls_m1, nls_b1, nls_m5, nls_b5 = process(nls_data)
unc_m1, unc_b1, unc_m5, unc_b5 = process(opt2_unconstrained_data)
lim_m1, lim_b1, lim_m5, lim_b5 = process(opt2_limited_data)

# --- PLOTTING ---
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 18, 'axes.titlesize': 22,
    'axes.labelsize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18,
    'legend.fontsize': 16, 'lines.linewidth': 4, 'lines.markersize': 0,
    'figure.titlesize': 24
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# --- PLOT A: 1-Iteration ---
# 1. NLS (Blue)
ax1.plot(epochs, nls_m1, label='Intended NLS', color='#1f77b4')
ax1.fill_between(epochs, nls_m1, nls_b1, color='#1f77b4', alpha=0.15)

# 2. Limited 2-OPT (Orange) - New
ax1.plot(epochs, lim_m1, label='Limited 2-OPT (N/4)', color='#ff7f0e', linestyle='-.')
ax1.fill_between(epochs, lim_m1, lim_b1, color='#ff7f0e', alpha=0.15)

# 3. Unconstrained 2-OPT (Red)
ax1.plot(epochs, unc_m1, label='Unconstrained 2-OPT', color='#d62728')
ax1.fill_between(epochs, unc_m1, unc_b1, color='#d62728', alpha=0.15)

# Baseline
ax1.axhline(y=bench_t1, color='green', linestyle='--', linewidth=3, label='Baseline')

ax1.set_title('(a) Construction Quality (1 Iteration)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Objective Cost')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right')

# --- PLOT B: 5-Iteration ---
# 1. NLS (Blue)
ax2.plot(epochs, nls_m5, label='Intended NLS', color='#1f77b4')
ax2.fill_between(epochs, nls_m5, nls_b5, color='#1f77b4', alpha=0.15)

# 2. Limited 2-OPT (Orange)
ax2.plot(epochs, lim_m5, label='Limited 2-OPT (N/4)', color='#ff7f0e', linestyle='-.')
ax2.fill_between(epochs, lim_m5, lim_b5, color='#ff7f0e', alpha=0.15)

# 3. Unconstrained 2-OPT (Red)
ax2.plot(epochs, unc_m5, label='Unconstrained 2-OPT', color='#d62728')
ax2.fill_between(epochs, unc_m5, unc_b5, color='#d62728', alpha=0.15)

# Baseline
ax2.axhline(y=bench_t5, color='green', linestyle='--', linewidth=3, label='Baseline')

ax2.set_title('(b) Refined Quality (5 Iterations)')
ax2.set_xlabel('Epochs')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_ylim(10.80, 11.35)

plt.tight_layout()
plt.savefig('nls_vs_limited_vs_unconstrained.pdf', format='pdf', bbox_inches='tight')
plt.show()