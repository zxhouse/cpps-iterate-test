# Convergent Iterate–Test Optimization for Programmable SDN–IIoT CPPS

Simulation testbed, experimental framework, and reproduction package for the research paper:
> **Convergent Iterate–Test Optimization for Programmable SDN–IIoT Cyber-Physical Production Systems: Theoretical Guarantees and Simulation Testbed Evidence**

---

## Overview

This repository contains the complete, reproducible simulation testbed implementing closed-loop iterate–test self-reconfiguration over programmable Software-Defined Networking (SDN) and Industrial Internet of Things (IIoT) Cyber-Physical Production Systems (CPPS).

The model evaluates a 12-feature binary configuration space ($2^{12} = 4,096$ candidate prototypes) across three industrial benchmark topologies under stochastic disturbances (long-range dependent cross-traffic, sensor bias, tool-wear drift, and multi-criteria trade-offs).

---

## Repository Structure

```text
.
├── sim_cpps_testbed.py         # 12-knob CPPS simulation environment and E1–E6 experiments runner
├── generate_astrj_figures.py   # Publication-quality schematic and empirical plot generator (Figures 1–8)
├── results_astrj.json          # Precomputed dataset from full experimental runs (E1–E6)
├── fig1_cpps_architecture.png  # Two-level iterate–test control architecture schematic
├── fig2_iterate_test_protocol.png # CPPS testbed architecture & 3 benchmark topologies (L1, L2, L3)
├── fig3_exp1_allocation_curse.png # Schematic of Experiments E1 & E2
├── fig4_exp2_digital_twin.png  # Empirical evaluation for Experiments E1 & E2
├── fig5_exp3_ablation.png      # Schematic of Experiments E3 & E4
├── fig6_exp4_bias_drift.png    # Empirical evaluation for Experiments E3 & E4
├── fig7_exp5_reduce_pareto.png # Schematic of Experiments E5 & E6
├── fig8_exp6_epoch_reframe.png # Empirical evaluation for Experiments E5 & E6
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Requirements and Installation

Python 3.9+ is recommended. Install required packages using `pip`:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy`
- `scipy`
- `matplotlib`
- `pillow`

---

## Reproducing the Experiments

### 1. Run Experiments E1–E6
To re-run the full simulation pipeline across all three industrial topologies and generate `results_astrj.json`:

```bash
python3 sim_cpps_testbed.py
```

The script conducts:
- **E1**: Test budget allocation policies (Single-shot, 2-per-5, UCB1, Thompson sampling, etc.) & quantification of the optimizer's curse.
- **E2**: Queueing Digital Twin pre-screening depth ($M_{\mathrm{pre}}$) and surrogate fidelity crossover ($\rho_{\mathrm{DT}}$).
- **E3**: Assumption ablation study (breaking Conditions A2–A5 of Theorem 1).
- **E4**: Empirical validation of $2B$ and $2D$ robustness bounds under sensor bias and drift.
- **E5**: Multi-stakeholder intent elicitation (REDUCE consistency rectification vs. Proposition 2 Pareto archive).
- **E6**: Expected-improvement-triggered epoch reframing ($\mathcal{F}_0 \to \mathcal{F}_1$).

### 2. Generate Figures
To reproduce all 8 publication schematics and quantitative plots from `results_astrj.json`:

```bash
python3 generate_astrj_figures.py
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
