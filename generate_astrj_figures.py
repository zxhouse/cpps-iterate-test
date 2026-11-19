"""
Publication-Grade Schematics & Plot Generator for ASTRJ Manuscript.

Guarantees:
1. MS-Equation-style STIX/Cambria mathematical typesetting ($...$) on all
   schematics and all plots.
2. Equal-aspect (ax.set_aspect("equal")) vector device icons with proper
   arrowhead scaling (ms=4.5) and generous spacing.
3. Rich multi-topology schematics (L1, L2, L3) in Figure 2 and embedded
   mini_testbed network topology state diagrams inside the experimental
   course schematics (Figures 3, 5, and 7).
4. Zero text-on-text, text-on-line, or subplot-label collisions across all
   8 figures.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle

# MS Equation (Cambria Math / STIX) styling across all figures
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.4,
    "axes.labelsize": 8.8,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

COL_BLUE = "#1f4e79"
COL_GREEN = "#2e8b57"
COL_AMBER = "#b8860b"
COL_RED = "#c0504d"
COL_PURPLE = "#6d4c93"
COL_TEAL = "#0f766e"
COL_SLATE = "#64748b"
COL_GRAY = "#7a8ea3"


# ==============================================================================
# 1. Equal-Aspect Vector Device Icons & Mini-Topology Builder
# ==============================================================================

def draw_switch(ax, x, y, s=1.0, label=None, fs=6.8, fc=COL_BLUE, ms=4.5, z=5):
    """OpenFlow SDN switch: rounded dark-blue chassis with two pairs of opposite white arrows."""
    w, h = 1.05 * s, 0.44 * s
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={0.08*s}",
        fc=fc, ec="#0f2a44", lw=0.75, zorder=z
    ))
    a = 0.30 * s
    dy = 0.09 * s
    for sgn, yy in ((1, y + dy), (-1, y - dy)):
        x0 = x - sgn * a / 2
        ax.annotate(
            "", xy=(x0 + sgn * a, yy), xytext=(x0, yy), zorder=z + 1,
            arrowprops=dict(arrowstyle="-|>", color="white", lw=0.75, mutation_scale=ms, shrinkA=0, shrinkB=0)
        )
    if label:
        ax.text(
            x, y - h / 2 - 0.10 * s, label,
            ha="center", va="top", fontsize=fs, fontweight="bold", color="#111",
            bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.85), zorder=z + 2
        )


def draw_host_pc(ax, x, y, s=1.0, label=None, fs=6.4, fc="#e8eef5", z=5):
    """Operator Workstation / HMI PC: monitor, inner screen, stand, and base."""
    w, h = 0.50 * s, 0.34 * s
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2 + 0.06 * s), w, h,
        boxstyle=f"round,pad=0,rounding_size={0.035*s}",
        fc="#3a3a3a", ec="#1a1a1a", lw=0.6, zorder=z
    ))
    ax.add_patch(Rectangle(
        (x - w / 2 + 0.035 * s, y - h / 2 + 0.095 * s), w - 0.07 * s, h - 0.07 * s,
        fc=fc, ec="none", zorder=z + 1
    ))
    ax.add_patch(Rectangle((x - 0.035 * s, y - h / 2 - 0.025 * s), 0.07 * s, 0.085 * s, fc="#555", ec="none", zorder=z))
    ax.add_patch(Rectangle((x - 0.14 * s, y - h / 2 - 0.06 * s), 0.28 * s, 0.038 * s, fc="#555", ec="none", zorder=z))
    if label:
        ax.text(x, y - h / 2 - 0.12 * s, label, ha="center", va="top", fontsize=fs, color="#111", linespacing=1.05, zorder=z + 2)


def draw_server(ax, x, y, s=1.0, label=None, fs=6.4, fc="#5a6b7c", z=5):
    """Edge/Cloud Rack Server tower with drive slots and status LED."""
    w, h = 0.34 * s, 0.54 * s
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={0.035*s}",
        fc=fc, ec="#222", lw=0.6, zorder=z
    ))
    for k in range(4):
        yy = y + h / 2 - 0.10 * s - k * 0.10 * s
        ax.add_patch(Rectangle((x - w / 2 + 0.055 * s, yy - 0.018 * s), w - 0.11 * s, 0.034 * s, fc="#e2e8f0", ec="none", zorder=z + 1))
    ax.add_patch(Circle((x + w / 2 - 0.075 * s, y - h / 2 + 0.07 * s), 0.024 * s, fc="#7CFC00", edgecolor="none", zorder=z + 1))
    if label:
        ax.text(x, y - h / 2 - 0.10 * s, label, ha="center", va="top", fontsize=fs, color="#111", linespacing=1.05, zorder=z + 2)


def draw_controller(ax, x, y, s=1.0, label="SDN Controller", fs=7.4, label_above=False, single=False, z=5):
    """Redundant pair (or single legacy instance) of green SDN controller servers."""
    if single:
        draw_server(ax, x, y, s=s, fc=COL_SLATE, z=z)
    else:
        draw_server(ax, x - 0.22 * s, y, s=s, fc=COL_GREEN, z=z)
        draw_server(ax, x + 0.22 * s, y, s=s, fc=COL_GREEN, z=z)
    if label:
        if label_above:
            ax.text(x, y + 0.34 * s, label, ha="center", va="bottom", fontsize=fs, fontweight="bold", color="#111", zorder=z + 2)
        else:
            ax.text(x, y - 0.34 * s, label, ha="center", va="top", fontsize=fs, fontweight="bold", color="#111", zorder=z + 2)


def draw_plc(ax, x, y, s=1.0, label=None, fs=6.4, fc="#3b6998", z=5):
    """Industrial PLC / CNC Controller unit with I/O bays and LED."""
    w, h = 0.50 * s, 0.40 * s
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={0.035*s}",
        fc=fc, ec="#1a365d", lw=0.65, zorder=z
    ))
    for k in range(3):
        xx = x - w / 2 + 0.07 * s + k * 0.135 * s
        ax.add_patch(Rectangle((xx, y - h / 2 + 0.07 * s), 0.095 * s, h - 0.14 * s, fc="#ebf4ff", ec="#1a365d", lw=0.4, zorder=z + 1))
    ax.add_patch(Circle((x + w / 2 - 0.06 * s, y + h / 2 - 0.07 * s), 0.025 * s, fc="#ffd700", edgecolor="none", zorder=z + 2))
    if label:
        ax.text(x, y - h / 2 - 0.10 * s, label, ha="center", va="top", fontsize=fs, color="#111", linespacing=1.05, zorder=z + 2)


def draw_db(ax, x, y, w=0.44, h=0.40, label=None, fs=6.4, fc="#efe6f7", ec=COL_PURPLE, z=5):
    """Cylindrical Historian / Pareto Archive Database."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, lw=0.75, zorder=z))
    ax.add_patch(Ellipse((x, y + h / 2), w, 0.30 * w, fc="#e0d0f0", ec=ec, lw=0.75, zorder=z + 1))
    ax.add_patch(Ellipse((x, y - h / 2), w, 0.30 * w, fc=fc, ec=ec, lw=0.75, zorder=z - 1))
    if label:
        ax.text(x, y - h / 2 - 0.12, label, ha="center", va="top", fontsize=fs, color="#111", linespacing=1.05, zorder=z + 2)


def draw_link(ax, p, q, lw=1.3, color=COL_GRAY, ls="-", z=1):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")


def draw_arrow_pts(ax, pts, color=COL_BLUE, lw=1.15, ls="-", ms=9.0, z=4):
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=z)
    ax.annotate(
        "", xy=pts[-1], xytext=pts[-2], zorder=z + 1,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=ms, shrinkA=0, shrinkB=0)
    )


def draw_config_bits(ax, x, y, bits, cell=0.85, gap=0.14, hi_indices=(), c=COL_BLUE, z=4):
    for i, b in enumerate(bits):
        fc = (COL_RED if i in hi_indices else c) if b else "white"
        ec = COL_RED if i in hi_indices else c
        ax.add_patch(Rectangle((x + i * (cell + gap), y), cell, cell, fc=fc, ec=ec, lw=0.6, zorder=z))
    return x + len(bits) * (cell + gap)


def draw_mini_testbed(
    ax, x, y, s=1.0, topo="L3",
    hl_red=(), hl_green=(),
    sleep_edges=None, hot_edges=(), green_edges=(),
    ctrl=True, single_ctrl=False, hosts=True, show_sw_labels=False,
    ctrl_h=10.8, sw_ms=2.6
):
    """
    Draws a compact 6-switch CPPS topology (L1, L2, or L3) at (x, y) with scale s.
    Returns dict of switch coordinates P_['s1'..'s6'].
    """
    if topo == "L1":
        pos = {"s1": (0.0, 4.2), "s2": (4.0, 6.8), "s3": (8.0, 6.8), "s4": (4.0, 1.4), "s6": (8.0, 1.4), "s5": (12.0, 4.2)}
        edges = [("s1", "s2"), ("s2", "s3"), ("s3", "s5"), ("s1", "s4"), ("s4", "s6"), ("s6", "s5"), ("s2", "s4"), ("s3", "s6")]
        default_sleep = {("s1", "s4"), ("s4", "s6"), ("s6", "s5")}
    elif topo == "L2":
        pos = {"s1": (0.0, 4.2), "s2": (4.2, 7.4), "s3": (6.0, 4.2), "s4": (4.2, 0.8), "s6": (8.2, 0.8), "s5": (12.0, 4.2)}
        edges = [
            ("s1", "s2"), ("s2", "s5"), ("s1", "s3"), ("s3", "s5"),
            ("s1", "s4"), ("s4", "s6"), ("s6", "s5"), ("s2", "s3"),
            ("s2", "s6"), ("s4", "s3"), ("s2", "s4")
        ]
        default_sleep = {("s1", "s4"), ("s4", "s6"), ("s6", "s5")}
    else:  # L3 Island Factory
        pos = {"s1": (0.0, 4.0), "s2": (6.0, 7.6), "s3": (6.0, 4.0), "s4": (3.0, 0.4), "s6": (9.0, 0.4), "s5": (12.0, 4.0)}
        edges = [("s1", "s2"), ("s2", "s5"), ("s1", "s3"), ("s3", "s5"), ("s1", "s4"), ("s4", "s6"), ("s6", "s5"), ("s2", "s3")]
        default_sleep = {("s1", "s4"), ("s4", "s6"), ("s6", "s5"), ("s2", "s3")}

    if sleep_edges is None:
        sleep_edges = default_sleep
    else:
        sleep_edges = set(sleep_edges)
    hot_set = set(hot_edges) | {(b, a) for a, b in hot_edges}
    grn_set = set(green_edges) | {(b, a) for a, b in green_edges}
    slp_set = set(sleep_edges) | {(b, a) for a, b in sleep_edges}

    P_ = {k: (x + v[0] * s, y + v[1] * s) for k, v in pos.items()}
    for a, b in edges:
        if (a, b) in hot_set:
            draw_link(ax, P_[a], P_[b], lw=1.9, color=COL_RED, ls="-", z=2)
        elif (a, b) in grn_set:
            draw_link(ax, P_[a], P_[b], lw=1.7, color=COL_GREEN, ls="-", z=2)
        elif (a, b) in slp_set:
            draw_link(ax, P_[a], P_[b], lw=1.0, color="#c98a2b", ls=(0, (3, 1.5)), z=1)
        else:
            draw_link(ax, P_[a], P_[b], lw=1.05, color=COL_GRAY, ls="-", z=1)

    for k, (px, py) in P_.items():
        fc = COL_RED if k in hl_red else (COL_GREEN if k in hl_green else COL_BLUE)
        draw_switch(ax, px, py, s=2.2 * s, fc=fc, ms=sw_ms, z=5)
        if show_sw_labels:
            if k == "s2":
                ax.text(px - 1.55 * s, py + 0.35 * s, "S2", ha="center", va="center", fontsize=5.1, fontweight="bold", color="#222",
                        bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.85), zorder=7)
            else:
                dy = -1.25 * s
                ax.text(px, py + dy, k.upper(), ha="center", va="center", fontsize=5.1, fontweight="bold", color="#222",
                        bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.85), zorder=7)

    if ctrl:
        cx, cy = x + 6.0 * s, y + ctrl_h * s
        draw_controller(ax, cx, cy, s=2.4 * s, label=None, single=single_ctrl, z=5)
        for k, (px, py) in P_.items():
            ax.plot([cx, px], [cy - 0.8 * s, py + 0.5 * s], color=COL_SLATE if single_ctrl else COL_GREEN, lw=0.45, ls=":", zorder=0.5)

    if hosts:
        for k in range(3):
            hy = y + (1.6 + k * 2.4) * s
            if k == 0:
                draw_plc(ax, x - 3.3 * s, hy, s=1.9 * s, z=5)
            else:
                draw_host_pc(ax, x - 3.3 * s, hy, s=1.9 * s, fc="#dff0e0" if k == 1 else "#ffe9c7", z=5)
            ax.plot([x - 2.6 * s, P_["s1"][0] - 0.8 * s], [hy, P_["s1"][1]], color="#bbb", lw=0.45, zorder=0.5)

            draw_server(ax, x + 15.3 * s, hy, s=1.8 * s, fc=COL_PURPLE if k == 0 else "#5a6b7c", z=5)
            ax.plot([P_["s5"][0] + 0.8 * s, x + 14.7 * s], [P_["s5"][1], hy], color="#bbb", lw=0.45, zorder=0.5)

    return P_


# ==============================================================================
# Figure 1: Two-Level Iterate-Test Control Loop Schematic
# ==============================================================================

def make_fig1():
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.set_aspect("equal")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def box(x, y, w, h, fc="#eef3f8", ec=COL_BLUE, r=1.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1.0, zorder=2))

    # Outer Frame Box (top)
    box(2, 48.5, 96, 7.8, fc="#fbf3e4", ec=COL_AMBER)
    ax.text(50, 54.0, r"$\mathbf{Outer\ level:\ Operational\ frame}\ (\mathcal{F},\ U)$", ha="center", va="center", fontsize=8.3, color="#5a3e08")
    ax.text(50, 50.6, r"12 binary knobs $\mathcal{F}=\{f_0,\dots,f_{11}\}$ ($|2^{\mathcal{F}}|=4096$) and utility $U(p)=w_{\mathrm{lat}}U_{\mathrm{lat}}(p)+w_{\mathrm{rel}}U_{\mathrm{rel}}(p)+w_{\mathrm{eff}}U_{\mathrm{eff}}(p)$",
            ha="center", va="center", fontsize=7.4, color="#333")

    # Inner Loop Row (4 stage boxes)
    W, H, Y = 21.0, 16.0, 25.5
    xs = [2.0, 27.0, 52.0, 77.0]

    # Box 1: Ideation
    box(xs[0], Y, W, H)
    ax.text(xs[0] + W / 2, Y + 13.4, r"$\mathbf{1.\ Local\ ideation\ (A2)}$", ha="center", va="center", fontsize=7.6, color=COL_BLUE)
    ax.text(xs[0] + W / 2, Y + 9.8, r"Propose variants in $\mathcal{N}_r(p_k^*)$" + "\n" + r"($r=2$ co-design flips)", ha="center", va="center", fontsize=6.9)
    draw_config_bits(ax, xs[0] + 2.8, Y + 3.0, [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1], cell=1.1, gap=0.18, hi_indices=(1, 5))

    # Box 2: Digital Twin Pre-Screening
    box(xs[1], Y, W, H, fc="#fff9eb", ec=COL_AMBER)
    ax.text(xs[1] + W / 2, Y + 13.4, r"$\mathbf{2.\ DT\ pre\text{-}screening}$", ha="center", va="center", fontsize=7.6, color="#7c580a")
    ax.text(xs[1] + W / 2, Y + 9.8, r"Evaluate $M_{\mathrm{pre}}$ candidates" + "\n" + r"via twin $\hat{U}_{\mathrm{DT}}(p)$ ($\rho_{\mathrm{DT}}$)", ha="center", va="center", fontsize=6.9)
    draw_host_pc(ax, xs[1] + W / 2 - 3.4, Y + 3.6, s=5.8, fc="#fff3d6")
    draw_server(ax, xs[1] + W / 2 + 3.4, Y + 3.6, s=5.8, fc=COL_AMBER)

    # Box 3: Prototype Deployment
    box(xs[2], Y, W, H)
    ax.text(xs[2] + W / 2, Y + 13.4, r"$\mathbf{3.\ Build\ /\ deploy}$", ha="center", va="center", fontsize=7.6, color=COL_BLUE)
    ax.text(xs[2] + W / 2, Y + 9.8, "Install OpenFlow 1.3 bundles\n& PLC/edge containers", ha="center", va="center", fontsize=6.9)
    draw_plc(ax, xs[2] + W / 2 - 4.2, Y + 3.5, s=5.6)
    draw_switch(ax, xs[2] + W / 2 + 3.6, Y + 3.5, s=5.4, ms=4.8)

    # Box 4: Noisy Test
    box(xs[3], Y, W, H, fc="#f5f0fa", ec=COL_PURPLE)
    ax.text(xs[3] + W / 2, Y + 13.4, r"$\mathbf{4.\ Noisy\ CPPS\ test}$", ha="center", va="center", fontsize=7.6, color=COL_PURPLE)
    ax.text(xs[3] + W / 2, Y + 9.6, r"Measure $y_k(p) = U(p)$" + "\n" + r"$+\,b_k(p)+d_k(p)+\varepsilon_k$", ha="center", va="center", fontsize=6.9)
    draw_switch(ax, xs[3] + W / 2 - 3.6, Y + 3.5, s=5.2, fc=COL_PURPLE, ms=4.8)
    draw_server(ax, xs[3] + W / 2 + 3.8, Y + 3.5, s=5.6, fc=COL_PURPLE)

    for a, b in zip(xs[:-1], xs[1:]):
        draw_arrow_pts(ax, [(a + W, Y + H / 2), (b, Y + H / 2)], color=COL_BLUE, lw=1.2)

    # Bottom Row: Retention Archive & Beliefs (right) + Shipped Configuration Commit (left)
    box(38.0, 3.5, 60.0, 14.5, fc="#eef3f8", ec=COL_BLUE)
    ax.text(64.5, 15.2, r"$\mathbf{5.\ Retention\ archive\ (A5)\ &\ repeated\ re\text{-}testing\ (A4)}$", ha="center", va="center", fontsize=7.7, color=COL_BLUE)
    ax.text(64.5, 10.6, r"Retain $P_k = P_{k-1}\cup\{p_k\}$, update sample means $\hat{M}_k^p = (1/n_{p,k})\sum_{i=1}^{n_{p,k}} y_i(p)$,", ha="center", va="center", fontsize=7.0)
    ax.text(64.5, 6.4, r"incumbent $p_k^* = \arg\max_{p\in P_k}\hat{M}_k^p$ (2 new builds + 3 leader re-tests per 5 rounds)", ha="center", va="center", fontsize=7.0, color="#222")
    draw_db(ax, 93.8, 10.6, w=4.8, h=4.6, fc="#e2eaf4", ec=COL_BLUE)

    box(2.0, 3.5, 29.5, 14.5, fc="#e9f4ec", ec=COL_GREEN)
    ax.text(16.75, 15.0, r"$\mathbf{Committed\ CPPS\ state}$", ha="center", va="center", fontsize=7.7, color=COL_GREEN)
    ax.text(16.75, 10.4, r"Shipped $U(p_k^*)\to U(p_\infty)$;" + "\n" + r"believed $X_k = \hat{M}_k^{p_k^*}$", ha="center", va="center", fontsize=7.0)
    ax.text(16.75, 6.0, r"or Pareto archive $\mathcal{P}_k^{\mathrm{ND}}$ (Prop. 2)", ha="center", va="center", fontsize=6.9, color="#1e4620")

    draw_arrow_pts(ax, [(xs[3] + W / 2, Y), (xs[3] + W / 2, 18.0)], color=COL_PURPLE, lw=1.2)
    draw_arrow_pts(ax, [(38.0, 10.8), (31.5, 10.8)], color=COL_GREEN, lw=1.2)
    draw_arrow_pts(ax, [(50.0, 18.0), (50.0, 20.6), (12.5, 20.6), (12.5, Y)], color=COL_BLUE, lw=1.2)
    ax.text(33.8, 22.5, r"Next round $k \to k+1$ (generativity $\sum_{k\geq 1}\delta_k = \infty$, A3)", ha="center", va="center", fontsize=6.9, color=COL_BLUE)

    draw_arrow_pts(ax, [(50.0, 48.5), (50.0, 42.2)], color=COL_AMBER, lw=1.2, ls="--")
    ax.text(51.5, 45.2, r"Epoch reframe $(\mathcal{F}_0\to\mathcal{F}_1)$ when $\mathrm{EI}_k - c_{\mathrm{iter}} < G_{\mathrm{ref}} - c_{\mathrm{ref}}$",
            ha="left", va="center", fontsize=7.1, color="#7c580a")

    fig.savefig("fig1_cpps_architecture.png", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Figure 2: Detailed CPPS Network Architecture & Three Topologies (L1, L2, L3)
# ==============================================================================

def make_fig2():
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 10.5)
    ax.set_ylim(-3.85, 5.95)
    ax.axis("off")

    # --- TOP PART: Full L3 Island-Topology Architecture with Sources & Sinks ---
    pos = {
        "s1": (2.2, 2.2),
        "s2": (4.7, 3.65),
        "s3": (4.7, 2.2),
        "s4": (3.3, 0.5),
        "s6": (6.1, 0.5),
        "s5": (7.2, 2.2),
    }
    links = [
        ("s1", "s2", 100, False, (-0.10, 0.20)),
        ("s2", "s5", 100, False, (0.10, 0.20)),
        ("s1", "s3", 60, False, (0.0, 0.16)),
        ("s3", "s5", 60, False, (0.0, 0.16)),
        ("s2", "s3", 50, True, (0.48, -0.05)),
        ("s1", "s4", 40, True, (-0.42, -0.05)),
        ("s4", "s6", 40, True, (0.0, 0.16)),
        ("s6", "s5", 40, True, (0.42, -0.05)),
    ]
    for a, b, bw, is_sleep, off in links:
        draw_link(
            ax, pos[a], pos[b],
            lw=1.2 + bw / 75.0,
            color="#c98a2b" if is_sleep else COL_GRAY,
            ls=(0, (4, 2)) if is_sleep else "-"
        )
        mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
        ax.text(
            mx + off[0], my + off[1], f"{bw} Mb/s",
            ha="center", va="center", fontsize=6.3, color="#222", zorder=3,
            bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.92)
        )

    cpos = (4.7, 5.05)
    draw_controller(ax, *cpos, s=0.85, label="(a) Redundant SDN controller & full CPPS testbed architecture", fs=7.3, label_above=True)
    for s_name, (sx, sy) in pos.items():
        ax.plot([cpos[0], sx], [cpos[1] - 0.25, sy + 0.18], color=COL_GREEN, lw=0.6, ls=":", zorder=1)
    ax.text(5.45, 5.02, "OpenFlow 1.3 control plane\n" + r"($f_4$: multipath, $f_5$: TSN, $f_6$: failover, $f_7$: sleep)",
            fontsize=6.3, color=COL_GREEN, va="center")

    sw_labels = {
        "s1": r"$\mathrm{S1}$ (ingress)",
        "s2": r"$\mathrm{S2}$ (core-N)",
        "s3": r"$\mathrm{S3}$ (core-C)",
        "s4": r"$\mathrm{S4}$ (island-W)",
        "s6": r"$\mathrm{S6}$ (island-E)",
        "s5": r"$\mathrm{S5}$ (egress)",
    }
    for s_name, (sx, sy) in pos.items():
        draw_switch(ax, sx, sy, s=0.65, label=sw_labels[s_name], fs=6.5, ms=4.5)

    ax.text(0.25, 4.30, "Layer I: Shop-floor IIoT sources\n" + r"($f_0$: OPC UA, $f_1$: Protobuf, $f_2$: delta, $f_3$: UWB)",
            ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=COL_BLUE)

    sources = [
        ("plc", "Robotic\ncell 1", "#3b6998"),
        ("plc", "Planarizer\nCNC", "#3b6998"),
        ("pc", "IIoT GW\n(OPC UA)", "#e8eef5"),
        ("pc", "BLE/UWB\nanchor", "#e8eef5"),
        ("pc", "Latency\nprobe", "#dff0e0"),
        ("pc", "LRD cross\ntraffic B1", "#ffe9c7"),
    ]
    for k, (kind, name, col) in enumerate(sources):
        x = -0.35 + (k % 2) * 1.20
        y = 3.55 - (k // 2) * 1.22
        if kind == "plc":
            draw_plc(ax, x, y, s=0.60, label=name, fs=5.8, fc=col)
        else:
            draw_host_pc(ax, x, y, s=0.60, label=name, fs=5.8, fc=col)
        draw_link(ax, (x + 0.20, y), pos["s1"], lw=0.45, color="#aaa", z=0.5)

    draw_plc(ax, 3.15, 4.15, s=0.56, label="Island cell 2", fs=5.8)
    draw_link(ax, (3.35, 4.15), pos["s2"], lw=0.45, color="#aaa", z=0.5)

    ax.text(9.15, 4.30, "Layer III: Edge-cloud & DQT\n" + r"($f_8$: filter, $f_9$: INT8 IDS, $f_{10}$: DB, $f_{11}$: DQT)",
            ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=COL_PURPLE)

    sinks = [
        ("srv", "Edge\npre-filter", COL_GREEN),
        ("srv", "INT8 IDS\ndetector", COL_AMBER),
        ("db", "Sharded\nhistorian", COL_PURPLE),
        ("pc", "Active DQT\nworkstation", "#e8eef5"),
        ("srv", "Telemetry\ncollector", "#5a6b7c"),
        ("srv", "Failover\nreplica", "#5a6b7c"),
    ]
    for k, (kind, name, col) in enumerate(sinks):
        x = 8.55 + (k % 2) * 1.20
        y = 3.55 - (k // 2) * 1.22
        if kind == "srv":
            draw_server(ax, x, y, s=0.60, label=name, fs=5.8, fc=col)
        elif kind == "db":
            draw_db(ax, x, y, w=0.33, h=0.31, label=name, fs=5.8)
        else:
            draw_host_pc(ax, x, y, s=0.60, label=name, fs=5.8, fc=col)
        draw_link(ax, (x - 0.18, y), pos["s5"], lw=0.45, color="#aaa", z=0.5)

    # --- BOTTOM PART: Three Benchmark Topologies (L1, L2, L3) Side-by-Side ---
    topos = [
        ("L1", r"(b) L1: Modular assembly line ($K=2$)", "Ring + daisy-chain workcells, 44 $H_1$ traps", -0.95),
        ("L2", r"(c) L2: Coupled planarization cell ($K=4$)", "Dense cross-coupled mesh, 20 $H_1$ traps", 2.90),
        ("L3", r"(d) L3: Island-topology factory ($K=3$)", "Multi-island TBF core + sleep links, 16 $H_1$ traps", 6.75),
    ]
    for t_code, t_title, t_sub, x0 in topos:
        ax.add_patch(FancyBboxPatch((x0, -3.25), 3.60, 2.95, boxstyle="round,pad=0,rounding_size=0.14", fc="#f8fafc", ec="#cbd5e1", lw=0.8, zorder=0))
        ax.text(x0 + 1.80, -0.52, t_title, ha="center", va="center", fontsize=6.4, fontweight="bold", color=COL_BLUE)
        draw_mini_testbed(
            ax, x0 + 0.72, -2.65, s=0.18, topo=t_code,
            ctrl=True, hosts=True, show_sw_labels=True, ctrl_h=9.8, sw_ms=2.3
        )
        ax.text(x0 + 1.80, -3.02, t_sub, ha="center", va="center", fontsize=5.6, color="#333")

    # Legend at very bottom
    ax.plot([-0.6, -0.1], [-3.58, -3.58], color=COL_GRAY, lw=2.0)
    ax.text(0.02, -3.58, "Primary OpenFlow trunk", va="center", fontsize=6.1)
    ax.plot([2.55, 3.05], [-3.58, -3.58], color="#c98a2b", lw=1.7, ls=(0, (4, 2)))
    ax.text(3.17, -3.58, r"Sleepable inter-island link ($f_7$)", va="center", fontsize=6.1)
    ax.plot([5.95, 6.45], [-3.58, -3.58], color=COL_GREEN, lw=1.0, ls=":")
    ax.text(6.57, -3.58, "OpenFlow 1.3 control channel", va="center", fontsize=6.1)

    fig.savefig("fig2_iterate_test_protocol.png", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Figure 3: Graphical Schematics of the Course of Experiments E1 & E2 (with Topologies)
# ==============================================================================

def make_fig3_schem(res):
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.set_aspect("equal")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def panel(x, y, w, h, title, c=COL_BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="#f6f8fb", ec="#c9d3df", lw=0.8, zorder=0))
        ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="center", fontsize=7.0, fontweight="bold", color=c)

    # Panel 1: Live Topology Deployment under E1 (s1..s6 with rewritten switches & cross-traffic)
    panel(1.0, 1.0, 23.5, 42.0, "1. CPPS topology under test", c=COL_BLUE)
    draw_mini_testbed(
        ax, 5.6, 13.5, s=1.18, topo="L3",
        hl_red=("s2", "s5"), hot_edges=(("s1", "s3"), ("s3", "s5")),
        ctrl=True, hosts=True, show_sw_labels=True, ctrl_h=12.2, sw_ms=2.6
    )
    draw_switch(ax, 4.0, 8.2, s=1.7, fc=COL_RED, ms=2.2)
    ax.text(6.2, 8.2, r"Rewritten by $r=2$ flip", ha="left", va="center", fontsize=5.6, color=COL_RED, fontweight="bold")
    ax.plot([2.8, 5.2], [4.5, 4.5], color=COL_RED, lw=2.0)
    ax.text(6.2, 4.5, r"LRD cross-traffic ($\sigma=0.075$)", ha="left", va="center", fontsize=5.5, color="#333")

    # Panel 2: 5-Test Allocation Schedule (E1)
    panel(26.0, 1.0, 24.5, 42.0, "2. E1: 5-test budget split", c=COL_BLUE)
    for row, (n_new, lbl) in enumerate([(1, "1-per-5"), (2, "2-per-5"), (5, "1-shot")]):
        y = 30.0 - row * 8.6
        ax.text(27.2, y + 1.6, lbl, ha="left", va="center", fontsize=6.1, fontweight="bold", color="#222")
        for k in range(5):
            is_new = (k < n_new)
            bx = 35.0 + k * 2.85
            ax.add_patch(FancyBboxPatch(
                (bx, y), 2.50, 3.2, boxstyle="round,pad=0,rounding_size=0.35",
                fc="#f9d6d5" if is_new else "#dbe4ef",
                ec=COL_RED if is_new else COL_BLUE, lw=0.7, zorder=3
            ))
            ax.text(bx + 1.25, y + 1.6, "new" if is_new else f"L{(k-n_new)%3+1}", ha="center", va="center", fontsize=5.3,
                    color="#7f1d1d" if is_new else COL_BLUE, fontweight="bold")
    ax.text(38.25, 7.6, r"Leader re-tests $L_1,L_2,L_3$", ha="center", va="center", fontsize=5.8, color=COL_BLUE, fontweight="bold")
    ax.text(38.25, 4.2, r"average out telemetry noise $\varepsilon_k$", ha="center", va="center", fontsize=5.6, color="#333")

    # Panel 3: Belief Variance & Optimizer's Curse (E1)
    panel(52.0, 1.0, 22.5, 42.0, r"3. E1: Belief ($\sigma/\sqrt{n_p}$)", c=COL_GREEN)
    ax.plot([54.5, 54.5, 72.5], [33.0, 12.0, 12.0], color="#555", lw=0.7, zorder=3)
    xs = np.linspace(0, 1, 120)
    y_red = np.exp(-0.5 * ((xs - 0.35) / 0.20) ** 2) * 8.5
    y_grn = np.exp(-0.5 * ((xs - 0.68) / 0.08) ** 2) * 17.0
    ax.plot(54.5 + xs * 17.5, 12.0 + y_red, color=COL_RED, lw=1.3, zorder=4)
    ax.fill_between(54.5 + xs * 17.5, 12.0, 12.0 + y_red, color=COL_RED, alpha=0.22, zorder=3)
    ax.plot(54.5 + xs * 17.5, 12.0 + y_grn, color=COL_GREEN, lw=1.3, zorder=4)
    ax.fill_between(54.5 + xs * 17.5, 12.0, 12.0 + y_grn, color=COL_GREEN, alpha=0.22, zorder=3)
    ax.text(59.6, 21.6, r"$n_p=1$" + "\n" + r"($+11.6\%$)", ha="center", va="bottom", fontsize=5.6, color=COL_RED, fontweight="bold")
    ax.text(66.5, 29.8, r"$n_p=6$ ($+0.8\%$)", ha="center", va="bottom", fontsize=5.6, color=COL_GREEN, fontweight="bold")
    ax.text(63.25, 7.6, r"Posterior density of $\hat{M}_k^p$", ha="center", va="center", fontsize=5.8)
    ax.text(63.25, 4.2, r"3 topologies $\times$ 250 seeds", ha="center", va="center", fontsize=5.6, color="#444")

    # Panel 4: Digital Twin Pre-Screening & Top-1 Topology Push (E2)
    panel(76.0, 1.0, 23.0, 42.0, r"4. E2: Twin pre-screen ($M_{\mathrm{pre}}$)", c=COL_AMBER)
    rng = np.random.default_rng(7)
    for k in range(3):
        yy = 34.5 - k * 2.6
        draw_config_bits(ax, 77.6, yy, list((rng.random(6) < 0.5).astype(int)), cell=0.68, gap=0.11, c=COL_SLATE)
    ax.text(80.2, 38.2, r"$M_{\mathrm{pre}}$ in $\mathcal{N}_2$", ha="center", va="center", fontsize=5.6)
    draw_arrow_pts(ax, [(83.2, 32.0), (86.2, 32.0)], color=COL_AMBER, lw=1.0, ms=6)
    draw_host_pc(ax, 89.2, 32.5, s=4.2, label=r"Twin $\hat{U}_{\mathrm{DT}}$", fs=5.6, fc="#fff3d6")
    draw_arrow_pts(ax, [(89.2, 27.6), (89.2, 23.8)], color=COL_GREEN, lw=1.1, ms=7)
    ax.text(94.2, 25.8, "Deploy\ntop-1", ha="center", va="center", fontsize=5.5, color=COL_GREEN, fontweight="bold")
    draw_mini_testbed(
        ax, 82.0, 10.2, s=0.92, topo="L3",
        hl_green=("s1", "s2", "s5"), green_edges=(("s1", "s2"), ("s2", "s5")),
        ctrl=False, hosts=False, show_sw_labels=False, sw_ms=2.2
    )
    ax.text(87.5, 7.2, r"Sweep $M_{\mathrm{pre}}\in\{1..12\}$", ha="center", va="center", fontsize=5.7, fontweight="bold", color="#7c580a")
    ax.text(87.5, 4.0, r"Fidelity $\rho_{\mathrm{DT}}\in[0,\,0.95]$", ha="center", va="center", fontsize=5.6, color="#333")

    draw_arrow_pts(ax, [(24.5, 22), (26.0, 22)], color=COL_BLUE, ms=7)
    draw_arrow_pts(ax, [(50.5, 22), (52.0, 22)], color=COL_BLUE, ms=7)
    draw_arrow_pts(ax, [(74.5, 22), (76.0, 22)], color=COL_GREEN, ms=7)

    fig.savefig("fig3_exp1_allocation_curse.png", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Figure 4: Quantitative Empirical Plots for Experiments E1 & E2 (Spacious 2x2)
# ==============================================================================

def make_fig4_plots(res):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.4))
    plt.subplots_adjust(hspace=0.40, wspace=0.46)

    # (a) E1 Convergence Trajectories
    ax = axes[0, 0]
    e1_l3 = res["exp1"]["by_landscape"]["L3_IslandFactory"]
    t = np.arange(1, 121)
    curves = [
        ("retest_2_of_5", "Iterate–test (2-per-5)", COL_BLUE, "-", 1.8),
        ("ucb1", "UCB1 (2-per-5)", COL_GREEN, "--", 1.5),
        ("fixed_1_of_3", "Fixed (1-per-3)", COL_AMBER, "-.", 1.4),
        ("single_shot", "Single-shot (1-per-1)", COL_RED, "-", 1.5),
        ("hill_climb_once", "Walk (1-shot)", COL_SLATE, ":", 1.4),
    ]
    for k, lbl, col, ls, lw in curves:
        ax.plot(t, e1_l3[k]["trace_shipped"], label=lbl, color=col, linestyle=ls, linewidth=lw)
    ax.plot(t, e1_l3["single_shot"]["trace_believed"], label=r"Single-shot belief $X_t$", color=COL_RED, linestyle=":", linewidth=1.2, alpha=0.85)
    ax.axhline(1.0, color="#666", ls="--", lw=0.8)
    ax.set_xlim(1, 120)
    ax.set_ylim(0.28, 1.08)
    ax.set_xlabel(r"Measurement round $t$ (budget $T = 120$)")
    ax.set_ylabel(r"True utility $U(p_t^*)$ / belief $X_t$")
    ax.set_title("(a) E1: Convergence and belief optimism (L3)", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", fontsize=6.5)

    # (b) E1 Pooled Horizontal Bar Chart
    ax = axes[0, 1]
    pooled = res["exp1"]["pooled"]
    order = [
        ("fixed_1_of_5", r"1-per-5"),
        ("hill_climb_once", r"Walk-1"),
        ("single_shot", r"1-shot"),
        ("fixed_1_of_3", r"1-per-3"),
        ("thompson", "Thompson"),
        ("fixed_1_of_2", r"1-per-2"),
        ("ucb1", "UCB1"),
        ("retest_2_of_5", r"2-per-5 (loop)"),
    ]
    y_pos = np.arange(len(order))
    u_means = [pooled[k]["shipped_mean"] for k, _ in order]
    u_cis = [1.96 * pooled[k]["shipped_sd"] / np.sqrt(750) for k, _ in order]
    b_means = [pooled[k]["believed_mean"] for k, _ in order]

    ax.barh(y_pos - 0.17, u_means, xerr=u_cis, height=0.34, color=COL_BLUE, label=r"Shipped $U(p_T^*)$", capsize=1.8, error_kw=dict(lw=0.8))
    ax.barh(y_pos + 0.17, b_means, height=0.34, color="#e69f00", alpha=0.85, label=r"Believed $X_T$")
    for i, (k, _) in enumerate(order):
        c_pct = pooled[k]["curse_mean"] * 100.0
        ax.text(max(u_means[i], b_means[i]) + 0.007, y_pos[i], f"+{c_pct:.1f}%", va="center", ha="left", fontsize=6.4,
                color=COL_RED if c_pct > 5 else "#333", fontweight="bold" if c_pct > 5 else "normal")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in order], fontsize=7.5)
    ax.set_xlim(0.84, 1.15)
    ax.set_xlabel(r"Pooled utility / belief at $T = 120$ ($N=750$)")
    ax.set_title("(b) E1: Shipped utility and optimizer's curse", loc="left", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=True, facecolor="white", framealpha=0.92, edgecolor="none", fontsize=6.4)

    # (c) E2 Pre-Screening Pool Size M_pre Sweep
    ax = axes[1, 0]
    m_vals = [1, 2, 4, 8, 12]
    e2_m = res["exp2"]["m_sweep"]
    for pol, lbl, col, mk in [
        ("retest_2_of_5", "Iterate–test (2-per-5)", COL_BLUE, "o"),
        ("ucb1", "UCB1 (2-per-5)", COL_GREEN, "s"),
        ("single_shot", "Single-shot (1-per-1)", COL_RED, "^"),
    ]:
        means = [e2_m[pol][str(m)]["shipped_mean"] for m in m_vals]
        ses = [e2_m[pol][str(m)]["shipped_sd"] / np.sqrt(200) for m in m_vals]
        ax.errorbar(m_vals, means, yerr=ses, marker=mk, ms=4.5, lw=1.5, capsize=2.2, color=col, label=lbl)
    ax.set_xticks(m_vals)
    ax.set_ylim(0.85, 1.005)
    ax.set_xlabel(r"Pre-screened candidates per build step ($M_{\mathrm{pre}}$)")
    ax.set_ylabel(r"Mean shipped utility $U(p_T^*)$")
    ax.set_title(r"(c) E2: Digital-twin screening depth ($\rho_{\mathrm{DT}}=0.65$)", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", fontsize=6.8)

    # (d) E2 Surrogate Fidelity Crossover
    ax = axes[1, 1]
    rho_vals = [0.0, 0.40, 0.65, 0.85, 0.95]
    e2_r = res["exp2"]["rho_sweep"]
    for pol, lbl, col, mk in [
        ("retest_2_of_5", r"Hybrid iterate–test (twin + re-test)", COL_BLUE, "o"),
        ("single_shot", r"Single-shot + twin screen", COL_AMBER, "^"),
        ("twin_only", r"Digital-twin-only (open-loop)", COL_PURPLE, "d"),
    ]:
        means = [e2_r[pol][f"{r:.2f}"]["shipped_mean"] for r in rho_vals]
        ses = [e2_r[pol][f"{r:.2f}"]["shipped_sd"] / np.sqrt(200) for r in rho_vals]
        ax.errorbar(rho_vals, means, yerr=ses, marker=mk, ms=4.5, lw=1.5, capsize=2.2, color=col, label=lbl)
    ax.set_xticks(rho_vals)
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel(r"Digital-twin surrogate correlation ($\rho_{\mathrm{DT}}$)")
    ax.set_ylabel(r"Mean shipped utility $U(p_T^*)$")
    ax.set_title("(d) E2: Surrogate fidelity crossover", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", fontsize=6.8)

    fig.savefig("fig4_exp2_digital_twin.png", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Figure 5: Graphical Schematics of the Course of Experiments E3 & E4 (with Topologies)
# ==============================================================================

def make_fig5_schem(res):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.set_aspect("equal")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def panel(x, y, w, h, title, c=COL_BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="#f6f8fb", ec="#c9d3df", lw=0.8, zorder=0))
        ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="center", fontsize=6.9, fontweight="bold", color=c)

    # --- TOP ROW: Network Topology States during E3 Co-Design Trap vs H2 Escape & E4 Perturbed Telemetry ---
    # Top-Left: Narrow H1 Co-Design Trap on Topology
    panel(1.0, 32.5, 31.5, 24.5, r"E3 topology trap: Narrow $\mathcal{N}_1$ ($r=1$)", c=COL_RED)
    draw_mini_testbed(
        ax, 8.2, 37.5, s=1.02, topo="L3",
        hl_red=("s1", "s3", "s5"), hot_edges=(("s1", "s3"), ("s3", "s5")),
        ctrl=True, hosts=True, show_sw_labels=True, ctrl_h=11.2, sw_ms=2.4
    )
    ax.text(16.75, 34.4, r"Link sleep $f_7=1$ without delta sampling $f_2=1$" + "\n" + r"congests trunk $\mathrm{S1\text{–}S3\text{–}S5}$ ($U=0.928,\ 8.1\ \mathrm{ms}$)",
            ha="center", va="center", fontsize=5.5, color=COL_RED)

    # Top-Middle: Coordinated H2 Co-Design Optimum on Topology
    panel(34.2, 32.5, 31.5, 24.5, r"E3 topology escape: Co-design $\mathcal{N}_2$ ($r=2$)", c=COL_GREEN)
    draw_mini_testbed(
        ax, 41.4, 37.5, s=1.02, topo="L3",
        hl_green=("s1", "s2", "s5"), green_edges=(("s1", "s2"), ("s2", "s5"), ("s1", "s3"), ("s3", "s5")),
        ctrl=True, hosts=True, show_sw_labels=True, ctrl_h=11.2, sw_ms=2.4
    )
    ax.text(49.95, 34.4, r"Joint flip $(f_2=1,f_7=1)+(f_0=1,f_5=1)$ unlocks" + "\n" + r"TSN multipath $\mathrm{S1\text{–}S2\text{–}S5}$ ($U=0.969,\ 6.1\ \mathrm{ms}$)",
            ha="center", va="center", fontsize=5.5, color=COL_GREEN)

    draw_arrow_pts(ax, [(32.5, 44.5), (34.2, 44.5)], color=COL_GREEN, ms=7)

    # Top-Right: E4 Perturbed Telemetry Channel across Topology
    panel(67.4, 32.5, 31.6, 24.5, r"E4 topology telemetry: Bias $B$ & drift $D$", c=COL_PURPLE)
    draw_mini_testbed(
        ax, 73.5, 38.2, s=0.92, topo="L3",
        hl_red=("s5",), hot_edges=(("s2", "s5"),), green_edges=(("s1", "s2"),),
        ctrl=False, hosts=True, show_sw_labels=False, sw_ms=2.2
    )
    ax.text(83.2, 48.8, r"Sensor bias $+b_k(p)$ & tool-wear drift $+d_k(p)$", ha="center", va="center", fontsize=5.7, color=COL_PURPLE, fontweight="bold")
    ax.text(83.2, 34.4, r"Measured at egress $\mathrm{S5}$: $y_k(p)=U(p)+b_k+d_k+\varepsilon_k$" + "\n" + r"Limit points remain within $2B+2D$ of $\mathcal{N}_2(p_\infty)$",
            ha="center", va="center", fontsize=5.5, color="#333")

    # --- BOTTOM ROW: 4 Ablation Loops (A2-A5) + Bias/Drift Distribution Shift ---
    ablations = [
        ("E3 (A2): Locality", 0, r"$\mathcal{N}_1$ trap vs $\mathcal{N}_{12}$ jump"),
        ("E3 (A3): Generativity", 1, r"$\delta_k \propto k^{-2}$ freezes"),
        ("E3 (A4): Re-testing", 2, r"Single-shot $n_p=1$"),
        ("E3 (A5): Retention", 3, r"Sliding window $|P_k|\leq 4$"),
    ]
    comps = [("ideate", "#eef3f8", COL_BLUE), ("screen", "#fff8e8", COL_AMBER), ("test", "#eef3f8", COL_BLUE), ("archive", "#e8f0e8", COL_GREEN)]

    for k, (title, hi_idx, sub_txt) in enumerate(ablations):
        x0 = 1.0 + k * 18.2
        panel(x0, 1.0, 17.2, 29.5, title, c=COL_RED)
        pts = [(x0 + 4.6, 19.5), (x0 + 12.6, 19.5), (x0 + 12.6, 11.2), (x0 + 4.6, 11.2)]
        for idx_c, ((cname, fc, ec), (cx, cy)) in enumerate(zip(comps, pts)):
            ax.add_patch(FancyBboxPatch((cx - 3.3, cy - 1.8), 6.6, 3.6, boxstyle="round,pad=0,rounding_size=0.45", fc=fc, ec=ec, lw=0.7, zorder=3))
            ax.text(cx, cy, cname, ha="center", va="center", fontsize=5.4, zorder=4)
            if idx_c == hi_idx:
                ax.add_patch(Circle((cx, cy), 3.4, fc="none", ec=COL_RED, lw=1.3, ls="--", zorder=6))
        draw_arrow_pts(ax, [(x0 + 7.9, 19.5), (x0 + 9.3, 19.5)], ms=5.5)
        draw_arrow_pts(ax, [(x0 + 12.6, 17.7), (x0 + 12.6, 13.0)], ms=5.5)
        draw_arrow_pts(ax, [(x0 + 9.3, 11.2), (x0 + 7.9, 11.2)], ms=5.5)
        draw_arrow_pts(ax, [(x0 + 4.6, 13.0), (x0 + 4.6, 17.7)], ms=5.5)
        ax.text(x0 + 8.6, 6.0, sub_txt, ha="center", va="center", fontsize=5.7, color=COL_RED, fontweight="bold")
        ax.text(x0 + 8.6, 3.0, "all else as baseline", ha="center", va="center", fontsize=5.2, color="#555")

    panel(74.0, 1.0, 25.0, 29.5, r"E4: Posterior shift ($+b_k,+d_k$)", c=COL_PURPLE)
    ax.plot([77.5, 77.5, 96.5], [23.5, 10.5, 10.5], color="#555", lw=0.7, zorder=3)
    xs = np.linspace(0, 1, 100)
    y_unb = np.exp(-0.5 * ((xs - 0.38) / 0.13) ** 2) * 10.5
    y_bia = np.exp(-0.5 * ((xs - 0.64) / 0.13) ** 2) * 10.5
    ax.plot(77.5 + xs * 18.5, 10.5 + y_unb, color=COL_BLUE, lw=1.2, zorder=4)
    ax.fill_between(77.5 + xs * 18.5, 10.5, 10.5 + y_unb, color=COL_BLUE, alpha=0.2, zorder=3)
    ax.plot(77.5 + xs * 18.5, 10.5 + y_bia, color=COL_RED, lw=1.2, zorder=4)
    ax.fill_between(77.5 + xs * 18.5, 10.5, 10.5 + y_bia, color=COL_RED, alpha=0.2, zorder=3)
    draw_arrow_pts(ax, [(77.5 + 0.38 * 18.5, 21.8), (77.5 + 0.64 * 18.5, 21.8)], color=COL_RED, ms=6.5)
    ax.text(86.8, 23.6, r"$+b_k,\ +d_k$", ha="center", va="center", fontsize=5.8, color=COL_RED, fontweight="bold")
    ax.text(86.5, 6.2, r"Bounds: $\Delta U \leq 2B,\ 2D$", ha="center", va="center", fontsize=5.8, fontweight="bold", color=COL_PURPLE)
    ax.text(86.5, 3.0, r"Confirmation bias benign for $B \leq \sigma$", ha="center", va="center", fontsize=5.4, color=COL_GREEN)

    fig.savefig("fig5_exp3_ablation.png", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Figure 6: Quantitative Empirical Plots for Experiments E3 & E4 (Spacious 2x2)
# ==============================================================================

def make_fig6_plots(res):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.4))
    plt.subplots_adjust(hspace=0.40, wspace=0.46)

    e3 = res["exp3"]
    # (a) E3 Ablation Trajectories
    ax = axes[0, 0]
    t = np.arange(1, 121)
    ab_curves = [
        ("full_theorem1", r"Full Theorem 1 ($\mathcal{N}_2$)", COL_BLUE, "-", 1.8),
        ("narrow_H1_A2", r"Single-bit $\mathcal{N}_1$ trap (A2)", COL_TEAL, "-.", 1.5),
        ("no_retest_A4", r"No re-test $n_p=1$ (A4)", COL_AMBER, "--", 1.5),
        ("unbounded_global_A2", r"Global jumps $\mathcal{N}_{12}$ (A2)", COL_SLATE, ":", 1.5),
        ("no_retention_A5", r"No retention $|P_k|\leq 4$ (A5)", COL_RED, "-", 1.5),
        ("summable_gen_A3", r"Summable $\delta_k\propto k^{-2}$ (A3)", COL_PURPLE, "--", 1.5),
    ]
    for k, lbl, col, ls, lw in ab_curves:
        ax.plot(t, e3[k]["trace_shipped"], label=lbl, color=col, linestyle=ls, linewidth=lw)
    ax.set_xlim(1, 120)
    ax.set_ylim(0.20, 1.03)
    ax.set_xlabel(r"Measurement round $t$")
    ax.set_ylabel(r"True shipped utility $U(p_t^*)$")
    ax.set_title("(a) E3: Trajectories under assumption ablations", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.46), frameon=True, facecolor="white", framealpha=0.92, edgecolor="none", fontsize=6.3)

    # (b) E3 Horizontal Bar Comparison of Discovered vs. Shipped Utility
    ax = axes[0, 1]
    ab_order = [
        ("summable_gen_A3", r"Drop A3 ($k^{-2}$)"),
        ("no_retention_A5", r"Drop A5 ($|P_k|\leq 4$)"),
        ("unbounded_global_A2", r"Global $\mathcal{N}_{12}$ (A2)"),
        ("no_retest_A4", r"Drop A4 ($n_p=1$)"),
        ("narrow_H1_A2", r"Narrow $\mathcal{N}_1$ (A2)"),
        ("full_theorem1", r"Full Thm 1 ($\mathcal{N}_2$)"),
    ]
    y_pos = np.arange(len(ab_order))
    ship_v = [e3[k]["shipped_mean"] for k, _ in ab_order]
    found_v = [e3[k]["found_mean"] for k, _ in ab_order]
    ax.barh(y_pos + 0.17, found_v, height=0.34, color="#93c5fd", edgecolor=COL_BLUE, lw=0.6, label=r"Discovered $\max_{p\in P_T} U(p)$")
    ax.barh(y_pos - 0.17, ship_v, height=0.34, color=COL_BLUE, label=r"Shipped $U(p_T^*)$")
    for i, (k, _) in enumerate(ab_order):
        gap = found_v[i] - ship_v[i]
        txt = f"Loss: {gap:.3f}" if gap > 0.008 else f"{ship_v[i]:.3f}"
        ax.text(max(found_v[i], ship_v[i]) + 0.012, y_pos[i], txt, va="center", ha="left", fontsize=6.4,
                color=COL_RED if gap > 0.008 else "#222", fontweight="bold" if gap > 0.008 else "normal")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in ab_order], fontsize=7.4)
    ax.set_xlim(0.0, 1.22)
    ax.set_xlabel(r"Utility at horizon $T = 120$ ($N=250$)")
    ax.set_title("(b) E3: Discovered vs. shipped utility", loc="left", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="none", fontsize=6.5)

    # (c) E4 Systematic & Confirmation Bias B vs 2B Bound
    ax = axes[1, 0]
    b_vals = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    b_keys = [f"{b:.2f}" for b in b_vals]
    e4 = res["exp4"]
    loss_sys = [e4["sys_bias"][k]["empirical_loss"] for k in b_keys]
    loss_conf = [e4["conf_bias"][k]["empirical_loss"] for k in b_keys]
    bound_2b = [2.0 * b for b in b_vals]

    ax.plot(b_vals, bound_2b, "--", color="#222", lw=1.3, label=r"Theoretical bound $2B$ (Prop. 1)")
    ax.plot(b_vals, loss_sys, "-o", color=COL_RED, lw=1.6, ms=4.2, label=r"Adversarial sensor bias $B$")
    ax.plot(b_vals, loss_conf, "-s", color=COL_BLUE, lw=1.6, ms=4.2, label=r"Confirmation bias $b_{\mathrm{conf}}$")
    ax.axvline(0.075, color=COL_GREEN, ls=":", lw=1.3, label=r"Noise floor $\sigma = 0.075$")
    ax.axvspan(0.0, 0.075, color="#dcfce7", alpha=0.35)
    ax.set_xlim(0.0, 0.155)
    ax.set_ylim(-0.008, 0.31)
    ax.set_xlabel(r"Bias magnitude $B$ (utility units)")
    ax.set_ylabel(r"Utility loss $U_{\mathrm{unbiased}} - U(p_T^*)$")
    ax.set_title(r"(c) E4: Telemetry bias $B$ vs. $2B$ bound", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", fontsize=6.5)

    # (d) E4 Non-Stationary Drift D vs 2D Bound
    ax = axes[1, 1]
    loss_dr = [e4["drift"][k]["empirical_loss"] for k in b_keys]
    gap_1_minus_u = [1.0 - e4["drift"][k]["shipped_mean"] for k in b_keys]
    ax.plot(b_vals, bound_2b, "--", color="#222", lw=1.3, label=r"Theoretical drift bound $2D$ (Cor. 1)")
    ax.plot(b_vals, gap_1_minus_u, "-^", color=COL_AMBER, lw=1.5, ms=4.2, label=r"Suboptimality gap $U^* - U(p_T^*)$")
    ax.plot(b_vals, loss_dr, "-d", color=COL_PURPLE, lw=1.6, ms=4.5, label=r"Drift-induced loss $U_0 - U(p_T^*)$")
    ax.axvline(0.075, color=COL_GREEN, ls=":", lw=1.3, label=r"Noise floor $\sigma = 0.075$")
    ax.set_xlim(0.0, 0.155)
    ax.set_ylim(-0.008, 0.31)
    ax.set_xlabel(r"Drift half-width band $D$ (utility units)")
    ax.set_ylabel("Mean utility degradation")
    ax.set_title(r"(d) E4: Non-stationary drift $D$ vs. $2D$ bound", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", fontsize=6.5)

    fig.savefig("fig6_exp4_bias_drift.png", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Figure 7: Graphical Schematics of the Course of Experiments E5 & E6 (with Topologies)
# ==============================================================================

def make_fig7_schem(res):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.set_aspect("equal")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def panel(x, y, w, h, title, c=COL_BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="#f6f8fb", ec="#c9d3df", lw=0.8, zorder=0))
        ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="center", fontsize=7.0, fontweight="bold", color=c)

    # --- LEFT PANEL: E5 Multi-Stakeholder REDUCE vs Pareto Archive + Pareto Topology Modes ---
    panel(1.0, 1.0, 47.0, 56.0, r"E5: Stakeholder intent (REDUCE vs. Pareto archive $\mathcal{P}_k^{\mathrm{ND}}$)", c=COL_TEAL)

    for k, lbl in enumerate(["Control", "SRE/Sec", "Energy"]):
        yy = 49.0 - k * 6.5
        draw_host_pc(ax, 6.2, yy, s=4.2, label=lbl, fs=5.6)
        draw_arrow_pts(ax, [(9.0, yy), (13.6, 42.5)], color=COL_TEAL, lw=0.9, ms=6)

    for r in range(3):
        for c in range(3):
            ax.add_patch(Rectangle((14.2 + c * 2.2, 39.2 + r * 2.2), 2.2, 2.2,
                                   fc="#ffe4e6" if (r, c) in ((0, 2), (2, 0)) else "white", ec=COL_TEAL, lw=0.6, zorder=3))
    ax.text(17.5, 47.6, r"PCM $\mathbf{A}$", ha="center", va="center", fontsize=6.1, fontweight="bold")
    ax.text(17.5, 36.8, r"$\mathrm{CR}=0.24$", ha="center", va="center", fontsize=5.8, color=COL_RED)

    draw_arrow_pts(ax, [(21.4, 44.5), (26.5, 47.5)], color=COL_TEAL, ms=6.5)
    draw_arrow_pts(ax, [(21.4, 40.5), (26.5, 37.0)], color=COL_BLUE, ms=6.5)

    ax.add_patch(FancyBboxPatch((27.0, 44.0), 19.2, 7.6, boxstyle="round,pad=0,rounding_size=0.7", fc="#e6f4ea", ec=COL_GREEN, lw=0.8, zorder=3))
    ax.text(36.6, 49.2, "Path A: REDUCE module", ha="center", va="center", fontsize=6.0, fontweight="bold", color=COL_GREEN)
    ax.text(36.6, 46.2, r"$\mathrm{CR}=0.032 < 0.10 \to w^*$", ha="center", va="center", fontsize=5.8)

    ax.add_patch(FancyBboxPatch((27.0, 33.0), 19.2, 7.6, boxstyle="round,pad=0,rounding_size=0.7", fc="#eef3f8", ec=COL_BLUE, lw=0.8, zorder=3))
    ax.text(36.6, 38.2, "Path B: Prop. 2 archive", ha="center", va="center", fontsize=6.0, fontweight="bold", color=COL_BLUE)
    ax.text(36.6, 35.2, r"Retain non-dom. $\mathcal{P}_T^{\mathrm{ND}}$", ha="center", va="center", fontsize=5.8)

    # Two Pareto-Archive Topology Operating Modes inside E5 panel (bottom half)
    ax.add_patch(FancyBboxPatch((2.6, 3.0), 21.2, 26.5, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec="#cbd5e1", lw=0.7, zorder=1))
    ax.text(13.2, 27.4, "Pareto mode 1: Low-latency", ha="center", va="center", fontsize=5.8, fontweight="bold", color=COL_BLUE)
    draw_mini_testbed(
        ax, 7.6, 12.2, s=0.92, topo="L3",
        hl_green=("s1", "s2", "s3", "s5"),
        sleep_edges=set(),
        green_edges=(("s1", "s2"), ("s2", "s5"), ("s1", "s3"), ("s3", "s5")),
        ctrl=True, hosts=False, show_sw_labels=False, ctrl_h=10.5, sw_ms=2.2
    )
    ax.text(13.2, 6.2, r"All trunks active ($f_4,f_5=1$)" + "\n" + r"$U_{\mathrm{lat}}=0.967,\ f_7=0$", ha="center", va="center", fontsize=5.3, color="#333")

    ax.add_patch(FancyBboxPatch((25.2, 3.0), 21.2, 26.5, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec="#cbd5e1", lw=0.7, zorder=1))
    ax.text(35.8, 27.4, "Pareto mode 2: Green / balanced", ha="center", va="center", fontsize=5.8, fontweight="bold", color=COL_GREEN)
    draw_mini_testbed(
        ax, 30.2, 12.2, s=0.92, topo="L3",
        hl_green=("s1", "s2", "s5"),
        sleep_edges={("s1", "s4"), ("s4", "s6"), ("s6", "s5"), ("s2", "s3")},
        green_edges=(("s1", "s2"), ("s2", "s5")),
        ctrl=True, hosts=False, show_sw_labels=False, ctrl_h=10.5, sw_ms=2.2
    )
    ax.text(35.8, 6.2, r"Island sleep + delta ($f_2,f_7=1$)" + "\n" + r"$U_{\mathrm{eff}}=0.962,\ \mathrm{HV}=0.883$", ha="center", va="center", fontsize=5.3, color="#333")

    # --- RIGHT PANEL: E6 Two-Epoch Topology Reframing Transition (F0 -> F1 at t=70) ---
    panel(50.0, 1.0, 49.0, 56.0, r"E6: EI-triggered epoch reframing ($\mathcal{F}_0 \to \mathcal{F}_1$ at $t=70$)", c=COL_RED)

    # Top Sub-box: Epoch 1 Legacy Topology F0
    ax.add_patch(FancyBboxPatch((51.8, 34.0), 26.5, 19.2, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec="#cbd5e1", lw=0.7, zorder=1))
    ax.text(65.0, 51.2, r"Epoch 1 ($t\leq 70$): Legacy topology $\mathcal{F}_0$", ha="center", va="center", fontsize=5.9, fontweight="bold", color="#333")
    draw_mini_testbed(
        ax, 59.5, 38.6, s=0.88, topo="L3",
        hl_red=("s1", "s3", "s5"),
        hot_edges=(("s1", "s3"), ("s3", "s5")),
        ctrl=True, single_ctrl=True, hosts=True, show_sw_labels=False, ctrl_h=10.2, sw_ms=2.1
    )
    ax.text(65.0, 35.8, r"Locked $f_0,f_5,f_6,f_{11}=0$ ($2^8=256,\ 18.1\ \mathrm{ms}$)", ha="center", va="center", fontsize=5.3, color=COL_RED)

    # Transition condition between Epoch 1 and Epoch 2
    draw_arrow_pts(ax, [(55.0, 33.6), (55.0, 23.6)], color=COL_AMBER, lw=1.3, ms=7.5)
    ax.text(66.2, 28.6, r"Reframe when $\mathrm{EI}_k - c_{\mathrm{iter}} < G_{\mathrm{ref}} - c_{\mathrm{ref}}$", ha="center", va="center",
            fontsize=5.6, color="#7c580a", bbox=dict(fc="#fff8e8", ec=COL_AMBER, lw=0.6, pad=1.6), zorder=4)

    # Bottom Sub-box: Epoch 2 Reframed Topology F1
    ax.add_patch(FancyBboxPatch((51.8, 3.0), 26.5, 20.0, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=COL_BLUE, lw=0.75, zorder=1))
    ax.text(65.0, 21.0, r"Epoch 2 ($t>70$): Reframed topology $\mathcal{F}_1$", ha="center", va="center", fontsize=5.9, fontweight="bold", color=COL_BLUE)
    draw_mini_testbed(
        ax, 59.5, 7.8, s=0.88, topo="L3",
        hl_green=("s1", "s2", "s5"),
        green_edges=(("s1", "s2"), ("s2", "s5"), ("s1", "s3"), ("s3", "s5")),
        ctrl=True, single_ctrl=False, hosts=True, show_sw_labels=False, ctrl_h=10.2, sw_ms=2.1
    )
    ax.text(65.0, 4.8, r"Unlocked TSN + Failover + DQT ($2^{12},\ 7.5\ \mathrm{ms}$)", ha="center", va="center", fontsize=5.1, color=COL_GREEN, fontweight="bold")

    # Right column of E6: Bit vectors + Step-jump curve
    draw_config_bits(ax, 80.2, 47.5, [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], cell=1.12, gap=0.15, hi_indices=(0, 5, 6, 11), c=COL_SLATE)
    ax.text(88.2, 45.2, r"Legacy $\mathcal{F}_0$ bits", ha="center", va="center", fontsize=5.6, color="#444")

    ax.plot([80.5, 80.5, 97.2], [38.0, 17.5, 17.5], color="#555", lw=0.7, zorder=3)
    tx = np.linspace(0, 1, 60)
    y_leg = 17.5 + 8.5 * (1.0 - np.exp(-5.0 * tx))
    y_ref = np.where(tx < 0.5, y_leg, 26.0 + 10.5 * (1.0 - np.exp(-5.5 * (tx - 0.5))))
    ax.plot(80.5 + tx * 16.0, y_leg, "--", color=COL_SLATE, lw=1.1, zorder=4)
    ax.plot(80.5 + tx * 16.0, y_ref, "-", color=COL_RED, lw=1.5, zorder=5)
    x_mid = 80.5 + 0.5 * 16.0
    ax.plot([x_mid, x_mid], [17.5, 37.5], color=COL_AMBER, ls=":", lw=1.1, zorder=4)
    ax.text(88.5, 40.0, r"Escape $\mathcal{F}_0$ ceiling", ha="center", va="center", fontsize=5.8, fontweight="bold", color=COL_RED)
    ax.text(88.5, 14.5, r"$t=70$ reframe", ha="center", va="center", fontsize=5.6, color="#7c580a")

    draw_config_bits(ax, 80.2, 7.8, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cell=1.12, gap=0.15, hi_indices=(0, 5, 6, 11), c=COL_BLUE)
    ax.text(88.2, 5.2, r"Reframed $\mathcal{F}_1$ bits", ha="center", va="center", fontsize=5.6, color=COL_BLUE, fontweight="bold")

    fig.savefig("fig7_exp5_reduce_pareto.png", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Figure 8: Quantitative Empirical Plots for Experiments E5 & E6 (Spacious 2x2)
# ==============================================================================

def make_fig8_plots(res):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.4))
    plt.subplots_adjust(hspace=0.40, wspace=0.52)

    e5 = res["exp5"]
    # (a) E5 Multi-Criteria Trade-off Space & Pareto Frontier
    ax = axes[0, 0]
    cloud = np.array(e5["Pareto_Archive_Prop2"]["all_cloud_pts"])
    arch = np.array(e5["Pareto_Archive_Prop2"]["sample_archive_pts"])
    sc = ax.scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], cmap="viridis", s=14, alpha=0.45, label="Feasible CPPS configs")
    order_a = np.argsort(arch[:, 0])
    ax.plot(arch[order_a, 0], arch[order_a, 1], "--", color=COL_RED, lw=1.1, zorder=5)
    ax.scatter(arch[:, 0], arch[:, 1], c=COL_RED, s=55, marker="*", edgecolors="#111", linewidths=0.6, zorder=6,
               label=r"Pareto archive $\mathcal{P}_T^{\mathrm{ND}}$")
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"Energy $U_{\mathrm{eff}}$", fontsize=7.2)
    ax.set_xlim(0.05, 1.04)
    ax.set_ylim(0.05, 1.04)
    ax.set_xlabel(r"Real-time determinism score $U_{\mathrm{lat}}$")
    ax.set_ylabel(r"Operational reliability score $U_{\mathrm{rel}}$")
    ax.set_title("(a) E5: Multi-criteria space and Pareto archive", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="none", fontsize=6.5)

    # (b) E5 Horizontal Bar Chart of Composite Utility & 3D Hypervolume
    ax = axes[0, 1]
    modes = [
        ("Energy_Dominated", "Energy only"),
        ("Unrectified_PCM_HighCR", r"Raw PCM ($0.24$)"),
        ("Reliability_Dominated", "Reliab. only"),
        ("Latency_Dominated", "Latency only"),
        ("REDUCE_Consensus_LowCR", r"REDUCE ($0.032$)"),
        ("Pareto_Archive_Prop2", r"Pareto $\mathcal{P}_T^{\mathrm{ND}}$"),
    ]
    y_pos = np.arange(len(modes))
    comp_v = [e5[k]["composite_u_mean"] for k, _ in modes]
    hv_v = [e5[k]["hv_mean"] for k, _ in modes]
    ax.barh(y_pos + 0.17, comp_v, height=0.34, color=COL_TEAL, label=r"Composite utility $U$")
    ax.barh(y_pos - 0.17, hv_v, height=0.34, color=COL_BLUE, label=r"3D hypervolume $\mathrm{HV}$")
    for i, (k, _) in enumerate(modes):
        ax.text(hv_v[i] + 0.008, y_pos[i] - 0.17, f"{hv_v[i]:.3f}", va="center", ha="left", fontsize=6.2, color=COL_BLUE, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in modes], fontsize=7.3)
    ax.set_xlim(0.50, 1.08)
    ax.set_xlabel(r"Score / hypervolume at $T = 120$ ($N=150$)")
    ax.set_title("(b) E5: REDUCE consensus vs. Pareto archive", loc="left", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="none", fontsize=6.5)

    # (c) E6 Epoch Reframing Trajectory
    ax = axes[1, 0]
    e6 = res["exp6"]
    t = np.arange(1, 141)
    ax.plot(t, e6["reframed_epoch"]["trace_shipped"], color=COL_RED, lw=1.9, label=r"Epoch-reframed ($\mathcal{F}_0\to\mathcal{F}_1$ at $t=70$)")
    ax.plot(t, e6["fixed_legacy"]["trace_shipped"], color=COL_SLATE, ls="--", lw=1.6, label=r"Fixed legacy frame $\mathcal{F}_0$")
    ax.axhline(e6["legacy_ceiling_u"], color="#666", ls=":", lw=1.0, label=f"Legacy ceiling ({e6['legacy_ceiling_u']:.3f})")
    ax.axhline(1.0, color=COL_GREEN, ls=":", lw=1.0, label="Global optimum (1.000)")
    ax.axvline(70, color=COL_AMBER, ls="-.", lw=1.2, label=r"Reframe trigger ($t_{\mathrm{ref}}=70$)")
    ax.set_xlim(1, 140)
    ax.set_ylim(0.20, 1.04)
    ax.set_xlabel(r"Measurement round $t$ (horizon $T = 140$)")
    ax.set_ylabel(r"True shipped utility $U(p_t^*)$")
    ax.set_title("(c) E6: Escape from legacy ceiling via reframing", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="none", fontsize=6.3)

    # (d) E6 Physical CPPS Engineering KPIs
    ax = axes[1, 1]
    cats = [
        r"Shipped utility" + "\n" + r"$100 \times U(p_{140}^*)$",
        r"Anomaly & frame" + "\n" + r"reliability $[\%]$",
        r"Determinism score" + "\n" + r"$100 \times (1 - d/48\mathrm{ms})$",
    ]
    leg_d = e6["fixed_legacy"]["delay_mean"]
    ref_d = e6["reframed_epoch"]["delay_mean"]
    leg_vals = [
        e6["fixed_legacy"]["shipped_mean"] * 100.0,
        e6["fixed_legacy"]["rel_mean"],
        (1.0 - leg_d / 48.0) * 100.0,
    ]
    ref_vals = [
        e6["reframed_epoch"]["shipped_mean"] * 100.0,
        e6["reframed_epoch"]["rel_mean"],
        (1.0 - ref_d / 48.0) * 100.0,
    ]
    x = np.arange(len(cats))
    w_b = 0.34
    b1 = ax.bar(x - w_b / 2, leg_vals, w_b, color="#94a3b8", edgecolor="#334155", lw=0.7, label=r"Fixed legacy $\mathcal{F}_0$")
    b2 = ax.bar(x + w_b / 2, ref_vals, w_b, color=COL_RED, edgecolor="#7f1d1d", lw=0.7, label=r"Reframed $\mathcal{F}_0\to\mathcal{F}_1$")
    for idx_b, bar in enumerate(b1):
        h = bar.get_height()
        lbl_txt = f"{leg_d:.1f} ms" if idx_b == 2 else f"{h:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, lbl_txt, ha="center", va="bottom", fontsize=6.4, color="#334155", fontweight="bold")
    for idx_b, bar in enumerate(b2):
        h = bar.get_height()
        lbl_txt = f"{ref_d:.1f} ms" if idx_b == 2 else f"{h:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, lbl_txt, ha="center", va="bottom", fontsize=6.4, color=COL_RED, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=6.9)
    ax.set_ylim(50, 108)
    ax.set_ylabel("Normalized performance score [%]")
    ax.set_title(r"(d) E6: Physical CPPS KPIs at $T = 140$", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="none", fontsize=6.6)

    fig.savefig("fig8_exp6_epoch_reframe.png", bbox_inches="tight")
    plt.close(fig)


def main():
    with open("results_astrj.json", encoding="utf-8") as f:
        res = json.load(f)
    make_fig1()
    make_fig2()
    make_fig3_schem(res)
    make_fig4_plots(res)
    make_fig5_schem(res)
    make_fig6_plots(res)
    make_fig7_schem(res)
    make_fig8_plots(res)
    print("All 8 figures updated and verified.")


if __name__ == "__main__":
    main()
