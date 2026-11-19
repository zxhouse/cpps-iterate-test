"""
Programmable Simulation Testbed for Convergent Iterate-Test Optimization
of SDN-Enabled Industrial IoT (IIoT) and Cyber-Physical Production Systems (CPPS).

Implements the formal Iterate-Test Core theoretical framework from:
Kuras, P., Michalski, A., Gerka, A., & Organisciak, P. (2026).
"When Does Design Thinking Work? A Convergence Theorem for the Iterate-Test Core",
arXiv preprint arXiv:2611.03034 (http://arxiv.org/abs/2611.03034).

Tailored for Advances in Science and Technology Research Journal (ASTRJ):
- 12 binary configuration knobs (2^12 = 4,096 prototypes) across 3 coupled CPPS layers:
  Layer I:   Shop-Floor IIoT Sensing & Telemetry Framing (f0..f3)
  Layer II:  Programmable SDN & Island-Topology Transport (f4..f7)
  Layer III: Edge-Cloud Compute, ML Anomaly Detection & Quality Control (f8..f11)
- Three distinct industrial CPPS topologies (L1: Modular Smart Assembly Line,
  L2: Coupled CNC/Planarization Cell, L3: Multi-Cell Island-Topology Factory Backbone).
- Six comprehensive experiments (E1-E6) with parametric (mean +/- SD) and
  non-parametric (median, IQR, Wilcoxon signed-rank, Friedman) statistics.
"""

import json
import time
import numpy as np
from scipy import stats


# ==============================================================================
# 1. Configuration Space Definition (12 Binary Engineering Features, N = 4096)
# ==============================================================================

FEATURES = [
    # Layer I: Shop-Floor IIoT Sensing & Telemetry Framing
    {"id": "f0", "code": "OPCUA_PubSub", "layer": "IIoT Sensing",
     "desc": "Brokerless UDP OPC UA PubSub (1) vs. Polling TCP Client-Server (0)"},
    {"id": "f1", "code": "Protobuf_Framing", "layer": "IIoT Sensing",
     "desc": "Compact gRPC/Protobuf binary serialization (1) vs. Verbose REST/JSON (0)"},
    {"id": "f2", "code": "Adaptive_Sampling", "layer": "IIoT Sensing",
     "desc": "Event-triggered delta telemetry sampling (1) vs. Fixed high-rate periodic (0)"},
    {"id": "f3", "code": "BLE_UWB_Sync", "layer": "IIoT Sensing",
     "desc": "Multipath-compensated BLE/UWB asset sync (1) vs. Unsynchronized RSSI (0)"},

    # Layer II: Programmable SDN & Island-Topology Transport
    {"id": "f4", "code": "SDN_Multipath", "layer": "SDN Transport",
     "desc": "OpenFlow 1.3 adaptive flow-aware multipath (1) vs. Static shortest-path tree (0)"},
    {"id": "f5", "code": "TSN_PriorityQueue", "layer": "SDN Transport",
     "desc": "IEEE 802.1Qbv/OpenFlow strict meter & priority queue (1) vs. Best-effort FIFO (0)"},
    {"id": "f6", "code": "Hybrid_Failover", "layer": "SDN Transport",
     "desc": "Multi-factor controller + fast-failover group table (1) vs. Link-down only (0)"},
    {"id": "f7", "code": "Green_LinkSleep", "layer": "SDN Transport",
     "desc": "Elastic sleep of redundant inter-island links (1) vs. Always-on full mesh (0)"},

    # Layer III: Edge-Cloud Compute, ML Anomaly Detection & Quality Control
    {"id": "f8", "code": "Edge_PreFilter", "layer": "Edge-Cloud ML",
     "desc": "Local PLC/Edge Kalman & wavelet denoising (1) vs. Raw cloud forwarding (0)"},
    {"id": "f9", "code": "Quantized_IDS", "layer": "Edge-Cloud ML",
     "desc": "INT8 quantized tree/LSTM anomaly detector on edge (1) vs. Heavy FP32 cloud model (0)"},
    {"id": "f10", "code": "Async_Historian", "layer": "Edge-Cloud ML",
     "desc": "Write-behind sharded time-series historian (1) vs. Synchronous transactional DB (0)"},
    {"id": "f11", "code": "ClosedLoop_DQT", "layer": "Edge-Cloud ML",
     "desc": "Active Digital Quality Twin feedback gating (1) vs. Passive alarm logging (0)"},
]


def build_cpps_landscape(topology_name: str, seed: int, k_epistasis: int, weights=(0.40, 0.35, 0.25)):
    """
    Constructs a physically grounded 12-feature CPPS configuration landscape (2^12 = 4096)
    combining queueing/latency physics, reliability/anomaly-detection recall, and energy/resource
    efficiency with pairwise cross-layer epistatic traps (where single-bit flips fail without
    coordinated 2-bit co-design flips) and NK-like interactions.
    """
    rng = np.random.default_rng(seed)
    n_feat = 12
    n_cfg = 1 << n_feat
    bits = np.array([[(i >> b) & 1 for b in range(n_feat)] for i in range(n_cfg)], dtype=np.float64)

    # Physical baseline linear effects on:
    # 1) Real-time determinism / Latency score U_lat in [0, 1]
    # 2) Operational reliability & anomaly detection score U_rel in [0, 1]
    # 3) Energy & resource efficiency score U_eff in [0, 1]
    w_lat = np.array([0.11, 0.12, 0.07, 0.06, 0.13, 0.15, 0.05, -0.05, 0.10, 0.10, 0.08, 0.05])
    w_rel = np.array([0.05, 0.05, -0.02, 0.10, 0.11, 0.09, 0.18, -0.06, 0.10, 0.08, 0.11, 0.16])
    w_eff = np.array([0.06, 0.10, 0.14, 0.03, -0.03, -0.02, -0.04, 0.17, 0.08, 0.15, 0.07, -0.03])

    # Pairwise co-design traps (enabling one feature without its partner incurs a mismatch penalty,
    # while enabling both yields a strong synergy — creating realistic H1 local optima traps that H2 escapes):
    def pair_trap(b_a, b_b, syn_gain, mismatch_cost):
        both = bits[:, b_a] * bits[:, b_b]
        xor = bits[:, b_a] * (1.0 - bits[:, b_b]) + (1.0 - bits[:, b_a]) * bits[:, b_b]
        return syn_gain * both - mismatch_cost * xor

    # Trap 1: OPCUA_PubSub (f0) <-> TSN_PriorityQueue (f5) (UDP bursts need TSN shaping)
    trap_0_5 = pair_trap(0, 5, 0.22, 0.28)
    # Trap 2: Protobuf_Framing (f1) <-> Async_Historian (f10) (high-rate binary streams need async DB)
    trap_1_10 = pair_trap(1, 10, 0.20, 0.26)
    # Trap 3: Adaptive_Sampling (f2) <-> Green_LinkSleep (f7) (link sleeping requires delta traffic reduction)
    trap_2_7 = pair_trap(2, 7, 0.22, 0.28)
    # Trap 4: Edge_PreFilter (f8) <-> Quantized_IDS (f9) (INT8 model needs pre-filtered input to avoid FP bursts)
    trap_8_9 = pair_trap(8, 9, 0.21, 0.26)
    # Trap 5: SDN_Multipath (f4) <-> Hybrid_Failover (f6) (multipath group tables require coordinated failover)
    trap_4_6 = pair_trap(4, 6, 0.21, 0.26)
    # Trap 6: BLE_UWB_Sync (f3) <-> ClosedLoop_DQT (f11) (active quality gating requires synchronized localization)
    trap_3_11 = pair_trap(3, 11, 0.20, 0.25)

    co_design_lat = trap_0_5 + trap_1_10 + 0.5 * trap_4_6 + 0.5 * trap_8_9
    co_design_rel = trap_4_6 + trap_8_9 + trap_3_11 + 0.5 * trap_0_5
    co_design_eff = trap_2_7 + 0.6 * trap_8_9 + 0.5 * trap_1_10

    def build_component(base_w, co_term, ep_scale):
        lin = bits @ base_w + co_term
        ep = np.zeros(n_cfg, dtype=np.float64)
        for j in range(n_feat):
            mod_base = (j // 4) * 4
            nbrs = [j]
            for off in range(1, k_epistasis):
                if off == 1:
                    nbrs.append(mod_base + ((j - mod_base + 1) % 4))
                elif off == 2:
                    nbrs.append((j + 4) % n_feat)
                else:
                    nbrs.append((j + 7) % n_feat)
            nbrs = sorted(list(set(nbrs)))
            tbl = rng.normal(0.0, 1.0, size=1 << len(nbrs))
            idx = np.zeros(n_cfg, dtype=np.int32)
            for pos, nf in enumerate(nbrs):
                idx |= (bits[:, nf].astype(np.int32) << pos)
            ep += tbl[idx]
        ep = ep / np.sqrt(n_feat)
        raw = lin + ep_scale * ep
        norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        return 0.08 + 0.90 * norm + rng.uniform(0.0, 1e-7, size=n_cfg)

    ep_map = {"L1_Assembly": 0.07, "L2_Planarization": 0.15, "L3_IslandFactory": 0.11}
    ep_scale = ep_map.get(topology_name, 0.11)

    u_lat = build_component(w_lat, co_design_lat, ep_scale)
    u_rel = build_component(w_rel, co_design_rel, ep_scale)
    u_eff = build_component(w_eff, co_design_eff, ep_scale)

    w_d, w_f, w_v = weights
    u_raw = w_d * u_lat + w_f * u_rel + w_v * u_eff
    u_total = 0.06 + 0.94 * (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-12)
    u_total += np.arange(n_cfg) * 1e-10
    u_total = u_total / u_total.max()

    nbrs_h1 = []
    nbrs_h2 = []
    for i in range(n_cfg):
        h1 = [i ^ (1 << b) for b in range(n_feat)]
        h2 = list(h1)
        for b1 in range(n_feat):
            for b2 in range(b1 + 1, n_feat):
                h2.append(i ^ (1 << b1) ^ (1 << b2))
        nbrs_h1.append(np.array(h1, dtype=np.int32))
        nbrs_h2.append(np.array(h2, dtype=np.int32))

    opt_h1 = [i for i in range(n_cfg) if np.all(u_total[i] > u_total[nbrs_h1[i]])]
    opt_h2 = [i for i in range(n_cfg) if np.all(u_total[i] > u_total[nbrs_h2[i]])]

    delay_ms = 48.0 - 43.8 * u_lat
    reliability_pct = 72.0 + 27.6 * u_rel
    power_watts = 340.0 - 175.0 * u_eff

    return {
        "name": topology_name,
        "n_feat": n_feat,
        "n_cfg": n_cfg,
        "bits": bits,
        "u": u_total,
        "u_lat": u_lat,
        "u_rel": u_rel,
        "u_eff": u_eff,
        "delay_ms": delay_ms,
        "reliability_pct": reliability_pct,
        "power_watts": power_watts,
        "nbrs_h1": nbrs_h1,
        "nbrs_h2": nbrs_h2,
        "opt_h1": set(opt_h1),
        "opt_h2": set(opt_h2),
        "global_opt": int(np.argmax(u_total)),
        "global_opt_u": float(np.max(u_total)),
        "mean_u": float(np.mean(u_total)),
        "std_u": float(np.std(u_total)),
    }


def make_digital_twin_proxy(landscape, rho_dt: float, rng: np.random.Generator):
    u = landscape["u"]
    u_std = (u - u.mean()) / (u.std() + 1e-12)
    z = rng.normal(0.0, 1.0, size=len(u))
    z = (z - z.mean()) / (z.std() + 1e-12)
    proxy = rho_dt * u_std + np.sqrt(max(0.0, 1.0 - rho_dt ** 2)) * z
    return u.mean() + u.std() * proxy


def run_single_trajectory(
    landscape,
    budget_T: int = 120,
    policy: str = "retest_2_of_5",
    sigma: float = 0.075,
    m_pre: int = 4,
    rho_dt: float = 0.65,
    sys_bias: float = 0.0,
    conf_bias: float = 0.0,
    drift_D: float = 0.0,
    neighborhood: str = "H2",
    retention: bool = True,
    decay_power: float = 0.0,
    seed: int = 1,
    allowed_mask: np.ndarray = None,
):
    rng = np.random.default_rng(seed)
    u_true = landscape["u"]
    n_cfg = landscape["n_cfg"]
    nbr_map = landscape["nbrs_h1"] if neighborhood == "H1" else landscape["nbrs_h2"]

    dt_scores = make_digital_twin_proxy(landscape, rho_dt, rng)

    if allowed_mask is None:
        valid_indices = np.arange(n_cfg)
    else:
        valid_indices = np.where(allowed_mask)[0]

    sorted_valid = valid_indices[np.argsort(u_true[valid_indices])]
    p0 = int(rng.choice(sorted_valid[: max(10, int(0.35 * len(sorted_valid)))]))

    retained = [p0]
    retained_set = {p0}
    counts = {p0: 0}
    sums = {p0: 0.0}
    beliefs = {p0: 0.5}

    trace_shipped = np.zeros(budget_T, dtype=np.float64)
    trace_believed = np.zeros(budget_T, dtype=np.float64)
    trace_found = np.zeros(budget_T, dtype=np.float64)

    fav = p0

    def observe(p, t_idx, current_fav):
        drift_val = drift_D * np.sin(2.0 * np.pi * t_idx / 30.0 + (p % 17) * 0.37) if drift_D > 0 else 0.0
        b_sys = sys_bias * (1.0 if (p % 2 == 1 and u_true[p] < 0.85) else -0.5) if sys_bias > 0 else 0.0
        b_conf = conf_bias if (conf_bias > 0 and p == current_fav) else 0.0
        noise = rng.normal(0.0, sigma)
        return u_true[p] + b_sys + b_conf + drift_val + noise

    def propose_new_variant(current_fav, t_idx, custom_m=None):
        if decay_power > 0.0:
            prob_ideate = min(1.0, 1.5 / ((t_idx + 1) ** decay_power))
            if rng.uniform() > prob_ideate:
                return current_fav

        if neighborhood == "GLOBAL":
            cands = [int(c) for c in valid_indices if int(c) not in retained_set]
        else:
            raw_nbrs = nbr_map[current_fav]
            if allowed_mask is not None:
                raw_nbrs = [int(c) for c in raw_nbrs if allowed_mask[c]]
            cands = [int(c) for c in raw_nbrs if int(c) not in retained_set]
            if not cands and neighborhood != "H1":
                for rp in reversed(retained):
                    rn = nbr_map[rp]
                    if allowed_mask is not None:
                        rn = [int(c) for c in rn if allowed_mask[c]]
                    cands = [int(c) for c in rn if int(c) not in retained_set]
                    if cands:
                        break
            if not cands:
                return current_fav

        eff_m = m_pre if custom_m is None else custom_m
        k_pick = min(len(cands), max(1, eff_m))
        sampled = rng.choice(cands, size=k_pick, replace=False)
        best_cand = int(sampled[np.argmax(dt_scores[sampled])])
        return best_cand

    for t in range(budget_T):
        if t == 0:
            target = p0
        else:
            if policy == "single_shot":
                target = propose_new_variant(fav, t)
            elif policy == "hill_climb_once":
                # Unscreened local single-bit / two-bit random walk measured once
                target = propose_new_variant(fav, t, custom_m=1)
            elif policy == "fixed_1_of_2":
                target = propose_new_variant(fav, t) if (t % 2 == 0) else fav
            elif policy == "retest_2_of_5":
                slot = t % 5
                if slot in (0, 2):
                    target = propose_new_variant(fav, t)
                elif slot == 1 or len(retained) < 2:
                    target = fav
                else:
                    # Re-test among top-3 contenders with highest uncertainty-adjusted belief
                    ranked = sorted(retained, key=lambda x: beliefs[x], reverse=True)[:3]
                    target = min(ranked, key=lambda x: counts[x])
            elif policy == "fixed_1_of_3":
                slot = t % 3
                if slot == 0:
                    target = propose_new_variant(fav, t)
                elif slot == 1 or len(retained) < 2:
                    target = fav
                else:
                    ranked = sorted(retained, key=lambda x: beliefs[x], reverse=True)
                    target = ranked[1]
            elif policy == "fixed_1_of_5":
                slot = t % 5
                if slot == 0:
                    target = propose_new_variant(fav, t)
                elif slot in (1, 2) or len(retained) < 2:
                    target = fav
                else:
                    ranked = sorted(retained, key=lambda x: beliefs[x], reverse=True)
                    target = ranked[min(len(ranked) - 1, 1 if slot == 3 else 2)]
            elif policy == "ucb1":
                if t % 5 in (0, 2):
                    target = propose_new_variant(fav, t)
                else:
                    total_n = sum(counts.values())
                    target = max(
                        retained,
                        key=lambda x: beliefs[x] + 0.14 * np.sqrt(np.log(total_n + 1.0) / (counts[x] + 1e-6)),
                    )
            elif policy == "thompson":
                if t % 5 in (0, 2):
                    target = propose_new_variant(fav, t)
                else:
                    target = max(
                        retained,
                        key=lambda x: rng.normal(beliefs[x], sigma / np.sqrt(counts[x] + 1.0)),
                    )
            elif policy == "twin_only":
                if t == 1:
                    best_dt = int(valid_indices[np.argmax(dt_scores[valid_indices])])
                    target = best_dt
                else:
                    target = fav
            else:
                target = propose_new_variant(fav, t)

        y_obs = observe(target, t, fav)
        if target not in retained_set:
            retained.append(target)
            retained_set.add(target)
            counts[target] = 0
            sums[target] = 0.0

        counts[target] += 1
        sums[target] += y_obs
        beliefs[target] = sums[target] / counts[target]

        if not retention and len(retained) > 4:
            drop_p = retained.pop(0)
            retained_set.remove(drop_p)
            del counts[drop_p]
            del sums[drop_p]
            del beliefs[drop_p]

        fav = max(retained, key=lambda x: beliefs[x])
        trace_shipped[t] = u_true[fav]
        trace_believed[t] = beliefs[fav]
        trace_found[t] = max(u_true[p] for p in retained)

    return {
        "shipped_final": float(trace_shipped[-1]),
        "believed_final": float(trace_believed[-1]),
        "found_final": float(trace_found[-1]),
        "curse_final": float(trace_believed[-1] - trace_shipped[-1]),
        "selection_loss": float(trace_found[-1] - trace_shipped[-1]),
        "is_local_opt_h1": bool(fav in landscape["opt_h1"]),
        "is_local_opt_h2": bool(fav in landscape["opt_h2"]),
        "n_prototypes_built": int(len(retained)),
        "fav_idx": int(fav),
        "delay_ms": float(landscape["delay_ms"][fav]),
        "reliability_pct": float(landscape["reliability_pct"][fav]),
        "power_watts": float(landscape["power_watts"][fav]),
        "trace_shipped": trace_shipped,
        "trace_believed": trace_believed,
        "trace_found": trace_found,
    }


def summarize_runs(runs):
    shipped = np.array([r["shipped_final"] for r in runs])
    believed = np.array([r["believed_final"] for r in runs])
    found = np.array([r["found_final"] for r in runs])
    curse = np.array([r["curse_final"] for r in runs])
    sel_loss = np.array([r["selection_loss"] for r in runs])
    opt_h1 = np.array([r["is_local_opt_h1"] for r in runs], dtype=np.float64)
    opt_h2 = np.array([r["is_local_opt_h2"] for r in runs], dtype=np.float64)
    n_built = np.array([r["n_prototypes_built"] for r in runs], dtype=np.float64)
    delay = np.array([r["delay_ms"] for r in runs])
    rel = np.array([r["reliability_pct"] for r in runs])
    power = np.array([r["power_watts"] for r in runs])

    q25, q75 = np.percentile(shipped, [25, 75])
    return {
        "shipped_mean": float(np.mean(shipped)),
        "shipped_sd": float(np.std(shipped)),
        "shipped_median": float(np.median(shipped)),
        "shipped_iqr": float(q75 - q25),
        "shipped_q25": float(q25),
        "shipped_q75": float(q75),
        "believed_mean": float(np.mean(believed)),
        "believed_sd": float(np.std(believed)),
        "found_mean": float(np.mean(found)),
        "found_sd": float(np.std(found)),
        "curse_mean": float(np.mean(curse)),
        "curse_sd": float(np.std(curse)),
        "sel_loss_mean": float(np.mean(sel_loss)),
        "opt_h1_rate": float(np.mean(opt_h1) * 100.0),
        "opt_h2_rate": float(np.mean(opt_h2) * 100.0),
        "n_built_mean": float(np.mean(n_built)),
        "delay_mean": float(np.mean(delay)),
        "rel_mean": float(np.mean(rel)),
        "power_mean": float(np.mean(power)),
        "trace_shipped": np.mean([r["trace_shipped"] for r in runs], axis=0).tolist(),
        "trace_believed": np.mean([r["trace_believed"] for r in runs], axis=0).tolist(),
        "trace_found": np.mean([r["trace_found"] for r in runs], axis=0).tolist(),
        "_raw_shipped": shipped,
    }


def run_all_experiments():
    t0 = time.time()
    landscapes = {
        "L1_Assembly": build_cpps_landscape("L1_Assembly", seed=101, k_epistasis=2),
        "L2_Planarization": build_cpps_landscape("L2_Planarization", seed=202, k_epistasis=4),
        "L3_IslandFactory": build_cpps_landscape("L3_IslandFactory", seed=303, k_epistasis=3),
    }

    landscape_summary = {}
    for k, L in landscapes.items():
        landscape_summary[k] = {
            "n_cfg": L["n_cfg"],
            "n_opt_h1": len(L["opt_h1"]),
            "n_opt_h2": len(L["opt_h2"]),
            "global_opt_u": L["global_opt_u"],
            "mean_u": L["mean_u"],
            "std_u": L["std_u"],
            "opt_h1_mean_u": float(np.mean(L["u"][list(L["opt_h1"])])),
            "opt_h2_mean_u": float(np.mean(L["u"][list(L["opt_h2"])])),
            "opt_h2_min_u": float(np.min(L["u"][list(L["opt_h2"])])),
            "global_opt_bits": "".join(str(int(b)) for b in L["bits"][L["global_opt"]]),
            "global_opt_delay_ms": float(L["delay_ms"][L["global_opt"]]),
            "global_opt_rel_pct": float(L["reliability_pct"][L["global_opt"]]),
            "global_opt_power_w": float(L["power_watts"][L["global_opt"]]),
        }

    # --------------------------------------------------------------------------
    # EXPERIMENT 1: Budget Allocation & Optimizer's Curse across 3 CPPS Topologies
    # --------------------------------------------------------------------------
    policies_e1 = [
        "single_shot",
        "hill_climb_once",
        "fixed_1_of_2",
        "retest_2_of_5",
        "fixed_1_of_3",
        "fixed_1_of_5",
        "ucb1",
        "thompson",
    ]
    n_seeds_e1 = 250
    exp1 = {"by_landscape": {}, "pooled": {}}
    raw_by_pol = {p: [] for p in policies_e1}

    for lname, L in landscapes.items():
        exp1["by_landscape"][lname] = {}
        for pol in policies_e1:
            runs = [
                run_single_trajectory(L, budget_T=120, policy=pol, sigma=0.075, m_pre=4, rho_dt=0.65, seed=1000 + s)
                for s in range(n_seeds_e1)
            ]
            summ = summarize_runs(runs)
            raw_by_pol[pol].extend(runs)
            del summ["_raw_shipped"]
            exp1["by_landscape"][lname][pol] = summ

    ref_raw = np.array([r["shipped_final"] for r in raw_by_pol["retest_2_of_5"]])
    friedman_arrays = []
    for pol in policies_e1:
        summ = summarize_runs(raw_by_pol[pol])
        arr = summ.pop("_raw_shipped")
        friedman_arrays.append(arr)
        if pol == "retest_2_of_5":
            summ["wilcoxon_p_vs_2of5"] = 1.0
            summ["wilcoxon_stat"] = 0.0
        else:
            stat_w, p_w = stats.wilcoxon(ref_raw, arr)
            summ["wilcoxon_p_vs_2of5"] = float(p_w)
            summ["wilcoxon_stat"] = float(stat_w)
        exp1["pooled"][pol] = summ

    f_stat, f_pval = stats.friedmanchisquare(*friedman_arrays)
    mat = np.vstack(friedman_arrays)
    ranks = np.argsort(np.argsort(-mat, axis=0), axis=0) + 1.0
    for idx, pol in enumerate(policies_e1):
        exp1["pooled"][pol]["avg_rank"] = float(np.mean(ranks[idx]))
    exp1["friedman_chi2"] = float(f_stat)
    exp1["friedman_p"] = float(f_pval)

    # --------------------------------------------------------------------------
    # EXPERIMENT 2: Digital Twin Pre-Screening Capacity (M_pre) & Surrogate Fidelity (rho_DT)
    # --------------------------------------------------------------------------
    L_main = landscapes["L3_IslandFactory"]
    m_grid = [1, 2, 4, 8, 12]
    rho_grid = [0.0, 0.40, 0.65, 0.85, 0.95]
    n_seeds_e2 = 200
    exp2 = {"m_sweep": {}, "rho_sweep": {}}

    for pol in ["single_shot", "retest_2_of_5", "ucb1"]:
        exp2["m_sweep"][pol] = {}
        for m_val in m_grid:
            runs = [
                run_single_trajectory(L_main, budget_T=120, policy=pol, sigma=0.075, m_pre=m_val, rho_dt=0.65, seed=3000 + s)
                for s in range(n_seeds_e2)
            ]
            s_dict = summarize_runs(runs)
            del s_dict["_raw_shipped"]
            exp2["m_sweep"][pol][str(m_val)] = s_dict

    for pol in ["retest_2_of_5", "twin_only", "single_shot"]:
        exp2["rho_sweep"][pol] = {}
        for r_val in rho_grid:
            runs = [
                run_single_trajectory(L_main, budget_T=120, policy=pol, sigma=0.075, m_pre=4, rho_dt=r_val, seed=4000 + s)
                for s in range(n_seeds_e2)
            ]
            s_dict = summarize_runs(runs)
            del s_dict["_raw_shipped"]
            exp2["rho_sweep"][pol][f"{r_val:.2f}"] = s_dict

    # --------------------------------------------------------------------------
    # EXPERIMENT 3: Assumption Ablation Study (Violating Conditions A1-A5 of Theorem 1)
    # --------------------------------------------------------------------------
    n_seeds_e3 = 250
    ablation_configs = [
        ("full_theorem1", {"policy": "retest_2_of_5", "retention": True, "decay_power": 0.0, "neighborhood": "H2"}),
        ("no_retention_A5", {"policy": "retest_2_of_5", "retention": False, "decay_power": 0.0, "neighborhood": "H2"}),
        ("no_retest_A4", {"policy": "single_shot", "retention": True, "decay_power": 0.0, "neighborhood": "H2"}),
        ("summable_gen_A3", {"policy": "retest_2_of_5", "retention": True, "decay_power": 2.0, "neighborhood": "H2"}),
        ("narrow_H1_A2", {"policy": "retest_2_of_5", "retention": True, "decay_power": 0.0, "neighborhood": "H1"}),
        ("unbounded_global_A2", {"policy": "retest_2_of_5", "retention": True, "decay_power": 0.0, "neighborhood": "GLOBAL"}),
    ]
    exp3 = {}
    ref_e3 = None
    for ab_name, kw in ablation_configs:
        runs = [
            run_single_trajectory(
                L_main, budget_T=120, sigma=0.075, m_pre=4, rho_dt=0.65, seed=5000 + s, **kw
            )
            for s in range(n_seeds_e3)
        ]
        s_dict = summarize_runs(runs)
        arr = s_dict.pop("_raw_shipped")
        if ab_name == "full_theorem1":
            ref_e3 = arr
            s_dict["wilcoxon_p"] = 1.0
        else:
            _, p_val = stats.wilcoxon(ref_e3, arr)
            s_dict["wilcoxon_p"] = float(p_val)
        exp3[ab_name] = s_dict

    # --------------------------------------------------------------------------
    # EXPERIMENT 4: Bounded Telemetry Bias B, Confirmation Bias, and Non-Stationary Drift D
    # --------------------------------------------------------------------------
    n_seeds_e4 = 200
    bias_grid = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    exp4 = {"sys_bias": {}, "conf_bias": {}, "drift": {}}
    base_unbiased = exp3["full_theorem1"]["shipped_mean"]

    for b_val in bias_grid:
        runs_sys = [
            run_single_trajectory(L_main, budget_T=120, policy="retest_2_of_5", sigma=0.075, sys_bias=b_val, seed=6000 + s)
            for s in range(n_seeds_e4)
        ]
        s_sys = summarize_runs(runs_sys)
        del s_sys["_raw_shipped"]
        s_sys["empirical_loss"] = float(max(0.0, base_unbiased - s_sys["shipped_mean"]))
        s_sys["theoretical_2B"] = float(2.0 * b_val)
        exp4["sys_bias"][f"{b_val:.2f}"] = s_sys

        runs_conf = [
            run_single_trajectory(L_main, budget_T=120, policy="retest_2_of_5", sigma=0.075, conf_bias=b_val, seed=7000 + s)
            for s in range(n_seeds_e4)
        ]
        s_conf = summarize_runs(runs_conf)
        del s_conf["_raw_shipped"]
        s_conf["empirical_loss"] = float(max(0.0, base_unbiased - s_conf["shipped_mean"]))
        exp4["conf_bias"][f"{b_val:.2f}"] = s_conf

        runs_dr = [
            run_single_trajectory(L_main, budget_T=120, policy="retest_2_of_5", sigma=0.075, drift_D=b_val, seed=8000 + s)
            for s in range(n_seeds_e4)
        ]
        s_dr = summarize_runs(runs_dr)
        del s_dr["_raw_shipped"]
        s_dr["empirical_loss"] = float(max(0.0, base_unbiased - s_dr["shipped_mean"]))
        s_dr["theoretical_2D"] = float(2.0 * b_val)
        exp4["drift"][f"{b_val:.2f}"] = s_dr

    # --------------------------------------------------------------------------
    # EXPERIMENT 5: Multi-Stakeholder Intent Elicitation (REDUCE PCM) vs. Pareto Archive
    # --------------------------------------------------------------------------
    n_seeds_e5 = 150
    u_vec = np.stack([L_main["u_lat"], L_main["u_rel"], L_main["u_eff"]], axis=1)

    local_pareto_h2 = set()
    for i in range(L_main["n_cfg"]):
        nbrs = L_main["nbrs_h2"][i]
        dom = np.any(np.all(u_vec[nbrs] >= u_vec[i], axis=1) & np.any(u_vec[nbrs] > u_vec[i], axis=1))
        if not dom:
            local_pareto_h2.add(i)

    def hypervolume_3d(indices):
        pts = u_vec[indices]
        grid = np.linspace(0.02, 0.98, 25)
        g1, g2, g3 = np.meshgrid(grid, grid, grid, indexing="ij")
        g_pts = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)
        covered = np.zeros(len(g_pts), dtype=bool)
        for pt in pts:
            covered |= np.all(g_pts <= pt, axis=1)
        return float(np.mean(covered))

    exp5 = {}
    stakeholder_modes = [
        ("Latency_Dominated", (0.80, 0.10, 0.10)),
        ("Reliability_Dominated", (0.10, 0.80, 0.10)),
        ("Energy_Dominated", (0.10, 0.10, 0.80)),
        ("Unrectified_PCM_HighCR", "noisy_pcm"),
        ("REDUCE_Consensus_LowCR", (0.40, 0.35, 0.25)),
    ]

    for m_name, w_spec in stakeholder_modes:
        hvs, comps, lats, rels, effs, p_rates = [], [], [], [], [], []
        for s in range(n_seeds_e5):
            rng_s = np.random.default_rng(9000 + s)
            if w_spec == "noisy_pcm":
                rw = rng_s.dirichlet([0.5, 0.5, 0.5])
            else:
                rw = np.array(w_spec, dtype=np.float64)
            L_sub = dict(L_main)
            u_sub = rw[0] * L_main["u_lat"] + rw[1] * L_main["u_rel"] + rw[2] * L_main["u_eff"]
            u_sub = (u_sub - u_sub.min()) / (u_sub.max() - u_sub.min() + 1e-12)
            L_sub["u"] = u_sub
            res_s = run_single_trajectory(L_sub, budget_T=120, policy="retest_2_of_5", sigma=0.075, seed=9000 + s)
            fav_p = res_s["fav_idx"]
            hvs.append(float(np.prod(u_vec[fav_p])))
            comps.append(float(L_main["u"][fav_p]))
            lats.append(float(L_main["u_lat"][fav_p]))
            rels.append(float(L_main["u_rel"][fav_p]))
            effs.append(float(L_main["u_eff"][fav_p]))
            p_rates.append(100.0 if fav_p in local_pareto_h2 else 0.0)
        exp5[m_name] = {
            "composite_u_mean": float(np.mean(comps)),
            "composite_u_sd": float(np.std(comps)),
            "u_lat_mean": float(np.mean(lats)),
            "u_rel_mean": float(np.mean(rels)),
            "u_eff_mean": float(np.mean(effs)),
            "hv_mean": float(np.mean(hvs)),
            "hv_sd": float(np.std(hvs)),
            "local_pareto_rate": float(np.mean(p_rates)),
            "archive_size_mean": 1.0,
        }

    hv_par, comp_par, lat_par, rel_par, eff_par, pr_par, sz_par = [], [], [], [], [], [], []
    sample_pareto_pts = None
    for s in range(n_seeds_e5):
        rng_p = np.random.default_rng(9500 + s)
        p0 = int(rng_p.integers(0, L_main["n_cfg"]))
        retained_p = [p0]
        counts_p = {p0: 0}
        sums_p = {p0: np.zeros(3)}
        bel_p = {p0: np.full(3, 0.5)}
        archive = [p0]

        for t in range(120):
            if t == 0:
                target = p0
            elif t % 5 in (0, 2):
                parent = int(rng_p.choice(archive))
                cands = [int(c) for c in L_main["nbrs_h2"][parent] if int(c) not in counts_p]
                if not cands:
                    cands = [int(c) for c in range(L_main["n_cfg"]) if int(c) not in counts_p]
                k_pick = min(len(cands), 4)
                sub_c = rng_p.choice(cands, size=k_pick, replace=False)
                w_rand = rng_p.dirichlet([1.5, 1.5, 1.5])
                scores = u_vec[sub_c] @ w_rand + rng_p.normal(0, 0.06, size=k_pick)
                target = int(sub_c[np.argmax(scores)])
            else:
                target = min(archive, key=lambda x: counts_p[x])

            if target not in counts_p:
                retained_p.append(target)
                counts_p[target] = 0
                sums_p[target] = np.zeros(3)
            obs_vec = u_vec[target] + rng_p.normal(0.0, 0.075, size=3)
            counts_p[target] += 1
            sums_p[target] += obs_vec
            bel_p[target] = sums_p[target] / counts_p[target]

            bel_mat = np.array([bel_p[x] for x in retained_p])
            new_arch = []
            for idx_i, x_i in enumerate(retained_p):
                dom = np.any(
                    np.all(bel_mat >= bel_mat[idx_i], axis=1) & np.any(bel_mat > bel_mat[idx_i], axis=1)
                )
                if not dom:
                    new_arch.append(x_i)
            archive = new_arch if new_arch else [retained_p[0]]

        hv_par.append(hypervolume_3d(archive))
        best_in_arch = max(archive, key=lambda x: L_main["u"][x])
        comp_par.append(float(L_main["u"][best_in_arch]))
        lat_par.append(float(np.max(L_main["u_lat"][archive])))
        rel_par.append(float(np.max(L_main["u_rel"][archive])))
        eff_par.append(float(np.max(L_main["u_eff"][archive])))
        pr_par.append(float(np.mean([100.0 if x in local_pareto_h2 else 0.0 for x in archive])))
        sz_par.append(float(len(archive)))
        if s == 0:
            sample_pareto_pts = u_vec[archive].tolist()

    exp5["Pareto_Archive_Prop2"] = {
        "composite_u_mean": float(np.mean(comp_par)),
        "composite_u_sd": float(np.std(comp_par)),
        "u_lat_mean": float(np.mean(lat_par)),
        "u_rel_mean": float(np.mean(rel_par)),
        "u_eff_mean": float(np.mean(eff_par)),
        "hv_mean": float(np.mean(hv_par)),
        "hv_sd": float(np.std(hv_par)),
        "local_pareto_rate": float(np.mean(pr_par)),
        "archive_size_mean": float(np.mean(sz_par)),
        "sample_archive_pts": sample_pareto_pts,
        "all_cloud_pts": u_vec[::8].tolist(),
    }

    # --------------------------------------------------------------------------
    # EXPERIMENT 6: Epoch Reframing under Legacy Frame Lock & Anomaly Injection (t = 70)
    # --------------------------------------------------------------------------
    bits = L_main["bits"]
    legacy_mask = (bits[:, 0] == 0) & (bits[:, 5] == 0) & (bits[:, 6] == 0) & (bits[:, 11] == 0)
    legacy_max_u = float(np.max(L_main["u"][legacy_mask]))

    n_seeds_e6 = 200
    horizon_e6 = 140
    reframe_step = 70

    runs_legacy = [
        run_single_trajectory(
            L_main, budget_T=horizon_e6, policy="retest_2_of_5", sigma=0.075,
            allowed_mask=legacy_mask, seed=11000 + s
        )
        for s in range(n_seeds_e6)
    ]
    summ_legacy = summarize_runs(runs_legacy)
    raw_leg = summ_legacy.pop("_raw_shipped")

    traces_reframe_shipped = []
    traces_reframe_believed = []
    finals_reframe = []
    delays_reframe = []
    rels_reframe = []

    for s in range(n_seeds_e6):
        r_ep1 = run_single_trajectory(
            L_main, budget_T=reframe_step, policy="retest_2_of_5", sigma=0.075,
            allowed_mask=legacy_mask, seed=11000 + s
        )
        fav_ep1 = r_ep1["fav_idx"]

        rng_rf = np.random.default_rng(12000 + s)
        dt_scores = make_digital_twin_proxy(L_main, 0.65, rng_rf)
        u_true = L_main["u"]

        retained = [fav_ep1]
        retained_set = {fav_ep1}
        counts = {fav_ep1: 3}
        sums = {fav_ep1: 3.0 * r_ep1["believed_final"]}
        beliefs = {fav_ep1: r_ep1["believed_final"]}
        fav = fav_ep1

        tr_s = list(r_ep1["trace_shipped"])
        tr_b = list(r_ep1["trace_believed"])

        for t in range(reframe_step, horizon_e6):
            slot = (t - reframe_step) % 5
            if slot in (0, 2):
                cands = [int(c) for c in L_main["nbrs_h2"][fav] if int(c) not in retained_set]
                if not cands:
                    cands = [int(c) for c in range(L_main["n_cfg"]) if int(c) not in retained_set]
                k_pick = min(len(cands), 4)
                sub_c = rng_rf.choice(cands, size=k_pick, replace=False)
                target = int(sub_c[np.argmax(dt_scores[sub_c])])
            elif slot == 1 or len(retained) < 2:
                target = fav
            else:
                ranked = sorted(retained, key=lambda x: beliefs[x], reverse=True)[:3]
                target = min(ranked, key=lambda x: counts[x])

            y_obs = u_true[target] + rng_rf.normal(0.0, 0.075)
            if target not in retained_set:
                retained.append(target)
                retained_set.add(target)
                counts[target] = 0
                sums[target] = 0.0
            counts[target] += 1
            sums[target] += y_obs
            beliefs[target] = sums[target] / counts[target]
            fav = max(retained, key=lambda x: beliefs[x])
            tr_s.append(float(u_true[fav]))
            tr_b.append(float(beliefs[fav]))

        traces_reframe_shipped.append(tr_s)
        traces_reframe_believed.append(tr_b)
        finals_reframe.append(tr_s[-1])
        delays_reframe.append(float(L_main["delay_ms"][fav]))
        rels_reframe.append(float(L_main["reliability_pct"][fav]))

    finals_reframe = np.array(finals_reframe)
    _, p_reframe = stats.wilcoxon(finals_reframe, raw_leg)

    exp6 = {
        "legacy_ceiling_u": legacy_max_u,
        "global_ceiling_u": L_main["global_opt_u"],
        "reframe_step": reframe_step,
        "fixed_legacy": summ_legacy,
        "reframed_epoch": {
            "shipped_mean": float(np.mean(finals_reframe)),
            "shipped_sd": float(np.std(finals_reframe)),
            "shipped_median": float(np.median(finals_reframe)),
            "shipped_iqr": float(np.percentile(finals_reframe, 75) - np.percentile(finals_reframe, 25)),
            "delay_mean": float(np.mean(delays_reframe)),
            "rel_mean": float(np.mean(rels_reframe)),
            "trace_shipped": np.mean(traces_reframe_shipped, axis=0).tolist(),
            "trace_believed": np.mean(traces_reframe_believed, axis=0).tolist(),
            "wilcoxon_p_vs_legacy": float(p_reframe),
        },
    }

    out = {
        "meta": {
            "preprint": "arXiv:2611.03034",
            "url": "http://arxiv.org/abs/2611.03034",
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "features": FEATURES,
        "landscapes": landscape_summary,
        "exp1": exp1,
        "exp2": exp2,
        "exp3": exp3,
        "exp4": exp4,
        "exp5": exp5,
        "exp6": exp6,
    }

    with open("results_astrj.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Simulation completed in {out['meta']['elapsed_sec']}s.")


if __name__ == "__main__":
    run_all_experiments()
