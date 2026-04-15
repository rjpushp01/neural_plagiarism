"""
Generate comprehensive metrics plots for ECC watermarking evaluation.
Covers: VAE z0 (ecc_evaluation), Shallow DDIM BCH (shallow_diffusion_ecc_eval),
        Shallow DDIM LDPC (shallow_diffusion_ldpc_eval), DWT-DCT (ecc_invisible_eval).
Saves all plots to evaluation_outputs/plots/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

REPO = "/home/pushp-raj/Documents/IE663_project_code/Neural-Plagiarism"
OUT  = os.path.join(REPO, "evaluation_outputs", "plots")
os.makedirs(OUT, exist_ok=True)

# ─── Color Palette ────────────────────────────────────────────────────────────
CLR = {
    "VAE z₀ (BCH)":    "#f5a623",
    "DDIM z₁₅ (BCH)":  "#4caf82",
    "DDIM z₁₅ (LDPC)": "#5c9be0",
}
METHODS = list(CLR.keys())
COLORS  = list(CLR.values())

STYLE = dict(figure_facecolor="#0d1117", axes_facecolor="#161b22",
             text_color="#e6edf3", grid_color="#30363d")

def apply_dark(fig, axes):
    fig.patch.set_facecolor(STYLE["figure_facecolor"])
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(STYLE["axes_facecolor"])
        ax.tick_params(colors=STYLE["text_color"], labelsize=9)
        ax.xaxis.label.set_color(STYLE["text_color"])
        ax.yaxis.label.set_color(STYLE["text_color"])
        ax.title.set_color(STYLE["text_color"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid_color"])
        ax.grid(color=STYLE["grid_color"], linestyle="--", alpha=0.5)

# ─── Load data ────────────────────────────────────────────────────────────────

def load_vae():
    path = os.path.join(REPO, "evaluation_outputs", "ecc_evaluation", "metrics.json")
    with open(path) as f:
        data = json.load(f)
    psnr_wm   = [d["watermark_quality"]["psnr"] for d in data]
    atk_psnr  = [d["attack_quality"]["psnr_vs_original"] for d in data]
    atk_ssim  = [d["attack_quality"]["ssim"] for d in data]
    atk_lpips = [d["attack_quality"]["lpips"] for d in data]
    ber_raw   = [d["post_attack_ecc"]["ber_raw"] for d in data]
    ber_final = [d["post_attack_ecc"]["ber_final"] for d in data]
    recovered = [d["post_attack_ecc"]["recovered"] for d in data]
    names     = [d["image_name"].replace("coco_0000000","").replace(".jpg","") for d in data]
    return dict(psnr_wm=psnr_wm, atk_psnr=atk_psnr, atk_ssim=atk_ssim,
                atk_lpips=atk_lpips, ber_raw=ber_raw, ber_final=ber_final,
                recovered=recovered, names=names)

def load_shallow_ecc():
    path = os.path.join(REPO, "evaluation_outputs", "shallow_diffusion_ecc_eval", "metrics.json")
    with open(path) as f:
        data = json.load(f)
    psnr_wm, atk_psnr, atk_ssim, atk_lpips = [], [], [], []
    ber_raw, ber_final, recovered, names = [], [], [], []
    for d in data:
        psnr_wm.append(d["psnr_wm"])
        aq = d.get("attack_quality", {})
        atk_psnr.append(aq.get("psnr_vs_original", aq.get("psnr", np.nan)))
        atk_ssim.append(aq.get("ssim", np.nan))
        atk_lpips.append(aq.get("lpips", np.nan))
        pa = d.get("post_attack", {})
        ber_raw.append(pa.get("ber_raw", np.nan))
        ber_final.append(pa.get("ber_final", np.nan))
        recovered.append(pa.get("message_recovered", False))
        names.append(d["name"].replace("coco_0000000","").replace(".jpg",""))
    return dict(psnr_wm=psnr_wm, atk_psnr=atk_psnr, atk_ssim=atk_ssim,
                atk_lpips=atk_lpips, ber_raw=ber_raw, ber_final=ber_final,
                recovered=recovered, names=names)

def load_shallow_ldpc():
    path = os.path.join(REPO, "evaluation_outputs", "shallow_diffusion_ldpc_eval", "metrics.json")
    with open(path) as f:
        data = json.load(f)
    psnr_wm, atk_psnr, atk_ssim, atk_lpips = [], [], [], []
    ber_raw, ber_final, recovered, names = [], [], [], []
    for d in data:
        psnr_wm.append(d["psnr_wm"])
        aq = d.get("attack_quality", {})
        atk_psnr.append(aq.get("psnr_vs_original", aq.get("psnr", np.nan)))
        atk_ssim.append(aq.get("ssim", np.nan))
        atk_lpips.append(aq.get("lpips", np.nan))
        pa = d.get("post_attack", {})
        ber_raw.append(pa.get("ber_raw", np.nan))
        ber_final.append(pa.get("ber_final", np.nan))
        recovered.append(pa.get("message_recovered", False))
        names.append(d["name"].replace("coco_0000000","").replace(".jpg",""))
    return dict(psnr_wm=psnr_wm, atk_psnr=atk_psnr, atk_ssim=atk_ssim,
                atk_lpips=atk_lpips, ber_raw=ber_raw, ber_final=ber_final,
                recovered=recovered, names=names)

vae   = load_vae()
s_bch = load_shallow_ecc()
s_ldp = load_shallow_ldpc()

ALL = {m: d for m, d in zip(METHODS, [vae, s_bch, s_ldp])}

images = [f"img{i}" for i in range(10)]

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 – Watermark Quality  (PSNR watermarked vs original, per image)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12,5))
x = np.arange(10)
w = 0.25
for i, (m, d) in enumerate(ALL.items()):
    ax.bar(x + i*w, d["psnr_wm"], w, label=m, color=COLORS[i], alpha=0.88)
ax.set_xlabel("Image Index")
ax.set_ylabel("PSNR (dB)")
ax.set_title("Watermark Quality — PSNR (Watermarked vs Original)")
ax.set_xticks(x + w)
ax.set_xticklabels([str(i) for i in range(10)])
ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
apply_dark(fig, [ax])
plt.tight_layout()
p = os.path.join(OUT, "01_wm_quality_psnr.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 – Post-Attack PSNR  (vs original)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12,5))
for i, (m, d) in enumerate(ALL.items()):
    ax.bar(x + i*w, d["atk_psnr"], w, label=m, color=COLORS[i], alpha=0.88)
ax.set_xlabel("Image Index")
ax.set_ylabel("PSNR (dB)")
ax.set_title("Post-Attack Quality — PSNR vs Original")
ax.set_xticks(x + w)
ax.set_xticklabels([str(i) for i in range(10)])
ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
apply_dark(fig, [ax])
plt.tight_layout()
p = os.path.join(OUT, "02_postattack_psnr.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 – Post-Attack SSIM
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12,5))
for i, (m, d) in enumerate(ALL.items()):
    vals = [v for v in d["atk_ssim"] if not np.isnan(v)]
    xs   = [xi for xi, v in zip(x, d["atk_ssim"]) if not np.isnan(v)]
    ax.bar(np.array(xs) + i*w, vals, w, label=m, color=COLORS[i], alpha=0.88)
ax.set_xlabel("Image Index")
ax.set_ylabel("SSIM")
ax.set_title("Post-Attack Structural Similarity (SSIM)")
ax.set_xticks(x + w)
ax.set_xticklabels([str(i) for i in range(10)])
ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
apply_dark(fig, [ax])
plt.tight_layout()
p = os.path.join(OUT, "03_postattack_ssim.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 – Post-Attack LPIPS  (lower = less perceptual damage)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12,5))
for i, (m, d) in enumerate(ALL.items()):
    vals = [v for v in d["atk_lpips"] if not np.isnan(v)]
    xs   = [xi for xi, v in zip(x, d["atk_lpips"]) if not np.isnan(v)]
    ax.bar(np.array(xs) + i*w, vals, w, label=m, color=COLORS[i], alpha=0.88)
ax.set_xlabel("Image Index")
ax.set_ylabel("LPIPS ↓")
ax.set_title("Post-Attack Perceptual Loss (LPIPS — lower is better)")
ax.set_xticks(x + w)
ax.set_xticklabels([str(i) for i in range(10)])
ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
apply_dark(fig, [ax])
plt.tight_layout()
p = os.path.join(OUT, "04_postattack_lpips.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 – Post-Attack Raw BER  per image, all methods
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12,5))
for i, (m, d) in enumerate(ALL.items()):
    ax.bar(x + i*w, [v*100 for v in d["ber_raw"]], w, label=m, color=COLORS[i], alpha=0.88)
ax.axhline(50, color="#e6edf3", lw=0.8, linestyle=":", label="Random (50%)")
ax.set_xlabel("Image Index")
ax.set_ylabel("Raw BER (%)")
ax.set_title("Post-Attack Raw Bit Error Rate (Before ECC Correction)")
ax.set_xticks(x + w)
ax.set_xticklabels([str(i) for i in range(10)])
ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
apply_dark(fig, [ax])
plt.tight_layout()
p = os.path.join(OUT, "05_postattack_ber_raw.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6 – Post-Attack Final BER  (after ECC correction)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12,5))
for i, (m, d) in enumerate(ALL.items()):
    ax.bar(x + i*w, [v*100 for v in d["ber_final"]], w, label=m, color=COLORS[i], alpha=0.88)
ax.set_xlabel("Image Index")
ax.set_ylabel("Final BER (%)")
ax.set_title("Post-Attack Final BER (After Full ECC Pipeline)")
ax.set_xticks(x + w)
ax.set_xticklabels([str(i) for i in range(10)])
ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
apply_dark(fig, [ax])
plt.tight_layout()
p = os.path.join(OUT, "06_postattack_ber_final.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 7 – Survival Rate  (bar chart)
# ═══════════════════════════════════════════════════════════════════════════════
survival = [sum(d["recovered"]) / len(d["recovered"]) * 100 for d in ALL.values()]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(METHODS, survival, color=COLORS, width=0.4, alpha=0.9)
for bar, val in zip(bars, survival):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom", color="#e6edf3", fontsize=11, fontweight="bold")
ax.set_ylim(0, 115)
ax.set_ylabel("Message Recovery Rate (%)")
ax.set_title("Post-Attack Watermark Survival Rate\n(% images where message fully recovered)")
apply_dark(fig, [ax])
ax.tick_params(axis='x', labelsize=9)
plt.tight_layout()
p = os.path.join(OUT, "07_survival_rate.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 8 – Radar / Spider chart: avg metrics per method
# ═══════════════════════════════════════════════════════════════════════════════
cats = ["Survival\n(%/100)", "PSNR_wm\n(norm)", "Atk PSNR\n(norm)", "Atk SSIM", "1-LPIPS"]
N = len(cats)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def norm(val, min_v, max_v):
    return (val - min_v) / (max_v - min_v + 1e-9)

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
ax.set_facecolor("#161b22")
fig.patch.set_facecolor("#0d1117")

all_psnr_wm  = [np.mean(d["psnr_wm"]) for d in ALL.values()]
all_atk_psnr = [np.nanmean(d["atk_psnr"]) for d in ALL.values()]
all_atk_ssim = [np.nanmean(d["atk_ssim"]) for d in ALL.values()]
all_atk_lpips= [np.nanmean(d["atk_lpips"]) for d in ALL.values()]

for i, (m, d) in enumerate(ALL.items()):
    surv   = sum(d["recovered"]) / len(d["recovered"])
    p_wm   = norm(np.mean(d["psnr_wm"]), min(all_psnr_wm), max(all_psnr_wm))
    a_psnr = norm(np.nanmean(d["atk_psnr"]), min(all_atk_psnr), max(all_atk_psnr))
    a_ssim = np.nanmean(d["atk_ssim"])
    a_lpips= 1 - np.nanmean(d["atk_lpips"])
    values = [surv, p_wm, a_psnr, a_ssim, a_lpips]
    values += values[:1]
    ax.plot(angles, values, color=COLORS[i], linewidth=2, label=m)
    ax.fill(angles, values, color=COLORS[i], alpha=0.12)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats, color="#e6edf3", fontsize=9)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25","0.5","0.75","1.0"], color="#8b949e", fontsize=7)
ax.tick_params(colors="#e6edf3")
ax.spines['polar'].set_color(STYLE["grid_color"])
ax.grid(color=STYLE["grid_color"], linestyle="--", alpha=0.4)
ax.set_title("Multi-Metric Radar Comparison", color="#e6edf3", fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
          facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
p = os.path.join(OUT, "08_radar_comparison.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 9 – BER Funnel: raw → final, per method (grouped lines)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
stages = ["Raw BER", "Final BER"]
for ax, (m, d), c in zip(axes, ALL.items(), COLORS):
    raw   = [v*100 for v in d["ber_raw"]]
    final = [v*100 for v in d["ber_final"]]
    for r, f, rec in zip(raw, final, d["recovered"]):
        lc = "#4caf82" if rec else "#e05c5c"
        ax.plot([0, 1], [r, f], color=lc, alpha=0.7, lw=1.5)
    ax.scatter([0]*10, raw,   color=c, s=40, zorder=5)
    ax.scatter([1]*10, final, color=c, s=40, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(stages)
    ax.set_title(m, fontsize=9)
    apply_dark(fig, [ax])
axes[0].set_ylabel("BER (%)")
fig.suptitle("BER Funnel: Raw → Final (Green = Recovered, Red = Failed)",
             color="#e6edf3", fontsize=12, y=1.02)
plt.tight_layout()
p = os.path.join(OUT, "09_ber_funnel_all.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 10 – Summary Dashboard  (2×3 grid)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0d1117")
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# (0,0) Survival
ax0 = fig.add_subplot(gs[0, 0])
bars = ax0.bar(range(3), survival, color=COLORS, alpha=0.9)
for bar, val in zip(bars, survival):
    ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f"{val:.0f}%", ha="center", color="#e6edf3", fontsize=8, fontweight="bold")
ax0.set_xticks(range(3))
ax0.set_xticklabels(["VAE\nz₀ BCH","DDIM\nBCH","DDIM\nLDPC"], fontsize=8)
ax0.set_ylim(0, 120)
ax0.set_title("Survival Rate (%)")
apply_dark(fig, [ax0])

# (0,1) Avg WM PSNR
ax1 = fig.add_subplot(gs[0, 1])
ax1.bar(range(3), [np.mean(d["psnr_wm"]) for d in ALL.values()], color=COLORS, alpha=0.9)
ax1.set_xticks(range(3))
ax1.set_xticklabels(["VAE\nz₀ BCH","DDIM\nBCH","DDIM\nLDPC"], fontsize=8)
ax1.set_title("Avg WM PSNR (dB)")
apply_dark(fig, [ax1])

# (0,2) Avg Post-attack PSNR
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(range(3), [np.nanmean(d["atk_psnr"]) for d in ALL.values()], color=COLORS, alpha=0.9)
ax2.set_xticks(range(3))
ax2.set_xticklabels(["VAE\nz₀ BCH","DDIM\nBCH","DDIM\nLDPC"], fontsize=8)
ax2.set_title("Avg Post-Attack PSNR (dB)")
apply_dark(fig, [ax2])

# (1,0) Avg Atk SSIM
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(range(3), [np.nanmean(d["atk_ssim"]) for d in ALL.values()], color=COLORS, alpha=0.9)
ax3.set_xticks(range(3))
ax3.set_xticklabels(["VAE\nz₀ BCH","DDIM\nBCH","DDIM\nLDPC"], fontsize=8)
ax3.set_title("Avg Post-Attack SSIM")
apply_dark(fig, [ax3])

# (1,1) Avg Atk LPIPS
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(range(3), [np.nanmean(d["atk_lpips"]) for d in ALL.values()], color=COLORS, alpha=0.9)
ax4.set_xticks(range(3))
ax4.set_xticklabels(["VAE\nz₀ BCH","DDIM\nBCH","DDIM\nLDPC"], fontsize=8)
ax4.set_title("Avg Post-Attack LPIPS ↓")
apply_dark(fig, [ax4])

# (1,2) Avg Final BER
ax5 = fig.add_subplot(gs[1, 2])
ax5.bar(range(3), [np.mean(d["ber_final"])*100 for d in ALL.values()], color=COLORS, alpha=0.9)
ax5.set_xticks(range(3))
ax5.set_xticklabels(["VAE\nz₀ BCH","DDIM\nBCH","DDIM\nLDPC"], fontsize=8)
ax5.set_title("Avg Final BER (%) ↓")
apply_dark(fig, [ax5])

patches = [mpatches.Patch(color=c, label=m) for c, m in zip(COLORS, METHODS)]
fig.legend(handles=patches, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02),
           facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)
fig.suptitle("ECC Watermarking — Full Comparative Metrics Dashboard",
             color="#e6edf3", fontsize=14, y=1.07)

p = os.path.join(OUT, "10_summary_dashboard.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")

# ─── Print avg table ──────────────────────────────────────────────────────────
print("\n── Average Metrics Summary ──")
print(f"{'Method':<22} {'Surv%':>6} {'WM PSNR':>8} {'AtkPSNR':>8} {'AtkSSIM':>8} {'LPIPS':>7} {'FinalBER':>9}")
for m, d in ALL.items():
    surv    = sum(d["recovered"])/len(d["recovered"])*100
    psnr_wm = np.mean(d["psnr_wm"])
    apsnr   = np.nanmean(d["atk_psnr"])
    assim   = np.nanmean(d["atk_ssim"])
    alpips  = np.nanmean(d["atk_lpips"])
    fber    = np.mean(d["ber_final"])*100
    print(f"{m:<22} {surv:>6.0f} {psnr_wm:>8.2f} {apsnr:>8.2f} {assim:>8.4f} {alpips:>7.4f} {fber:>9.2f}")
print(f"\nAll plots saved to: {OUT}")
