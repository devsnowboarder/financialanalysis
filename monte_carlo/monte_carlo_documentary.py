import os
import json
import asyncio
import numpy as np

# Force headless Agg backend on macOS to prevent MoviePy canvas buffer crashes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import edge_tts

# MoviePy v1.x and v2.x compatibility layer
try:
    from moviepy import VideoClip, AudioFileClip, concatenate_videoclips
    MOVIEPY_V2 = True
except ImportError:
    from moviepy.editor import VideoClip, AudioFileClip, concatenate_videoclips
    MOVIEPY_V2 = False

# =====================================================================
# 1. CONFIGURATION & IRS UNIFORM LIFETIME TABLE (Age 75+ SECURE 2.0)
# =====================================================================
DEFAULT_CONFIG = {
    "client_name": "Portfolio Stress-Test",
    "start_age": 66,
    "end_age": 95,
    "initial_savings": 500000,
    "monthly_ss": 3500,
    "monthly_withdrawal": 1500,
    "mean_return": 0.07,
    "volatility": 0.15,
    "num_simulations": 1000,
    "rmd_start_age": 75,
    "shock_year_index": 3,
    "shock_return": -0.22,
    "voice_anchor": "en-US-ChristopherNeural",
    "voice_analyst_fallback": "en-US-EricNeural"
}

IRS_RMD_TABLE = {
    75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8,
    85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9,
    90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9
}

def load_config(path="retirement_config.json"):
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                user_cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k in user_cfg and user_cfg[k] is not None:
                    cfg[k] = type(v)(user_cfg[k])
            return cfg
        except Exception as e:
            print(f"[Warning] Could not parse {path} ({e}). Using safe defaults.")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    return cfg

# =====================================================================
# 2. TAX ENGINE (Federal + California Single Filer Approximations)
# =====================================================================
def estimate_fed_ca_tax(ira_withdrawal, ss_annual):
    provisional_income = ira_withdrawal + (0.5 * ss_annual)
    if provisional_income <= 25000:
        taxable_ss = 0.0
    elif provisional_income <= 34000:
        taxable_ss = 0.50 * (provisional_income - 25000)
    else:
        taxable_ss = min(0.85 * ss_annual, 4500 + 0.85 * (provisional_income - 34000))

    fed_gross = ira_withdrawal + taxable_ss
    fed_taxable = max(0.0, fed_gross - 18000.0)

    fed_tax = 0.0
    brackets = [(11925, 0.10), (48475, 0.12), (103350, 0.22), (197300, 0.24)]
    prev_limit = 0.0
    for limit, rate in brackets:
        if fed_taxable > prev_limit:
            fed_tax += (min(fed_taxable, limit) - prev_limit) * rate
            prev_limit = limit

    ca_taxable = max(0.0, ira_withdrawal - 11000.0)
    ca_tax = 0.0
    ca_brackets = [(10756, 0.01), (25499, 0.02), (40245, 0.04), (55866, 0.06), (70606, 0.08), (360000, 0.093)]
    prev_ca = 0.0
    for limit, rate in ca_brackets:
        if ca_taxable > prev_ca:
            ca_tax += (min(ca_taxable, limit) - prev_ca) * rate
            prev_ca = limit

    return fed_tax + ca_tax

# =====================================================================
# 3. MONTE CARLO & SEQUENCE-OF-RETURNS STRESS SIMULATOR
# =====================================================================
def run_simulations(cfg):
    np.random.seed(42)
    ages = np.arange(int(cfg["start_age"]), int(cfg["end_age"]) + 1, dtype=int)
    n_years = len(ages)
    n_sims = int(cfg["num_simulations"])

    paths = np.zeros((n_sims, n_years), dtype=float)
    paths[:, 0] = float(cfg["initial_savings"])

    ss_annual = float(cfg["monthly_ss"]) * 12.0
    base_withdrawal = float(cfg["monthly_withdrawal"]) * 12.0

    for s in range(n_sims):
        bal = float(cfg["initial_savings"])
        for y in range(1, n_years):
            age = int(ages[y])
            ret = np.random.normal(cfg["mean_return"], cfg["volatility"])
            infl_withdrawal = base_withdrawal * ((1.025) ** y)

            rmd = (bal / IRS_RMD_TABLE.get(age, 10.0)) if (age >= cfg["rmd_start_age"] and bal > 0) else 0.0
            actual_withdrawal = max(infl_withdrawal, rmd)
            health_shock = 50000.0 if (age >= 80 and np.random.rand() < 0.15) else 0.0

            bal = (bal - actual_withdrawal - health_shock) * (1.0 + ret)
            paths[s, y] = max(0.0, bal)

    p10 = np.percentile(paths, 10, axis=0).ravel()
    p50 = np.percentile(paths, 50, axis=0).ravel()
    p90 = np.percentile(paths, 90, axis=0).ravel()

    idx_90_matches = np.where(ages == 90)[0]
    idx_90 = int(idx_90_matches[0]) if len(idx_90_matches) > 0 else (n_years - 1)
    survival_90 = float(np.mean(paths[:, idx_90] > 0) * 100.0)

    shock_path = np.zeros(n_years, dtype=float)
    shock_path[0] = float(cfg["initial_savings"])

    cashflows = {
        "ss": np.full(n_years, ss_annual, dtype=float),
        "base_w": np.zeros(n_years, dtype=float),
        "rmd_excess": np.zeros(n_years, dtype=float),
        "taxes": np.zeros(n_years, dtype=float)
    }

    bal = float(cfg["initial_savings"])
    for y in range(n_years):
        age = int(ages[y])
        infl_w = base_withdrawal * ((1.025) ** y)
        rmd = (bal / IRS_RMD_TABLE.get(age, 10.0)) if (age >= cfg["rmd_start_age"] and bal > 0) else 0.0

        total_ira_w = min(bal, max(infl_w, rmd))
        rmd_excess = max(0.0, rmd - infl_w) if bal >= rmd else 0.0
        reg_w = max(0.0, total_ira_w - rmd_excess)

        cashflows["base_w"][y] = reg_w
        cashflows["rmd_excess"][y] = rmd_excess
        cashflows["taxes"][y] = estimate_fed_ca_tax(total_ira_w, ss_annual)

        if y < n_years - 1:
            if y + 1 == cfg["shock_year_index"]:
                r = cfg["shock_return"]
            elif y + 1 == cfg["shock_year_index"] + 1:
                r = -0.10
            else:
                r = cfg["mean_return"]
            bal = max(0.0, (bal - total_ira_w) * (1.0 + r))
            shock_path[y + 1] = bal

    return {
        "ages": ages,
        "paths": paths,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "idx_90": idx_90,
        "survival_90": survival_90,
        "shock_path": shock_path,
        "cashflows": cashflows
    }

# =====================================================================
# 4. DUAL-VOICE AUDIO ENGINE (Single Event Loop)
# =====================================================================
async def synthesize_all_tts(tasks):
    for text, voice, out_path in tasks:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out_path)

def build_audio_tracks(cfg, sim_data, audio_dir="audio_tracks"):
    os.makedirs(audio_dir, exist_ok=True)
    idx_90 = sim_data["idx_90"]
    med_90 = sim_data["p50"][idx_90]
    shock_90 = sim_data["shock_path"][idx_90]
    surv_90 = sim_data["survival_90"]

    scenes = [
        {
            "id": "scene1_intro",
            "type": "wire",
            "voice": cfg["voice_anchor"],
            "text": (
                f"Welcome to the Monte Carlo Retirement Runway and Tax Stress-Tester. "
                f"We are modeling one thousand market trajectories from age {cfg['start_age']} to {cfg['end_age']}, "
                f"starting with {cfg['initial_savings']:,} dollars in savings, {cfg['monthly_ss']:,} dollars per month "
                f"in Social Security, and {cfg['monthly_withdrawal']:,} dollars in monthly portfolio withdrawals. "
                f"Across all one thousand simulations, the portfolio achieves a {surv_90:.1f} percent survival rate at age 90."
            )
        },
        {
            "id": "scene2_shock",
            "type": "custom_or_fallback",
            "custom_file": "custom_voice/year3_shock.m4a",
            "voice": cfg["voice_analyst_fallback"],
            "text": (
                f"Here is where sequence-of-returns risk changes the picture. Watch the crimson trajectory in Year 3 at age 69. "
                f"A twenty-two percent market drop combined with a ten percent aftershock forces the portfolio to liquidate shares at the bottom. "
                f"Instead of compounding along the median path to {int(med_90):,} dollars by age 90, the shocked runway compresses to "
                f"{int(shock_90):,} dollars—wiping out {int(max(0, med_90 - shock_90)):,} dollars in late-life cushion."
            )
        },
        {
            "id": "scene3_rmd_tax",
            "type": "wire",
            "voice": cfg["voice_anchor"],
            "text": (
                f"On the right panel, look at the income and tax stack beginning at age {cfg['rmd_start_age']}. "
                f"Under SECURE 2.0 rules, Required Minimum Distributions kick in using the IRS Uniform Lifetime Table. "
                f"Even on the shocked trajectory, mandatory IRA withdrawals push provisional income past the 85 percent "
                f"Social Security taxation threshold, creating a compounding Federal and California tax drag."
            )
        },
        {
            "id": "scene4_strategy",
            "type": "custom_or_fallback",
            "custom_file": "custom_voice/tax_strategy.m4a",
            "voice": cfg["voice_analyst_fallback"],
            "text": (
                "The takeaway is clear: the golden window between age 66 and 74 is your prime tax-harvesting runway. "
                "By executing partial Roth conversions during low-tax years before age 75 RMDs begin, and holding a two-year "
                "cash buffer to avoid selling equities during a Year 3 crash, you protect both your tax bracket and your age 90 survival."
            )
        }
    ]

    with open("narration_script.txt", "w") as f:
        for s in scenes:
            f.write(f"=== {s['id'].upper()} ({s['type']}) ===\n{s['text']}\n\n")

    audio_paths = []
    tts_tasks = []
    for s in scenes:
        if s["type"] == "custom_or_fallback" and os.path.exists(s["custom_file"]):
            print(f"[Audio] Using custom recorded track: {s['custom_file']}")
            audio_paths.append(s["custom_file"])
        else:
            out_mp3 = os.path.join(audio_dir, f"{s['id']}.mp3")
            tts_tasks.append((s["text"], s["voice"], out_mp3))
            audio_paths.append(out_mp3)

    if tts_tasks:
        print(f"[Audio] Synthesizing {len(tts_tasks)} TTS tracks via edge_tts...")
        asyncio.run(synthesize_all_tts(tts_tasks))

    return scenes, audio_paths

# =====================================================================
# 5. MATPLOTLIB DARK-THEME DOCUMENTARY RENDERER (Dimension-Safe)
# =====================================================================
def create_scene_clip(scene_idx, duration, cfg, sim_data):
    ages = np.asarray(sim_data["ages"], dtype=float).ravel()
    n_years = len(ages)
    paths = np.asarray(sim_data["paths"], dtype=float)
    p10 = np.asarray(sim_data["p10"], dtype=float).ravel()
    p50 = np.asarray(sim_data["p50"], dtype=float).ravel()
    p90 = np.asarray(sim_data["p90"], dtype=float).ravel()
    shock_path = np.asarray(sim_data["shock_path"], dtype=float).ravel()
    cf = sim_data["cashflows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("#0B0F19")

    def make_frame(t):
        progress = min(1.0, max(0.0, t / max(0.1, duration * 0.85)))
        ax1.clear()
        ax2.clear()

        for ax in (ax1, ax2):
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#9CA3AF", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#374151")
            ax.grid(True, color="#1F2937", linestyle="--", linewidth=0.8)

        # Strictly clamp slice length so x and y always match 1-to-1
        max_idx = max(2, min(n_years, int(round(progress * n_years)))) if scene_idx in (0, 1) else n_years
        cur_ages = ages[:max_idx]
        L = len(cur_ages)

        # --- LEFT PANEL ---
        for i in range(min(50, paths.shape[0])):
            y_spaghetti = paths[i, :L].ravel() / 1000.0
            ax1.plot(cur_ages, y_spaghetti, color="#3B82F6", alpha=0.08, linewidth=0.8)

        ax1.fill_between(cur_ages, p10[:L] / 1000.0, p90[:L] / 1000.0,
                         color="#3B82F6", alpha=0.22, label="10th–90th Percentile Fan")
        ax1.plot(cur_ages, p50[:L] / 1000.0, color="#10B981", linewidth=2.8, label="Median Trajectory (50th)")

        if scene_idx >= 1:
            ax1.plot(cur_ages, shock_path[:L] / 1000.0, color="#EF4444", linewidth=3.2,
                     label="Year 3 Shock (-22% Drawdown)")
            shock_idx = min(int(cfg["shock_year_index"]), n_years - 1)
            if L > shock_idx:
                shock_age = ages[shock_idx]
                shock_val = shock_path[shock_idx] / 1000.0
                ax1.scatter([shock_age], [shock_val], color="#EF4444", s=90, zorder=5)
                ax1.annotate("Yr 3 Crash (-22%)", xy=(shock_age, shock_val),
                             xytext=(shock_age - 1.5, shock_val + 180),
                             color="#FCA5A5", fontweight="bold", fontsize=10,
                             arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.5))

        ax1.axvline(90, color="#F59E0B", linestyle=":", linewidth=1.5, label=f"Age 90 Survival: {sim_data['survival_90']:.1f}%")
        ax1.set_title("1,000 Monte Carlo Runs vs. Sequence-of-Returns Shock", color="#F9FAFB", fontsize=13.5, fontweight="bold", pad=12)
        ax1.set_xlabel("Age", color="#D1D5DB", fontsize=11)
        ax1.set_ylabel("Portfolio Balance ($ Thousands)", color="#D1D5DB", fontsize=11)
        ax1.set_xlim(ages[0], ages[-1])
        ax1.set_ylim(0, max(float(np.max(p90)) / 1000.0 * 1.05, 1200.0))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%0.0fk"))
        ax1.legend(loc="upper left", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=9)

        # --- RIGHT PANEL ---
        bar_idx = L if scene_idx < 2 else (max(2, min(n_years, int(round(progress * n_years)))) if scene_idx == 2 else n_years)
        b_ages = ages[:bar_idx]
        B = len(b_ages)

        ss_k = np.asarray(cf["ss"][:B], dtype=float).ravel() / 1000.0
        base_k = np.asarray(cf["base_w"][:B], dtype=float).ravel() / 1000.0
        rmd_k = np.asarray(cf["rmd_excess"][:B], dtype=float).ravel() / 1000.0
        tax_k = np.asarray(cf["taxes"][:B], dtype=float).ravel() / 1000.0

        ax2.bar(b_ages, ss_k, color="#3B82F6", width=0.65, label="Social Security ($42k/yr)")
        ax2.bar(b_ages, base_k, bottom=ss_k, color="#10B981", width=0.65, label="Base IRA Withdrawal")
        ax2.bar(b_ages, rmd_k, bottom=ss_k + base_k, color="#F59E0B", width=0.65, label="Mandatory RMD Excess (Age 75+)")
        ax2.bar(b_ages, tax_k, bottom=ss_k + base_k + rmd_k, color="#EF4444", width=0.65, label="Fed + CA Income Tax Drag")

        ax2.axvline(cfg["rmd_start_age"] - 0.5, color="#F59E0B", linestyle="--", linewidth=1.5)
        ax2.text(cfg["rmd_start_age"] + 0.3, 95, "SECURE 2.0 RMDs Start (Age 75)", color="#FBBF24", fontsize=9.5, fontweight="bold")

        if scene_idx == 3:
            ax2.axvspan(66, 74.5, color="#10B981", alpha=0.12, label="Roth Conversion Window (66–74)")

        ax2.set_title("Annual Cash-Flow Stack, RMD Surge & Tax Bracket Drag", color="#F9FAFB", fontsize=13.5, fontweight="bold", pad=12)
        ax2.set_xlabel("Age", color="#D1D5DB", fontsize=11)
        ax2.set_ylabel("Annual Cash Flow & Taxes ($ Thousands)", color="#D1D5DB", fontsize=11)
        ax2.set_xlim(ages[0] - 0.8, ages[-1] + 0.8)
        ax2.set_ylim(0, 115)
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%0.0fk"))
        ax2.legend(loc="upper left", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=9)

        fig.suptitle(
            f"RETIREMENT RUNWAY & TAX STRESS-TESTER  |  Age 66–95  |  90th Birthday Survival: {sim_data['survival_90']:.1f}%",
            color="#60A5FA", fontsize=15, fontweight="bold", y=0.97
        )
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.10, wspace=0.18)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    clip = VideoClip(make_frame, duration=duration)
    return clip

# =====================================================================
# 6. MAIN PIPELINE ASSEMBLY
# =====================================================================
def main():
    print("1. Loading retirement configuration...")
    cfg = load_config("retirement_config.json")

    print("2. Running 1,000 Monte Carlo paths & Year-3 Sequence Shock model...")
    sim_data = run_simulations(cfg)

    print("3. Generating Dual-Voice Narration Tracks...")
    scenes, audio_paths = build_audio_tracks(cfg, sim_data)

    print("4. Rendering Animated Matplotlib Documentary Scenes...")
    video_clips = []
    for idx, (scene, audio_path) in enumerate(zip(scenes, audio_paths)):
        audio_clip = AudioFileClip(audio_path)
        scene_duration = float(audio_clip.duration) + 0.4
        print(f"   -> Rendering {scene['id']} ({scene_duration:.1f}s)...")
        v_clip = create_scene_clip(idx, scene_duration, cfg, sim_data)

        if MOVIEPY_V2:
            v_clip = v_clip.with_audio(audio_clip)
        else:
            v_clip = v_clip.set_audio(audio_clip)
        video_clips.append(v_clip)

    print("5. Concatenating and exporting final MP4...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    output_filename = "monte_carlo_tax_stress_test.mp4"
    final_video.write_videofile(
        output_filename,
        fps=12,
        codec="libx264",
        audio_codec="aac"
    )
    plt.close("all")
    print(f"\n[Done] Documentary exported to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    main()