import asyncio
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import edge_tts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel
from moviepy import VideoClip, AudioFileClip, VideoFileClip, concatenate_videoclips

# Map yfinance period codes to natural spoken narration phrases
PERIOD_LABELS = {
    "1mo": "Over the past month",
    "3mo": "Over the past three months",
    "6mo": "Over the past six months",
    "9mo": "Over the past nine months",
    "ytd": "Year to date",
    "1y": "Over the past year",
    "2y": "Over the past two years",
    "5y": "Over the past five years",
}

# Common ticker typos mapped to their official Yahoo Finance symbols
TICKER_TYPO_MAP = {
    "APPL": "AAPL",
    "GOOG": "GOOGL",
    "BRK.B": "BRK-B",
    "BRKB": "BRK-B",
    "BRK.A": "BRK-A",
    "FB": "META",
}

DEFAULT_BATCH = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "NFLX", "NVDA", "AMD"]


# 0. Startup System Dependency Check
def verify_ffmpeg():
    """Checks if ffmpeg and ffprobe are installed and available in the system PATH."""
    required_tools = ["ffmpeg", "ffprobe"]
    missing = [tool for tool in required_tools if not shutil.which(tool)]

    if missing:
        print("\n" + "=" * 60)
        print(f"[!] FATAL ERROR: Missing required tool(s): {', '.join(missing)}")
        print("=" * 60)
        print("FFmpeg is required by MoviePy to render videos and by the live mic recorder.")
        print("To install it on macOS, open your terminal and run:")
        print("    brew install ffmpeg")
        print("\nAfter installation completes, restart PyCharm and re-run your script.")
        print("=" * 60 + "\n")
        sys.exit(1)


# 1. Define Structured Schemas
class DocumentaryScene(BaseModel):
    scene_id: int
    title: str
    narration: str
    chart_type: str  # 'price_ma', 'drawdown', 'normalized_comp'


class BatchSummaryItem(BaseModel):
    ticker: str
    total_return: float
    max_drawdown: float
    video_path: Path


# 2. Fetch Data & Generate Data-Driven Storyboard
def build_storyboard(
        ticker: str,
        benchmark: str,
        period: str,
        df_bench_cached: pd.DataFrame
) -> tuple[pd.DataFrame, List[DocumentaryScene], float, float]:
    print(f"[*] Fetching {period} market data for {ticker}...")
    df_target = yf.Ticker(ticker).history(period=period)

    if df_target.empty:
        raise ValueError(f"No data returned from Yahoo Finance for '{ticker}'.")

    df = pd.DataFrame({
        "Close": df_target["Close"],
        "Volume": df_target["Volume"],
        "Bench_Close": df_bench_cached["Close"]
    }).dropna()

    if df.empty:
        raise ValueError(f"No overlapping trading dates for {ticker} and {benchmark}.")

    sma_window = 20 if period in ("1mo", "3mo") else 50
    df["SMA"] = df["Close"].rolling(sma_window, min_periods=1).mean()
    df["Norm_Target"] = (df["Close"] / df["Close"].iloc[0] - 1) * 100
    df["Norm_Bench"] = (df["Bench_Close"] / df["Bench_Close"].iloc[0] - 1) * 100

    rolling_max = df["Close"].cummax()
    df["Drawdown"] = ((df["Close"] - rolling_max) / rolling_max) * 100

    start_price = float(df["Close"].iloc[0])
    end_price = float(df["Close"].iloc[-1])
    total_return = float(df["Norm_Target"].iloc[-1])
    bench_return = float(df["Norm_Bench"].iloc[-1])
    max_dd = float(df["Drawdown"].min())

    time_phrase = PERIOD_LABELS.get(period, f"Over the {period} period")

    scenes = [
        DocumentaryScene(
            scene_id=1,
            title=f"{ticker} ({period.upper()}): Price & {sma_window}-Day Trend",
            narration=(
                f"{time_phrase}, {ticker} moved from {start_price:.2f} dollars "
                f"to {end_price:.2f} dollars. Watch how the {sma_window}-day moving average "
                f"tracked structural momentum shifts across the timeframe."
            ),
            chart_type="price_ma"
        ),
        DocumentaryScene(
            scene_id=2,
            title=f"{ticker} ({period.upper()}): Underwater Drawdown Risk",
            narration=(
                f"High returns rarely come without volatility. Looking at the underwater drawdown profile, "
                f"{ticker} experienced a maximum pullback of {abs(max_dd):.1f} percent from its peak "
                f"before stabilizing."
            ),
            chart_type="drawdown"
        ),
        DocumentaryScene(
            scene_id=3,
            title=f"Relative Alpha ({period.upper()}): {ticker} vs {benchmark}",
            narration=(
                f"When normalized on a percentage basis against {benchmark}, the performance gap stands out. "
                f"{ticker} delivered a cumulative return of {total_return:.1f} percent, "
                f"compared to {bench_return:.1f} percent for the benchmark."
            ),
            chart_type="normalized_comp"
        )
    ]
    return df, scenes, total_return, max_dd


def save_ticker_narration_script(ticker: str, scenes: List[DocumentaryScene]):
    """Saves a dedicated narration text file for each ticker in custom_voice/ (e.g., googl_narration.txt)."""
    custom_dir = Path("custom_voice")
    custom_dir.mkdir(exist_ok=True)

    filename = f"{ticker.lower()}_narration.txt"
    ticker_script_path = custom_dir / filename

    lines = []
    for scene in scenes:
        lines.append(f"[{ticker}_scene_{scene.scene_id}]")
        lines.append(scene.narration)
        lines.append("")

    script_content = "\n".join(lines)
    ticker_script_path.write_text(script_content, encoding="utf-8")

    print(f"\n--- 📝 Narration Script Saved: custom_voice/{filename} ---")
    print(script_content.strip())
    print("-" * 60)


# 3. Create Animated VideoClip Locked to Voiceover Duration
def create_animated_scene_clip(
        df: pd.DataFrame,
        scene: DocumentaryScene,
        ticker: str,
        benchmark: str,
        clip_duration: float,
        draw_duration: float
) -> tuple[VideoClip, plt.Figure]:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    dates = df.index
    n_points = len(df)
    ax.set_xlim(dates[0], dates[-1])

    if scene.chart_type == "price_ma":
        y_min = min(df["Close"].min(), df["SMA"].min()) * 0.95
        y_max = max(df["Close"].max(), df["SMA"].max()) * 1.05
        ax.set_ylim(y_min, y_max)
        line1, = ax.plot([], [], color="#00f5d4", linewidth=2.5, label=f"{ticker} Close ($)")
        line2, = ax.plot([], [], color="#ff9f1c", linewidth=2, linestyle="--", label="Moving Average")
        ax.set_ylabel("Price (USD)", fontsize=14, color="#c9d1d9")

    elif scene.chart_type == "drawdown":
        y_min = min(df["Drawdown"].min() * 1.15, -5.0)
        ax.set_ylim(y_min, 2.0)
        line1, = ax.plot([], [], color="#ff4d6d", linewidth=2.5, label="Drawdown (%)")
        ax.axhline(0, color="#4a5568", linewidth=1)
        ax.set_ylabel("Drawdown from Peak (%)", fontsize=14, color="#c9d1d9")
        fill_poly = [None]

    elif scene.chart_type == "normalized_comp":
        y_min = min(df["Norm_Target"].min(), df["Norm_Bench"].min()) - 5
        y_max = max(df["Norm_Target"].max(), df["Norm_Bench"].max()) + 5
        ax.set_ylim(y_min, y_max)
        line1, = ax.plot([], [], color="#00bbf9", linewidth=2.5, label=f"{ticker} Return (%)")
        line2, = ax.plot([], [], color="#8d99ae", linewidth=2, linestyle=":", label=f"{benchmark} Return (%)")
        ax.axhline(0, color="#4a5568", linewidth=1)
        ax.set_ylabel("Cumulative Return (%)", fontsize=14, color="#c9d1d9")

    ax.set_title(scene.title, fontsize=22, fontweight="bold", pad=20, color="#ffffff")
    ax.grid(color="#21262d", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=14, frameon=True, facecolor="#161b22", edgecolor="#30363d")
    ax.tick_params(colors="#8b949e", labelsize=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#30363d")

    plt.tight_layout(pad=3.0)

    def make_frame(t: float) -> np.ndarray:
        progress = min(1.0, t / max(draw_duration, 0.1))
        idx = max(2, int(progress * n_points))
        current_dates = dates[:idx]

        if scene.chart_type == "price_ma":
            line1.set_data(current_dates, df["Close"].iloc[:idx])
            line2.set_data(current_dates, df["SMA"].iloc[:idx])

        elif scene.chart_type == "drawdown":
            vals = df["Drawdown"].iloc[:idx]
            line1.set_data(current_dates, vals)
            if fill_poly[0] is not None:
                fill_poly[0].remove()
            fill_poly[0] = ax.fill_between(current_dates, vals, 0, color="#ff4d6d", alpha=0.4)

        elif scene.chart_type == "normalized_comp":
            line1.set_data(current_dates, df["Norm_Target"].iloc[:idx])
            line2.set_data(current_dates, df["Norm_Bench"].iloc[:idx])

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    return VideoClip(make_frame, duration=clip_duration), fig


# 4. Multi-Mode Voiceover Engine (AI TTS / Pre-Recorded Files / Live Mac Mic)
def record_live_mac_mic(prompt_text: str, voice_key: str, dest_path: Path, custom_dir: Path) -> bool:
    saved_custom_mp3 = custom_dir / f"{voice_key}.mp3"

    if saved_custom_mp3.exists() and saved_custom_mp3.stat().st_size > 1000:
        reuse = input(
            f"    [?] Found existing recording '{saved_custom_mp3.name}'. Reuse it? [Y/n] (default Y): "
        ).strip().lower()
        if reuse not in ("n", "no"):
            shutil.copy(saved_custom_mp3, dest_path)
            return True

    print("\n    ------------------------------------------------------------")
    print(f"    🎙️ LIVE MIC RECORDING [{voice_key}]")
    print("    Read the script below aloud:")
    print(f"\n    \"{prompt_text}\"\n")
    print("    ------------------------------------------------------------")
    action = input(
        "    Press ENTER to start recording (or type 's' + ENTER to skip to AI voice): "
    ).strip().lower()
    if action == "s":
        return False

    print("    🔴 RECORDING NOW... Speak clearly, then press ENTER as soon as you finish!")
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "avfoundation", "-i", ":0",
            "-acodec", "libmp3lame", "-ab", "192k", str(saved_custom_mp3)
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    input()
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()

    if saved_custom_mp3.exists() and saved_custom_mp3.stat().st_size > 1000:
        shutil.copy(saved_custom_mp3, dest_path)
        print(f"    [+] Saved your voice recording to custom_voice/{saved_custom_mp3.name}!\n")
        return True

    print("    [!] Microphone capture failed or was empty; falling back to AI TTS.\n")
    return False


async def synthesize_voiceover(
        text: str,
        output_audio_path: Path,
        voice_key: str = "",
        voice_mode: str = "1",
        voice: str = "en-US-ChristopherNeural"
):
    custom_dir = Path("custom_voice")
    custom_dir.mkdir(exist_ok=True)

    script_log = custom_dir / "narration_scripts_to_read.txt"
    with open(script_log, "a", encoding="utf-8") as f:
        f.write(f"[{voice_key or output_audio_path.stem}]\n{text}\n\n")

    if voice_mode == "2" and voice_key:
        for ext in (".m4a", ".mp3", ".wav"):
            candidate = custom_dir / f"{voice_key}{ext}"
            if candidate.exists():
                print(f"    🎙️ Using pre-recorded custom audio: {candidate.name}")
                shutil.copy(candidate, output_audio_path)
                return
        print(f"    [*] No custom file found for '{voice_key}' in custom_voice/; using AI TTS fallback.")

    elif voice_mode == "3" and voice_key:
        recorded_ok = record_live_mac_mic(text, voice_key, output_audio_path, custom_dir)
        if recorded_ok:
            return

    communicate = edge_tts.Communicate(text, voice, rate="+0%")
    await communicate.save(str(output_audio_path))


# 5. Single-Ticker Video Builder
async def build_documentary(
        ticker: str,
        benchmark: str,
        period: str,
        df_bench_cached: pd.DataFrame,
        output_dir: Path,
        voice_mode: str = "1"
) -> Optional[BatchSummaryItem]:
    work_dir = Path("build_assets") / ticker
    work_dir.mkdir(parents=True, exist_ok=True)

    df, scenes, total_return, max_dd = build_storyboard(
        ticker=ticker,
        benchmark=benchmark,
        period=period,
        df_bench_cached=df_bench_cached
    )

    # Automatically generate and save {ticker}_narration.txt
    save_ticker_narration_script(ticker, scenes)

    video_clips = []
    audio_clips = []
    figures_to_close = []

    try:
        for scene in scenes:
            print(f"    -> Preparing Scene {scene.scene_id}: {scene.title}")
            audio_path = work_dir / f"scene_{scene.scene_id}.mp3"
            voice_key = f"{ticker}_scene_{scene.scene_id}"

            await synthesize_voiceover(
                text=scene.narration,
                output_audio_path=audio_path,
                voice_key=voice_key,
                voice_mode=voice_mode
            )
            audio_clip = AudioFileClip(str(audio_path))
            audio_clips.append(audio_clip)

            draw_duration = audio_clip.duration
            clip_duration = draw_duration + 0.6

            anim_clip, fig = create_animated_scene_clip(
                df=df,
                scene=scene,
                ticker=ticker,
                benchmark=benchmark,
                clip_duration=clip_duration,
                draw_duration=draw_duration
            )
            anim_clip = anim_clip.with_audio(audio_clip)
            video_clips.append(anim_clip)
            figures_to_close.append(fig)

        output_video = output_dir / f"{ticker}_{period}_animated_brief.mp4"
        print(f"[*] Exporting MP4: {output_video.name}...")
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_video.write_videofile(
            str(output_video),
            fps=15,
            codec="libx264",
            audio_codec="aac",
            logger="bar"
        )
        final_video.close()

        return BatchSummaryItem(
            ticker=ticker,
            total_return=total_return,
            max_drawdown=max_dd,
            video_path=output_video
        )

    finally:
        for clip in video_clips:
            clip.close()
        for a_clip in audio_clips:
            a_clip.close()
        for fig in figures_to_close:
            plt.close(fig)


# 6. Opening Leaderboard Scene Generator
async def build_leaderboard_intro(
        items: List[BatchSummaryItem],
        benchmark: str,
        bench_return: float,
        period: str,
        output_dir: Path,
        voice_mode: str = "1"
) -> Optional[Path]:
    if not items:
        return None

    print("\n==========================================")
    print("🏆 Generating Opening Leaderboard Scene...")
    print("==========================================")

    work_dir = Path("build_assets") / "leaderboard"
    work_dir.mkdir(parents=True, exist_ok=True)

    sorted_desc = sorted(items, key=lambda x: x.total_return, reverse=True)
    leader = sorted_desc[0]
    laggard = sorted_desc[-1]
    beat_count = sum(1 for x in sorted_desc if x.total_return > bench_return)

    time_phrase = PERIOD_LABELS.get(period, f"Over the {period} period")
    narration = (
        f"Portfolio Leaderboard Recap. {time_phrase}, {beat_count} out of {len(sorted_desc)} "
        f"tracked equities outperformed the {benchmark} benchmark return of {bench_return:.1f} percent. "
        f"{leader.ticker} led the pack at {leader.total_return:+.1f} percent, "
        f"while {laggard.ticker} trailed at {laggard.total_return:+.1f} percent. "
        f"Let's break down each position in order of performance."
    )

    audio_path = work_dir / "leaderboard_intro.mp3"
    await synthesize_voiceover(
        text=narration,
        output_audio_path=audio_path,
        voice_key="leaderboard_intro",
        voice_mode=voice_mode
    )
    audio_clip = AudioFileClip(str(audio_path))
    draw_duration = audio_clip.duration
    clip_duration = draw_duration + 0.8

    sorted_asc = list(reversed(sorted_desc))
    labels = [x.ticker for x in sorted_asc]
    final_returns = np.array([x.total_return for x in sorted_asc], dtype=float)
    bar_colors = [
        "#00f5d4" if r >= bench_return else ("#00bbf9" if r >= 0 else "#ff4d6d")
        for r in final_returns
    ]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    min_val = min(float(final_returns.min()), bench_return, 0.0)
    max_val = max(float(final_returns.max()), bench_return, 0.0)
    span = max(max_val - min_val, 10.0)
    ax.set_xlim(min_val - span * 0.18, max_val + span * 0.22)

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, np.zeros_like(final_returns), color=bar_colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=15, fontweight="bold", color="#ffffff")

    ax.axvline(0, color="#4a5568", linewidth=1.2)
    ax.axvline(
        bench_return,
        color="#ff9f1c",
        linestyle="--",
        linewidth=2.2,
        label=f"{benchmark} Benchmark ({bench_return:+.1f}%)"
    )

    ax.set_title(
        f"Portfolio Leaderboard ({period.upper()}): Ranked by Cumulative Return",
        fontsize=22,
        fontweight="bold",
        pad=20,
        color="#ffffff"
    )
    ax.set_xlabel("Cumulative Return (%)", fontsize=14, color="#c9d1d9")
    ax.grid(axis="x", color="#21262d", linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=14, frameon=True, facecolor="#161b22", edgecolor="#30363d")
    ax.tick_params(colors="#8b949e", labelsize=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#30363d")

    value_texts = [
        ax.text(0, y, "", va="center", fontsize=13, fontweight="bold", color="#ffffff")
        for y in y_pos
    ]

    plt.tight_layout(pad=3.0)

    def make_frame(t: float) -> np.ndarray:
        raw_p = min(1.0, t / max(draw_duration, 0.1))
        eased_p = 1.0 - (1.0 - raw_p) ** 2
        current_vals = final_returns * eased_p

        for bar, val, txt in zip(bars, current_vals, value_texts):
            bar.set_width(val)
            offset = span * 0.015 if val >= 0 else -span * 0.015
            txt.set_position((val + offset, bar.get_y() + bar.get_height() / 2.0))
            txt.set_ha("left" if val >= 0 else "right")
            txt.set_text(f"{val:+.1f}%")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    intro_path = output_dir / f"00_LEADERBOARD_{period.upper()}.mp4"
    anim_clip = VideoClip(make_frame, duration=clip_duration).with_audio(audio_clip)
    try:
        anim_clip.write_videofile(
            str(intro_path),
            fps=15,
            codec="libx264",
            audio_codec="aac",
            logger="bar"
        )
        return intro_path
    finally:
        anim_clip.close()
        audio_clip.close()
        plt.close(fig)


# 7. Lossless Master Video Stitcher
def stitch_master_video(video_paths: List[Path], output_dir: Path, period: str) -> Optional[Path]:
    if len(video_paths) < 2:
        print("[*] Fewer than 2 clips to stitch; skipping master stitch.")
        return None

    master_path = output_dir / f"PORTFOLIO_MASTER_RECAP_{period.upper()}.mp4"
    concat_list_file = output_dir / "ffmpeg_concat_list.txt"

    print(f"\n[*] Stitching {len(video_paths)} clips into {master_path.name}...")

    try:
        lines = [f"file '{p.resolve()}'" for p in video_paths]
        concat_list_file.write_text("\n".join(lines), encoding="utf-8")

        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy", str(master_path)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        concat_list_file.unlink(missing_ok=True)
        size_mb = master_path.stat().st_size / (1024 * 1024)
        print(f"[+] Instant lossless master video saved: {master_path.resolve()} ({size_mb:.2f} MB)")
        return master_path

    except Exception as e:
        print(f"[*] Fast FFmpeg copy unavailable ({e}); falling back to MoviePy re-encode...")
        concat_list_file.unlink(missing_ok=True)

    loaded_clips = []
    try:
        loaded_clips = [VideoFileClip(str(p)) for p in video_paths]
        master_clip = concatenate_videoclips(loaded_clips, method="compose")
        master_clip.write_videofile(
            str(master_path),
            fps=15,
            codec="libx264",
            audio_codec="aac",
            logger="bar"
        )
        master_clip.close()
        return master_path
    finally:
        for c in loaded_clips:
            c.close()


# 8. Helper to Load & Normalize Batch Tickers
def parse_batch_tickers(raw_input: str) -> List[str]:
    raw_input = raw_input.strip()

    if not raw_input:
        stocklist_file = Path("stocklist.txt")
        if stocklist_file.exists():
            print("[*] Loading tickers from stocklist.txt...")
            lines = stocklist_file.read_text().splitlines()
            raw_tokens = [line.strip().upper() for line in lines if line.strip() and not line.startswith("#")]
        else:
            print(f"[*] Using default batch watchlist: {', '.join(DEFAULT_BATCH)}")
            raw_tokens = DEFAULT_BATCH
    else:
        raw_tokens = [t.strip().upper() for t in raw_input.replace(",", " ").split() if t.strip()]

    cleaned_tickers = []
    for token in raw_tokens:
        corrected = TICKER_TYPO_MAP.get(token, token)
        if corrected != token:
            print(f"[*] Auto-corrected typo '{token}' -> '{corrected}'")
        if corrected not in cleaned_tickers:
            cleaned_tickers.append(corrected)

    return cleaned_tickers


# 9. Master Batch Orchestrator
async def run_batch_pipeline(
        tickers: List[str],
        benchmark: str,
        period: str,
        combine_master: bool = True,
        voice_mode: str = "1"
):
    output_dir = Path("completed_videos")
    output_dir.mkdir(exist_ok=True)

    script_log = Path("custom_voice") / "narration_scripts_to_read.txt"
    if script_log.exists():
        script_log.unlink()

    print(f"\n[*] Pre-fetching benchmark '{benchmark}' ({period}) data once for the entire batch...")
    df_bench_cached = yf.Ticker(benchmark).history(period=period)
    if df_bench_cached.empty:
        print(f"[!] Fatal Error: Could not fetch benchmark data for '{benchmark}'. Check symbol.")
        return

    bench_total_return = float(
        (df_bench_cached["Close"].iloc[-1] / df_bench_cached["Close"].iloc[0] - 1) * 100
    )

    summary_items: List[BatchSummaryItem] = []
    failed = []

    total = len(tickers)
    for idx, symbol in enumerate(tickers, start=1):
        print(f"\n==========================================")
        print(f"🎬 Processing [{idx}/{total}]: {symbol} (vs {benchmark})")
        print(f"==========================================")
        try:
            item = await build_documentary(
                ticker=symbol,
                benchmark=benchmark,
                period=period,
                df_bench_cached=df_bench_cached,
                output_dir=output_dir,
                voice_mode=voice_mode
            )
            if item:
                summary_items.append(item)
        except Exception as e:
            print(f"[!] Skipping '{symbol}' due to error: {e}")
            failed.append((symbol, str(e)))

    master_video_path = None
    if combine_master and len(summary_items) > 1:
        ranked_items = sorted(summary_items, key=lambda x: x.total_return, reverse=True)

        intro_clip_path = await build_leaderboard_intro(
            items=ranked_items,
            benchmark=benchmark,
            bench_return=bench_total_return,
            period=period,
            output_dir=output_dir,
            voice_mode=voice_mode
        )

        ordered_video_paths = [x.video_path for x in ranked_items]
        if intro_clip_path:
            ordered_video_paths.insert(0, intro_clip_path)

        master_video_path = stitch_master_video(ordered_video_paths, output_dir, period)

    print("\n==========================================")
    print(f"✅ BATCH COMPLETE: {len(summary_items)}/{total} individual videos generated")
    print(f"📂 Output Folder: {output_dir.resolve()}")
    print(f"📝 Full Narration Script Log: {(Path('custom_voice') / 'narration_scripts_to_read.txt').resolve()}")
    if summary_items:
        print("\n📊 Leaderboard Ranking:")
        for rank, item in enumerate(sorted(summary_items, key=lambda x: x.total_return, reverse=True), start=1):
            print(
                f"   #{rank} {item.ticker:<6} | Return: {item.total_return:+6.1f}% | Max DD: {item.max_drawdown:6.1f}% | {item.video_path.name}")
    if master_video_path:
        print(f"\n🏆 Master Portfolio Video: {master_video_path.name}")
    if failed:
        print("\n⚠️ Failed Tickers:")
        for sym, err in failed:
            print(f"   - {sym}: {err}")
    print("==========================================")


if __name__ == "__main__":
    verify_ffmpeg()

    raw_tickers = input(
        "Enter tickers separated by commas (e.g., AAPL, NVDA, TSLA) "
        "or press Enter for stocklist.txt / default batch: "
    )
    batch_list = parse_batch_tickers(raw_tickers)

    raw_bench = input("Enter benchmark ticker (default SPY): ").strip().upper() or "SPY"
    user_bench = TICKER_TYPO_MAP.get(raw_bench, raw_bench)

    user_period = input(
        "Enter timeframe [1mo, 3mo, 6mo, 9mo, ytd, 1y, 2y, 5y] (default 1y): "
    ).strip().lower() or "1y"

    if user_period not in PERIOD_LABELS:
        print(f"[!] Unrecognized period '{user_period}', defaulting to '1y'.")
        user_period = "1y"

    print("\nSelect Voiceover Mode:")
    print("  [1] AI Neural Voice (edge-tts automatic)")
    print("  [2] Pre-recorded audio files in custom_voice/ folder (.m4a, .mp3, .wav)")
    print("  [3] Record MY voice live in terminal via Mac microphone")
    user_voice_mode = input("Enter voice mode [1/2/3] (default 1): ").strip() or "1"
    if user_voice_mode not in ("1", "2", "3"):
        user_voice_mode = "1"

    stitch_choice = input(
        "Stitch all batch videos into one ranked master MP4 with Leaderboard Intro? [Y/n] (default Y): "
    ).strip().lower()
    should_stitch = stitch_choice not in ("n", "no")

    asyncio.run(
        run_batch_pipeline(
            tickers=batch_list,
            benchmark=user_bench,
            period=user_period,
            combine_master=should_stitch,
            voice_mode=user_voice_mode
        )
    )