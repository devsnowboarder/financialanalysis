import asyncio
import sys, subprocess
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "edge-tts", "yfinance", "matplotlib", "pandas", "moviepy>=2.0.0", "pydantic"
])
import os
from pathlib import Path
from typing import List
import edge_tts
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from pydantic import BaseModel
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips


# 1. Define Structured Storyboard Schema
class DocumentaryScene(BaseModel):
    scene_id: int
    title: str
    narration: str
    chart_type: str  # 'price_ma', 'drawdown', 'normalized_comp'


# 2. Fetch Data & Generate Data-Driven Storyboard
def build_storyboard(ticker: str = "NVDA", benchmark: str = "SPY", period: str = "1y") -> tuple[
    pd.DataFrame, List[DocumentaryScene]]:
    print(f"[*] Fetching {period} market data for {ticker} and {benchmark}...")
    df_target = yf.Ticker(ticker).history(period=period)
    df_bench = yf.Ticker(benchmark).history(period=period)

    df = pd.DataFrame({
        "Close": df_target["Close"],
        "Volume": df_target["Volume"],
        "Bench_Close": df_bench["Close"]
    }).dropna()

    # Calculate metrics for dynamic narration
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Norm_Target"] = (df["Close"] / df["Close"].iloc[0] - 1) * 100
    df["Norm_Bench"] = (df["Bench_Close"] / df["Bench_Close"].iloc[0] - 1) * 100

    rolling_max = df["Close"].cummax()
    df["Drawdown"] = ((df["Close"] - rolling_max) / rolling_max) * 100

    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    total_return = df["Norm_Target"].iloc[-1]
    bench_return = df["Norm_Bench"].iloc[-1]
    max_dd = df["Drawdown"].min()

    # Tip: You can swap this hardcoded list with a Gemini API structured output call
    # passing the summary stats above so an LLM writes custom commentary!
    scenes = [
        DocumentaryScene(
            scene_id=1,
            title=f"{ticker}: Price Trajectory & 50-Day Trend",
            narration=(
                f"Over the past year, {ticker} moved from {start_price:.2f} dollars "
                f"to {end_price:.2f} dollars. Notice how the 50-day moving average acted "
                f"as a structural pivot during major momentum shifts."
            ),
            chart_type="price_ma"
        ),
        DocumentaryScene(
            scene_id=2,
            title=f"{ticker}: Underwater Drawdown Risk",
            narration=(
                f"High returns rarely come without volatility. Looking at the underwater drawdown profile, "
                f"{ticker} experienced a maximum pullback of {abs(max_dd):.1f} percent from its peak, "
                f"testing investor conviction before recovering."
            ),
            chart_type="drawdown"
        ),
        DocumentaryScene(
            scene_id=3,
            title=f"Relative Alpha: {ticker} vs {benchmark}",
            narration=(
                f"When normalized on a percentage basis against {benchmark}, the divergence is clear. "
                f"{ticker} delivered a cumulative return of {total_return:.1f} percent, "
                f"compared to {bench_return:.1f} percent for the broader index."
            ),
            chart_type="normalized_comp"
        )
    ]
    return df, scenes


# 3. Render 1080p Dark-Mode Chart for Each Scene
def render_scene_chart(df: pd.DataFrame, scene: DocumentaryScene, ticker: str, benchmark: str, output_path: Path):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)  # 1920x1080 resolution
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    if scene.chart_type == "price_ma":
        ax.plot(df.index, df["Close"], color="#00f5d4", linewidth=2.5, label=f"{ticker} Close ($)")
        ax.plot(df.index, df["SMA_50"], color="#ff9f1c", linewidth=2, linestyle="--", label="50-Day SMA")
        ax.set_ylabel("Price (USD)", fontsize=14, color="#c9d1d9")

    elif scene.chart_type == "drawdown":
        ax.fill_between(df.index, df["Drawdown"], 0, color="#ff4d6d", alpha=0.45)
        ax.plot(df.index, df["Drawdown"], color="#ff4d6d", linewidth=2, label="Drawdown (%)")
        ax.set_ylabel("Drawdown from Peak (%)", fontsize=14, color="#c9d1d9")

    elif scene.chart_type == "normalized_comp":
        ax.plot(df.index, df["Norm_Target"], color="#00bbf9", linewidth=2.5, label=f"{ticker} Return (%)")
        ax.plot(df.index, df["Norm_Bench"], color="#8d99ae", linewidth=2, linestyle=":",
                label=f"{benchmark} Return (%)")
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
    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


# 4. Synthesize Voiceover Audio (Async)
async def synthesize_voiceover(text: str, output_audio_path: Path, voice: str = "en-US-ChristopherNeural"):
    communicate = edge_tts.Communicate(text, voice, rate="+0%")
    await communicate.save(str(output_audio_path))


# 5. Master Orchestrator
async def build_documentary(ticker: str = "NVDA", benchmark: str = "SPY"):
    work_dir = Path("../AIExamples/build_assets")
    work_dir.mkdir(exist_ok=True)

    df, scenes = build_storyboard(ticker=ticker, benchmark=benchmark)
    video_clips = []

    for scene in scenes:
        print(f"[*] Processing Scene {scene.scene_id}: {scene.title}")
        img_path = work_dir / f"scene_{scene.scene_id}.png"
        audio_path = work_dir / f"scene_{scene.scene_id}.mp3"

        # Generate Chart & Voiceover
        render_scene_chart(df, scene, ticker, benchmark, img_path)
        await synthesize_voiceover(scene.narration, audio_path)

        # Load Audio & Lock Image Duration to Audio Duration (+0.5s breathing room)
        audio_clip = AudioFileClip(str(audio_path))
        clip_duration = audio_clip.duration + 0.5

        img_clip = (
            ImageClip(str(img_path))
            .with_duration(clip_duration)
            .with_audio(audio_clip)
        )
        video_clips.append(img_clip)

    output_video = f"{ticker}_documentary_brief.mp4"
    print(f"[*] Rendering final MP4 to {output_video}...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video.write_videofile(
        output_video,
        fps=24,
        codec="libx264",
        audio_codec="aac"
    )
    print(f"[+] Complete! Saved to {os.path.abspath(output_video)}")

async def run_batch():
    watchlist = ["AAPL", "MSFT", "TSLA", "AMD"]
    for symbol in watchlist:
        await build_documentary(ticker=symbol, benchmark="SPY")

if __name__ == "__main__":
    asyncio.run(run_batch())