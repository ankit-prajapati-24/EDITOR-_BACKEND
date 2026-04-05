def extract_beat_timestamps(audio_path, min_gap=0.3):
    """
    Extracts beat timestamps from an audio file using librosa.

    Args:
        audio_path (str): Path to the audio file.
        min_gap    (float): Minimum gap in seconds between beats (default: 0.3).
                            Beats closer than this are filtered out to avoid jitter.

    Returns:
        List[float]: Beat timestamps in seconds, e.g. [0.0, 0.5, 1.0, ...]
    """
    import librosa
    import numpy as np

    if not hasattr(extract_beat_timestamps, "_installed"):
        try:
            import librosa  # noqa: F811
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa", "-q"])
            import librosa  # noqa: F811
        extract_beat_timestamps._installed = True

    # ── Load audio ────────────────────────────────────────────────────────────
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # ── Detect beats ──────────────────────────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")

    # ── Convert frames → seconds ──────────────────────────────────────────────
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times = np.round(beat_times, 3)

    # ── Filter out beats that are too close together ───────────────────────────
    filtered = []
    last_t   = -min_gap

    for t in beat_times:
        if t - last_t >= min_gap:
            filtered.append(float(t))
            last_t = t

    # Always include 0.0 as a reference point if not already present
    if filtered and filtered[0] > 0.0:
        filtered.insert(0, 0.0)

    print(f"✅ Detected {len(filtered)} beats  |  Tempo: {float(tempo):.1f} BPM  |  Audio: {y.shape[0]/sr:.1f}s")

    return filtered


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    BASE = "/content/drive/MyDrive/editor"

    beats = extract_beat_timestamps(
        audio_path=f"{BASE}/shisha.mp3",
        min_gap=0.3,
    )

    print(f"First 10 beats : {beats[:20]}")
    print(f"Total beats    : {len(beats)}")