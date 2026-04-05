def generate_captions(
    audio_path,
    model_size="small",
    language=None,
    min_segment_duration=0.5,
    merge_gap=0.3,
    max_merge_duration=5.0,
):
    import os
    import re
    import whisper

    NOISE_PATTERNS = re.compile(
        r"^\s*[\(\[\{].*?[\)\]\}]\s*$"
        r"|^\s*\.{2,}\s*$"
        r"|^\s*$",
        re.IGNORECASE,
    )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"⏳ Loading Whisper '{model_size}' model ...")
    model = whisper.load_model(model_size)

    transcribe_opts = dict(verbose=False, word_timestamps=False)
    if language:
        transcribe_opts["language"] = language
        print(f"🌐 Language forced: {language}")
    else:
        print("🌐 Language: auto-detect")

    print(f"🎙️  Transcribing: {audio_path} ...")
    result = model.transcribe(audio_path, **transcribe_opts)

    def clean_text(raw):
        text = raw.strip()
        text = text[0].upper() + text[1:] if text else text
        text = re.sub(r" {2,}", " ", text)
        return text

    filtered = []
    for seg in result["segments"]:
        duration = seg["end"] - seg["start"]
        text = clean_text(seg["text"])

        if duration < min_segment_duration:
            continue
        if NOISE_PATTERNS.match(text):
            continue
        if len(text) < 2:
            continue

        filtered.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": text,
        })

    # ── Merge ─────────────────────────────────────────
    merged = []
    for seg in filtered:
        if (
            merged
            and (seg["start"] - merged[-1]["end"]) <= merge_gap
            and (seg["end"] - merged[-1]["start"]) <= max_merge_duration
        ):
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = merged[-1]["text"].rstrip(".") + " " + seg["text"]
        else:
            merged.append(dict(seg))

    # ── Split into subtitle-friendly chunks ───────────
    def split_segment(seg):
        text = seg["text"]
        start = seg["start"]
        end = seg["end"]
        duration = end - start

        # If already short enough
        if duration <= 4 and len(text.split()) <= 10:
            return [seg]

        words = text.split()
        chunks = []
        current = []

        for word in words:
            current.append(word)
            joined = " ".join(current)

            # split conditions
            if (
                len(joined) >= 35
                or word.endswith((".", ",", "!", "?"))
                or len(current) >= 8
            ):
                chunks.append(joined.strip())
                current = []

        if current:
            chunks.append(" ".join(current).strip())

        # time distribution
        total_chars = sum(len(c) for c in chunks)
        time_cursor = start
        result_chunks = []

        for chunk in chunks:
            proportion = len(chunk) / total_chars if total_chars else 1 / len(chunks)
            chunk_duration = max(1.5, min(4.0, duration * proportion))

            chunk_start = time_cursor
            chunk_end = min(end, chunk_start + chunk_duration)

            result_chunks.append({
                "start": round(chunk_start, 3),
                "end": round(chunk_end, 3),
                "text": chunk,
            })

            time_cursor = chunk_end

        # fix last end
        if result_chunks:
            result_chunks[-1]["end"] = end

        return result_chunks

    final_captions = []
    for seg in merged:
        final_captions.extend(split_segment(seg))

    print(f"✅ Done! {len(final_captions)} subtitle-friendly segments.")
    return final_captions

# ── Quick-start (run directly or in Google Colab) ─────────────────────────────
if __name__ == "__main__":
    # Install dependencies (run once):
    #   !pip install openai-whisper
    #   !sudo apt install -y ffmpeg

    captions = generate_captions(
        audio_path="/content/drive/MyDrive/editor/shisha.mp3",
        model_size="large",     # tiny | base | small | medium | large
        language=None,          # e.g. "en", "hi", "fr" — or None to auto-detect
        min_segment_duration=0.5,
        merge_gap=0.3,
        max_merge_duration=5.0,
    )

    for cap in captions:
        print(f"[{cap['start']:>7.2f}s → {cap['end']:>7.2f}s]  {cap['text']}")