def create_video_from_image_and_audio(
    image_paths,
    audio_path,
    output_path,
    duration=10,
    cinematic=True,
    transition_duration=0.6,
):
    """
    Creates an MP4 slideshow video from multiple images with background audio.

    Args:
        image_paths         (list): List of image file paths.
                                    e.g. ["img1.jpg", "img2.jpg", "img3.jpg"]
        audio_path           (str): Path to the background audio file.
        output_path          (str): Path where the final MP4 will be saved.
        duration             (int): Fallback total duration (seconds) used when
                                    audio duration cannot be determined. (default: 10)
        cinematic           (bool): Apply Ken Burns zoom effect on each image. (default: True)
        transition_duration (float): Cross-fade length in seconds between images. (default: 0.6)
    """
    import os
    import numpy as np
    from moviepy.editor import (
        ImageClip,
        AudioFileClip,
        CompositeVideoClip,
        concatenate_audioclips,
        concatenate_videoclips,
    )

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not image_paths:
        raise ValueError("image_paths must be a non-empty list.")

    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # ── Load audio & resolve total video duration ─────────────────────────────
    audio_clip     = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    video_duration = max(audio_duration, duration)

    # ── Per-image clip duration (overlap-aware) ───────────────────────────────
    n = len(image_paths)
    # Each clip overlaps its neighbour by transition_duration, so the effective
    # timeline advances by (clip_dur - transition_duration) per clip except the last.
    # Solve: clip_dur + (n-1)*(clip_dur - transition_duration) = video_duration
    #  =>   clip_dur = (video_duration + (n-1)*transition_duration) / n
    clip_dur = (video_duration + (n - 1) * transition_duration) / n
    clip_dur = max(clip_dur, transition_duration + 0.5)   # safety floor

    # ── Build individual image clips ──────────────────────────────────────────
    def make_clip(img_path, clip_duration, zoom):
        """Return an ImageClip, optionally with a slow Ken Burns zoom."""
        base = ImageClip(img_path, duration=clip_duration)

        if not zoom:
            return base.fadein(transition_duration).fadeout(transition_duration)

        # Ken Burns: smoothly zoom from 1.0x → 1.08x (subtle & cinematic)
        w, h = base.size
        zoom_start, zoom_end = 1.0, 1.08

        def zoom_frame(get_frame, t):
            frame  = get_frame(t)
            factor = zoom_start + (zoom_end - zoom_start) * (t / clip_duration)
            new_w  = int(w * factor)
            new_h  = int(h * factor)

            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            arr = np.array(img)

            # Centre-crop back to original size
            x0 = (new_w - w) // 2
            y0 = (new_h - h) // 2
            return arr[y0:y0 + h, x0:x0 + w]

        zoomed = base.fl(zoom_frame, apply_to=["mask"])
        return zoomed.fadein(transition_duration).fadeout(transition_duration)

    clips = [make_clip(p, clip_dur, cinematic) for p in image_paths]

    # ── Concatenate with crossfade overlap ────────────────────────────────────
    # padding=-transition_duration makes clips overlap so fades blend cleanly.
    final_video = concatenate_videoclips(
        clips,
        method="compose",
        padding=-transition_duration,
    )

    # Trim to exact target duration (rounding may add a stray frame)
    final_video = final_video.subclip(0, min(final_video.duration, video_duration))

    # ── Loop or trim audio to match final video duration ──────────────────────
    actual_duration = final_video.duration

    if audio_duration < actual_duration:
        loops_needed = int(actual_duration // audio_duration) + 1
        audio_clip   = concatenate_audioclips([audio_clip] * loops_needed)

    audio_clip  = audio_clip.subclip(0, actual_duration)
    final_video = final_video.set_audio(audio_clip)

    # ── Export ────────────────────────────────────────────────────────────────
    final_video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    print(f"✅ Video saved to: {output_path}")
    print(f"   Images   : {n}")
    print(f"   Duration : {actual_duration:.1f}s  ({clip_dur:.1f}s per image)")
    print(f"   Cinematic: {cinematic}  |  Transition: {transition_duration}s crossfade")


# ── Quick-start (run directly or in Google Colab) ─────────────────────────────
if __name__ == "__main__":
    # Install dependencies if needed:
    #   !pip install moviepy pillow

    create_video_from_image_and_audio(
        image_paths=[
            "/content/drive/MyDrive/editor/img.png",
            "/content/drive/MyDrive/editor/img2.png",
            "/content/drive/MyDrive/editor/img3.png",
            "/content/drive/MyDrive/editor/img4.png",
            "/content/drive/MyDrive/editor/img5.png",
        ],
        audio_path="/content/drive/MyDrive/editor/bairan.mp3",
        output_path="slideshow.mp4",
        cinematic=True,
        transition_duration=0.6,
    )