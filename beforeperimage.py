def create_video_from_image_and_audio(
    image_paths,
    audio_path,
    output_path,
    duration=10,
    cinematic=True,
    transition_duration=0.6,
    transition_type="fade",
):
    """
    Creates an MP4 slideshow video from multiple images with background audio
    and selectable transition effects between clips.

    Args:
        image_paths         (list):  List of image file paths.
        audio_path           (str):  Path to the background audio file.
        output_path          (str):  Path where the final MP4 will be saved.
        duration             (int):  Fallback total duration (seconds) when
                                     audio duration cannot be determined. (default: 10)
        cinematic           (bool):  Apply Ken Burns zoom effect on each image. (default: True)
        transition_duration (float): Transition length in seconds between images. (default: 0.6)
        transition_type      (str):  Transition style to use between clips.
                                     Options: "fade" | "slide_left" | "slide_right" |
                                              "slide_up" | "slide_down" | "zoom" |
                                              "blur" | "random"
                                     (default: "fade")
    """
    import os
    import random as _random
    import numpy as np
    from moviepy.editor import (
        ImageClip,
        AudioFileClip,
        CompositeVideoClip,
        concatenate_audioclips,
        concatenate_videoclips,
    )

    VALID_TRANSITIONS = {"fade", "slide_left", "slide_right", "slide_up", "slide_down", "zoom", "blur", "random"}

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not image_paths:
        raise ValueError("image_paths must be a non-empty list.")

    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if transition_type not in VALID_TRANSITIONS:
        raise ValueError(f"transition_type must be one of {VALID_TRANSITIONS}. Got: '{transition_type}'")

    # ── Load audio & resolve total video duration ─────────────────────────────
    audio_clip     = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    video_duration = max(audio_duration, duration)

    # ── Per-image clip duration (overlap-aware) ───────────────────────────────
    n = len(image_paths)
    # Solve: clip_dur + (n-1)*(clip_dur - transition_duration) = video_duration
    clip_dur = (video_duration + (n - 1) * transition_duration) / n
    clip_dur = max(clip_dur, transition_duration + 0.5)   # safety floor

    # ── Frame dtype safety helper ─────────────────────────────────────────────
    def to_uint8(frame):
        """Ensure frame is uint8 RGB — MoviePy may pass float64 after compositing."""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        # Drop alpha channel if present (RGBA → RGB)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        return frame

    # ── Ken Burns zoom helper ─────────────────────────────────────────────────
    def apply_ken_burns(base_clip, clip_duration):
        """Overlay a slow 1.0x → 1.08x zoom (Ken Burns effect) on a clip."""
        w, h = base_clip.size
        zoom_start, zoom_end = 1.0, 1.08

        def zoom_frame(get_frame, t):
            frame  = to_uint8(get_frame(t))
            factor = zoom_start + (zoom_end - zoom_start) * (t / clip_duration)
            new_w  = int(w * factor)
            new_h  = int(h * factor)

            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            arr = np.array(img, dtype=np.uint8)

            x0 = (new_w - w) // 2
            y0 = (new_h - h) // 2
            return arr[y0:y0 + h, x0:x0 + w]

        return base_clip.fl(zoom_frame, apply_to=["mask"])

    # ── Transition builders ───────────────────────────────────────────────────
    def make_fade_transition(clip_a, clip_b, tdur):
        """Classic crossfade: A fades out while B fades in."""
        a = clip_a.fadeout(tdur)
        b = clip_b.fadein(tdur)
        return a, b

    def make_slide_transition(clip_a, clip_b, tdur, direction):
        """
        Slide transition: B enters from <direction>, A stays put then is
        replaced once B fully covers the frame.

        direction: "left" | "right" | "up" | "down"
        """
        w, h = clip_a.size

        # Vector by which B travels across the frame
        vectors = {
            "left":  ( w,  0),   # B starts at right edge, moves left
            "right": (-w,  0),   # B starts at left edge,  moves right
            "up":    ( 0,  h),   # B starts at bottom,     moves up
            "down":  ( 0, -h),   # B starts at top,        moves down
        }
        dx, dy = vectors[direction]

        def slide_pos(t):
            progress = min(t / tdur, 1.0)
            # Ease in-out (smoothstep)
            ease = progress * progress * (3 - 2 * progress)
            return (int(dx * (1 - ease)), int(dy * (1 - ease)))

        # B slides in on top of A; A simply stays
        b_moving = clip_b.set_position(slide_pos)
        composite = CompositeVideoClip([clip_a, b_moving], size=clip_a.size)
        return composite

    def make_zoom_transition(clip_a, clip_b, tdur):
        """
        Zoom transition: only the TAIL of A scales up and only the HEAD of B
        scales down. Both are crossfaded so they blend at the overlap window.
        """
        w, h   = clip_a.size
        dur_a  = clip_a.duration

        def zoom_out_frame(get_frame, t):
            frame = to_uint8(get_frame(t))
            # Only zoom during the last `tdur` seconds of clip A
            time_into_tail = t - (dur_a - tdur)
            if time_into_tail <= 0:
                return frame
            progress = min(time_into_tail / tdur, 1.0)   # 0.0 → 1.0
            factor   = 1.0 + 0.12 * progress             # 1.0 → 1.12x scale-up

            new_w = int(w * factor)
            new_h = int(h * factor)

            from PIL import Image
            img = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
            arr = np.array(img, dtype=np.uint8)

            x0 = (new_w - w) // 2
            y0 = (new_h - h) // 2
            return arr[y0:y0 + h, x0:x0 + w]

        def zoom_in_frame(get_frame, t):
            frame = to_uint8(get_frame(t))
            # Only zoom during the first `tdur` seconds of clip B
            if t >= tdur:
                return frame
            progress = min(t / tdur, 1.0)                # 0.0 → 1.0
            factor   = 1.12 - 0.12 * progress            # 1.12x → 1.0 scale-down

            new_w = int(w * factor)
            new_h = int(h * factor)

            from PIL import Image
            img = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
            arr = np.array(img, dtype=np.uint8)

            x0 = (new_w - w) // 2
            y0 = (new_h - h) // 2
            return arr[y0:y0 + h, x0:x0 + w]

        a_zoomed = clip_a.fl(zoom_out_frame).fadeout(tdur)
        b_zoomed = clip_b.fl(zoom_in_frame).fadein(tdur)
        return a_zoomed, b_zoomed

    def make_blur_transition(clip_a, clip_b, tdur):
        """
        Blur transition: only the TAIL of clip A blurs out, and only the
        HEAD of clip B blurs in. The rest of each clip is untouched.
        """
        from PIL import ImageFilter, Image

        dur_a = clip_a.duration
        dur_b = clip_b.duration

        def blur_frame_a(get_frame, t):
            frame = to_uint8(get_frame(t))
            # Only blur during the last `tdur` seconds of clip A
            time_into_tail = t - (dur_a - tdur)
            if time_into_tail <= 0:
                return frame                            # before transition window
            progress = min(time_into_tail / tdur, 1.0) # 0.0 → 1.0
            radius   = int(progress * 18)               # 0 → 18 px
            if radius == 0:
                return frame
            img = Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius))
            return np.array(img, dtype=np.uint8)

        def blur_frame_b(get_frame, t):
            frame = to_uint8(get_frame(t))
            # Only de-blur during the first `tdur` seconds of clip B
            if t >= tdur:
                return frame                            # after transition window
            progress = min(t / tdur, 1.0)              # 0.0 → 1.0
            radius   = int((1.0 - progress) * 18)      # 18 → 0 px
            if radius == 0:
                return frame
            img = Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius))
            return np.array(img, dtype=np.uint8)

        a_blurred = clip_a.fl(blur_frame_a).fadeout(tdur)
        b_blurred = clip_b.fl(blur_frame_b).fadein(tdur)
        return a_blurred, b_blurred

    # ── Choose transitions per pair ───────────────────────────────────────────
    NON_RANDOM = list(VALID_TRANSITIONS - {"random"})

    def pick_transition():
        if transition_type == "random":
            return _random.choice(NON_RANDOM)
        return transition_type

    # ── Build individual image clips ──────────────────────────────────────────
    def make_base_clip(img_path, clip_duration):
        base = ImageClip(img_path, duration=clip_duration)
        if cinematic:
            base = apply_ken_burns(base, clip_duration)
        return base

    raw_clips = [make_base_clip(p, clip_dur) for p in image_paths]

    # ── Assemble with selected transitions ────────────────────────────────────
    # Strategy:
    #  • "fade", "zoom", "blur" → use MoviePy's native overlap (padding=-tdur)
    #    after pre-applying fade-in/out on each clip.
    #  • "slide_*" → build composite clips for every adjacent pair, then
    #    concatenate the non-overlapping segments.

    slide_directions = {
        "slide_left":  "left",
        "slide_right": "right",
        "slide_up":    "up",
        "slide_down":  "down",
    }

    def build_clip_sequence():
        """
        Builds the final clip list. Each adjacent pair gets exactly one
        transition, chosen once per pair. Returns (clips, has_any_slide)
        so the caller can set padding correctly.
        """
        processed    = []
        has_slide    = False

        for i, clip in enumerate(raw_clips):
            if i == 0:
                # First clip: fadein only — fadeout will be applied when pairing with clip 1
                processed.append(clip.fadein(transition_duration))
                continue

            chosen = pick_transition()   # one pick per pair

            if chosen in slide_directions:
                has_slide = True
                direction = slide_directions[chosen]
                tdur      = transition_duration
                prev_clip = processed[-1]
                p_dur     = prev_clip.duration

                tail         = prev_clip.subclip(max(0, p_dur - tdur), p_dur)
                head         = clip.subclip(0, min(tdur, clip.duration))
                composite    = make_slide_transition(tail, head, tdur, direction)
                composite    = composite.set_duration(tdur)
                trimmed_prev = prev_clip.subclip(0, max(0, p_dur - tdur))

                processed[-1] = trimmed_prev
                processed.append(composite)

                body = clip.subclip(min(tdur, clip.duration))
                # Last clip needs a fadeout; otherwise just append body
                if i == len(raw_clips) - 1:
                    processed.append(body.fadeout(transition_duration))
                else:
                    processed.append(body)

            elif chosen == "zoom":
                a_out, b_in  = make_zoom_transition(processed[-1], clip, transition_duration)
                processed[-1] = a_out
                if i == len(raw_clips) - 1:
                    processed.append(b_in.fadeout(transition_duration))
                else:
                    processed.append(b_in)

            elif chosen == "blur":
                a_out, b_in  = make_blur_transition(processed[-1], clip, transition_duration)
                processed[-1] = a_out
                if i == len(raw_clips) - 1:
                    processed.append(b_in.fadeout(transition_duration))
                else:
                    processed.append(b_in)

            else:  # "fade"
                a_out, b_in  = make_fade_transition(processed[-1], clip, transition_duration)
                processed[-1] = a_out
                if i == len(raw_clips) - 1:
                    processed.append(b_in.fadeout(transition_duration))
                else:
                    processed.append(b_in)

        return processed, has_slide

    processed_clips, has_slide = build_clip_sequence()

    # ── Concatenate ───────────────────────────────────────────────────────────
    # Slide composites are pre-built full-duration bricks — no overlap needed.
    # Fade / zoom / blur use fadein+fadeout baked onto clips, so they need
    # padding=-transition_duration to overlap and blend correctly.
    if has_slide and transition_type in slide_directions:
        # Pure slide mode: no padding
        final_video = concatenate_videoclips(processed_clips, method="compose")
    else:
        # Fade / zoom / blur (or random which may mix types):
        # use overlap for alpha-blend transitions; slide segments are already baked.
        final_video = concatenate_videoclips(
            processed_clips,
            method="compose",
            padding=-transition_duration,
        )

    # Trim to exact target duration
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
    print(f"   Images     : {n}")
    print(f"   Duration   : {actual_duration:.1f}s  ({clip_dur:.1f}s per image)")
    print(f"   Cinematic  : {cinematic}")
    print(f"   Transition : {transition_type}  ({transition_duration}s each)")


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
        transition_type="slide_left",       # swap to any supported type or "random"
    )