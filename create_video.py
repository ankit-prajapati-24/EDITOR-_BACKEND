def create_video_from_image_and_audio(
    image_paths,
    audio_path,
    output_path,
    duration=10,
    cinematic=True,
    transition_duration=0.6,
    transition_type="fade",
):
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

    VALID_TRANSITIONS = {
        "fade", "slide_left", "slide_right", "slide_up", "slide_down",
        "zoom", "blur", "random",
    }
    NON_RANDOM_TRANSITIONS = list(VALID_TRANSITIONS - {"random"})

    slide_directions = {
        "slide_left":  "left",
        "slide_right": "right",
        "slide_up":    "up",
        "slide_down":  "down",
    }

    if not image_paths:
        raise ValueError("image_paths must be a non-empty list.")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if transition_type not in VALID_TRANSITIONS:
        raise ValueError(
            f"transition_type must be one of {VALID_TRANSITIONS}. Got: '{transition_type}'"
        )

    def normalize_entry(entry, idx):
        if isinstance(entry, str):
            path      = entry
            img_dur   = None
            img_trans = None
        elif isinstance(entry, dict):
            if "path" not in entry:
                raise ValueError(f"Entry at index {idx} is missing required key 'path'.")
            path      = entry["path"]
            img_dur   = entry.get("duration", None)
            img_trans = entry.get("transition", None)
            if img_dur is not None and (not isinstance(img_dur, (int, float)) or img_dur <= 0):
                raise ValueError(
                    f"Entry at index {idx}: 'duration' must be a positive number. Got: {img_dur}"
                )
            if img_trans is not None and img_trans not in VALID_TRANSITIONS:
                raise ValueError(
                    f"Entry at index {idx}: 'transition' must be one of {VALID_TRANSITIONS}. "
                    f"Got: '{img_trans}'"
                )
        else:
            raise TypeError(
                f"Entry at index {idx} must be a string path or a dict. Got: {type(entry)}"
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        return {"path": path, "duration": img_dur, "transition": img_trans}

    configs = [normalize_entry(e, i) for i, e in enumerate(image_paths)]
    n       = len(configs)

    audio_clip     = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    video_duration = max(audio_duration, duration)

    fixed_total = sum(c["duration"] for c in configs if c["duration"] is not None)
    auto_count  = sum(1 for c in configs if c["duration"] is None)

    if auto_count > 0:
        total_overlap  = (n - 1) * transition_duration
        remaining_time = max(video_duration - fixed_total + total_overlap, 0)
        auto_clip_dur  = max(remaining_time / auto_count, transition_duration + 0.5)
    else:
        auto_clip_dur  = transition_duration + 0.5

    for c in configs:
        if c["duration"] is None:
            c["duration"] = auto_clip_dur

    # ── Frame dtype safety ────────────────────────────────────────────────────
    def to_uint8(frame):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        return frame

    # ── Ken Burns ─────────────────────────────────────────────────────────────
    def apply_ken_burns(base_clip, clip_duration):
        w, h = base_clip.size
        zoom_start, zoom_end = 1.0, 1.08

        def zoom_frame(get_frame, t):
            frame        = to_uint8(get_frame(t))
            factor       = zoom_start + (zoom_end - zoom_start) * (t / clip_duration)
            new_w, new_h = int(w * factor), int(h * factor)
            from PIL import Image
            arr = np.array(
                Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS),
                dtype=np.uint8,
            )
            x0, y0 = (new_w - w) // 2, (new_h - h) // 2
            return arr[y0:y0 + h, x0:x0 + w]

        return base_clip.fl(zoom_frame, apply_to=["mask"])

    # ── Transition builders ───────────────────────────────────────────────────
    def make_slide_transition(clip_a, clip_b, tdur, direction):
        w, h = clip_a.size

        enter_offset = {
            "left":  ( w,  0),
            "right": (-w,  0),
            "up":    ( 0,  h),
            "down":  ( 0, -h),
        }
        ex, ey = enter_offset[direction]
        ox, oy = -ex, -ey

        def pos_b(t):
            p    = min(t / tdur, 1.0)
            ease = p * p * (3 - 2 * p)
            return (int(ex * (1 - ease)), int(ey * (1 - ease)))

        def pos_a(t):
            p    = min(t / tdur, 1.0)
            ease = p * p * (3 - 2 * p)
            return (int(ox * ease), int(oy * ease))

        moving_a = clip_a.set_position(pos_a)
        moving_b = clip_b.set_position(pos_b)
        return CompositeVideoClip([moving_a, moving_b], size=(w, h))

    def make_zoom_transition(clip_a, clip_b, tdur):
        """clip_a and clip_b are already tail/head slices of duration=tdur"""
        w, h = clip_a.size

        def zoom_out_frame(get_frame, t):
            frame        = to_uint8(get_frame(t))
            factor       = 1.0 + 0.12 * min(t / tdur, 1.0)
            new_w, new_h = int(w * factor), int(h * factor)
            from PIL import Image
            arr = np.array(
                Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS),
                dtype=np.uint8,
            )
            return arr[(new_h - h) // 2:(new_h + h) // 2,
                       (new_w - w) // 2:(new_w + w) // 2]

        def zoom_in_frame(get_frame, t):
            frame        = to_uint8(get_frame(t))
            factor       = 1.12 - 0.12 * min(t / tdur, 1.0)
            new_w, new_h = int(w * factor), int(h * factor)
            from PIL import Image
            arr = np.array(
                Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS),
                dtype=np.uint8,
            )
            return arr[(new_h - h) // 2:(new_h + h) // 2,
                       (new_w - w) // 2:(new_w + w) // 2]

        return clip_a.fl(zoom_out_frame), clip_b.fl(zoom_in_frame)

    def make_blur_transition(clip_a, clip_b, tdur):
        """clip_a and clip_b are already tail/head slices of duration=tdur"""
        from PIL import ImageFilter, Image

        def blur_out(get_frame, t):
            frame  = to_uint8(get_frame(t))
            radius = int(min(t / tdur, 1.0) * 18)
            if radius == 0:
                return frame
            return np.array(
                Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius)),
                dtype=np.uint8,
            )

        def blur_in(get_frame, t):
            frame  = to_uint8(get_frame(t))
            radius = int((1.0 - min(t / tdur, 1.0)) * 18)
            if radius == 0:
                return frame
            return np.array(
                Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius)),
                dtype=np.uint8,
            )

        return clip_a.fl(blur_out), clip_b.fl(blur_in)

    # ── Transition resolver ───────────────────────────────────────────────────
    def resolve_transition(cfg):
        t = cfg["transition"] if cfg["transition"] is not None else transition_type
        if t == "random":
            t = _random.choice(NON_RANDOM_TRANSITIONS)
        return t

    # ── Build base clips ──────────────────────────────────────────────────────
    def make_base_clip(cfg):
        clip = ImageClip(cfg["path"], duration=cfg["duration"])
        if cinematic:
            clip = apply_ken_burns(clip, cfg["duration"])
        return clip

    raw_clips = [make_base_clip(c) for c in configs]

    # ── Assemble sequence ─────────────────────────────────────────────────────
    def build_clip_sequence():
        processed             = []
        transition_types_used = []

        for i, clip in enumerate(raw_clips):
            is_last = (i == n - 1)

            if i == 0:
                next_chosen = resolve_transition(configs[0]) if n > 1 else "fade"
                if next_chosen in slide_directions:
                    processed.append(clip)
                else:
                    processed.append(clip.fadein(transition_duration))
                continue

            chosen = resolve_transition(configs[i - 1])
            transition_types_used.append(chosen)
            tdur = transition_duration

            prev_clip    = processed[-1]
            p_dur        = prev_clip.duration
            tail         = prev_clip.subclip(max(0, p_dur - tdur), p_dur)
            trimmed_prev = prev_clip.subclip(0, max(0, p_dur - tdur))
            head         = clip.subclip(0, min(tdur, clip.duration))
            body         = clip.subclip(min(tdur, clip.duration))

            if chosen in slide_directions:
                # ── Slide ──────────────────────────────────────────────────
                composite = make_slide_transition(
                    tail, head, tdur, slide_directions[chosen]
                )
                composite = composite.set_duration(tdur)

                processed[-1] = trimmed_prev
                processed.append(composite)
                processed.append(body.fadeout(tdur) if is_last else body)

            elif chosen == "zoom":
                # ── Zoom ───────────────────────────────────────────────────
                zoom_tail, zoom_head = make_zoom_transition(tail, head, tdur)
                composite = CompositeVideoClip(
                    [zoom_tail.fadeout(tdur), zoom_head.fadein(tdur)],
                    size=tail.size,
                ).set_duration(tdur)

                processed[-1] = trimmed_prev
                processed.append(composite)
                processed.append(body.fadeout(tdur) if is_last else body)

            elif chosen == "blur":
                # ── Blur ───────────────────────────────────────────────────
                blur_tail, blur_head = make_blur_transition(tail, head, tdur)
                composite = CompositeVideoClip(
                    [blur_tail.fadeout(tdur), blur_head.fadein(tdur)],
                    size=tail.size,
                ).set_duration(tdur)

                processed[-1] = trimmed_prev
                processed.append(composite)
                processed.append(body.fadeout(tdur) if is_last else body)

            else:
                # ── Fade ───────────────────────────────────────────────────
                fade_composite = CompositeVideoClip(
                    [tail.fadeout(tdur), head.fadein(tdur)],
                    size=tail.size,
                ).set_duration(tdur)

                processed[-1] = trimmed_prev
                processed.append(fade_composite)
                processed.append(body.fadeout(tdur) if is_last else body)

        # Fadeout on last clip if last transition was not a slide
        last_chosen = resolve_transition(configs[-2]) if n > 1 else None
        if last_chosen not in slide_directions and processed:
            processed[-1] = processed[-1].fadeout(transition_duration)

        return processed

    processed_clips = build_clip_sequence()

    # ── Concatenate — no padding, all transitions are pre-baked ──────────────
    final_video = concatenate_videoclips(processed_clips, method="compose")
    final_video = final_video.subclip(0, min(final_video.duration, video_duration))

    # ── Loop or trim audio ────────────────────────────────────────────────────
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

    durations_summary   = [f"{c['duration']:.1f}s" for c in configs]
    transitions_summary = [
        (configs[i]["transition"] or transition_type) for i in range(n - 1)
    ]

    print(f"✅ Video saved to: {output_path}")
    print(f"   Images      : {n}")
    print(f"   Duration    : {actual_duration:.1f}s total")
    print(f"   Per-image   : {', '.join(durations_summary)}")
    print(f"   Transitions : {', '.join(transitions_summary)}")
    print(f"   Cinematic   : {cinematic}  |  Transition dur: {transition_duration}s")


# ── Quick-start examples ──────────────────────────────────────────────────────
if __name__ == "__main__":

    BASE = "/content/drive/MyDrive/editor"

    # # ── Example 1: Simple string list ────────────────────────────────────────
    # create_video_from_image_and_audio(
    #     image_paths=[
    #         f"{BASE}/img.png",
    #         f"{BASE}/img2.png",
    #         f"{BASE}/img3.png",
    #         f"{BASE}/img4.png",
    #         f"{BASE}/img5.png",
    #     ],
    #     audio_path=f"{BASE}/bairan.mp3",
    #     output_path="slideshow_simple.mp4",
    #     cinematic=True,
    #     transition_duration=0.6,
    #     transition_type="fade",
    # )

    # ── Example 2: Per-image config ──────────────────────────────────────────
    # create_video_from_image_and_audio(
    #     image_paths=[
    #         {"path": f"{BASE}/img.png",  "duration": 3, "transition": "zoom"},
    #         {"path": f"{BASE}/img2.png", "duration": 3, "transition": "blur"},
    #         {"path": f"{BASE}/img3.png", "duration": 3, "transition": "slide_down"},
    #         {"path": f"{BASE}/img4.png", "duration": 3, "transition": "slide_left"},
    #         {"path": f"{BASE}/img5.png", "duration": 3, "transition": "slide_up"},
    #     ],
    #     audio_path=f"{BASE}/bairan.mp3",
    #     output_path=f"{BASE}/up_down_left_right_zoom.mp4",
    #     cinematic=True,
    #     transition_duration=0.6,
    # )

    # ── Example 3: Mixed ─────────────────────────────────────────────────────
    create_video_from_image_and_audio(
        image_paths=[
            {"path": f"{BASE}/img.png",  "duration": 0.5, "transition": "slide_right"},
            {"path": f"{BASE}/img2.png"},
             {"path": f"{BASE}/img2.png"},
             {"path": f"{BASE}/img4.png"},
             {"path": f"{BASE}/img2.png"},
             {"path": f"{BASE}/img5.png"},
             {"path": f"{BASE}/img2.png"},
            {"path": f"{BASE}/img3.png", "transition": "blur"},
        ],
        audio_path=f"{BASE}/bairan.mp3",
        output_path="slideshow_mixed.mp4",
        cinematic=True,
        transition_duration=0.6,
        transition_type="random",
    )