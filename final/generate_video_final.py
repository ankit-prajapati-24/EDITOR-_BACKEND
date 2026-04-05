def create_video_from_image_and_audio(
    image_paths,
    audio_path = "C:\\editor_project\\assets\\shisha.mp3",
    output_path = "C:\\editor_project\\assets\\final_output.mp4",
    duration=1.2,
    cinematic=True,
    transition_duration=0.6,
    transition_type="fade",
    motion_type="zoom_in",
    motion_speed=1.0,
    layout_mode="blur_bg",
    frame_size=(1080, 1920),
    log_progress=True,
    progress_callback=None,
):
    import os
    import time
    import random as _random
    import numpy as np
    from moviepy import (
        ImageClip,
        AudioFileClip,
        CompositeVideoClip,
        concatenate_audioclips,
        concatenate_videoclips,
    )
    from moviepy.video import fx as vfx
    from proglog import ProgressBarLogger

    BASE = "C:\\editor_project\\assets\\"
    # Traverse images and prepend base path for relative paths
    def _with_base(p):
        return p if os.path.isabs(p) else os.path.join(BASE, p)

    fixed_paths = []
    for entry in image_paths:
        if isinstance(entry, dict):
            new_entry = dict(entry)
            new_entry["path"] = _with_base(str(new_entry["path"]))
            fixed_paths.append(new_entry)
        else:
            fixed_paths.append(_with_base(str(entry)))
    image_paths = fixed_paths

    VALID_TRANSITIONS = {
        "fade", "slide_left", "slide_right", "slide_up", "slide_down",
        "zoom", "blur", "random",
    }
    NON_RANDOM_TRANSITIONS = list(VALID_TRANSITIONS - {"random"})
    VALID_MOTIONS   = {"zoom_in", "zoom_out", "move_left", "move_right"}
    VALID_LAYOUTS   = {"fit", "fill", "blur_bg"}

    slide_directions = {
        "slide_left":  "left",
        "slide_right": "right",
        "slide_up":    "up",
        "slide_down":  "down",
    }

    FW, FH = frame_size   # frame width, frame height

    if not image_paths:
        raise ValueError("image_paths must be a non-empty list.")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if transition_type not in VALID_TRANSITIONS:
        raise ValueError(f"transition_type must be one of {VALID_TRANSITIONS}. Got: '{transition_type}'")
    if motion_type not in VALID_MOTIONS:
        raise ValueError(f"motion_type must be one of {VALID_MOTIONS}. Got: '{motion_type}'")
    if not isinstance(motion_speed, (int, float)) or motion_speed <= 0:
        raise ValueError(f"motion_speed must be a positive number. Got: {motion_speed}")
    if layout_mode not in VALID_LAYOUTS:
        raise ValueError(f"layout_mode must be one of {VALID_LAYOUTS}. Got: '{layout_mode}'")

    def normalize_entry(entry, idx):
        if isinstance(entry, str):
            return {"path": entry, "duration": None, "transition": None,
                    "motion": None, "motion_speed": None}
        elif isinstance(entry, dict):
            if "path" not in entry:
                raise ValueError(f"Entry at index {idx} is missing required key 'path'.")
            path      = entry["path"]
            img_dur   = entry.get("duration", None)
            img_trans = entry.get("transition", None)
            img_mot   = entry.get("motion", None)
            img_speed = entry.get("motion_speed", None)
            if img_dur is not None and (not isinstance(img_dur, (int, float)) or img_dur <= 0):
                raise ValueError(f"Entry at index {idx}: 'duration' must be a positive number. Got: {img_dur}")
            if img_trans is not None and img_trans not in VALID_TRANSITIONS:
                raise ValueError(f"Entry at index {idx}: 'transition' must be one of {VALID_TRANSITIONS}. Got: '{img_trans}'")
            if img_mot is not None and img_mot not in VALID_MOTIONS:
                raise ValueError(f"Entry at index {idx}: 'motion' must be one of {VALID_MOTIONS}. Got: '{img_mot}'")
            if img_speed is not None and (not isinstance(img_speed, (int, float)) or img_speed <= 0):
                raise ValueError(f"Entry at index {idx}: 'motion_speed' must be a positive number. Got: {img_speed}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            return {"path": path, "duration": img_dur, "transition": img_trans,
                    "motion": img_mot, "motion_speed": img_speed}
        else:
            raise TypeError(f"Entry at index {idx} must be a string or dict. Got: {type(entry)}")

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

    # ── Layout composer ───────────────────────────────────────────────────────
    def compose_frame(image_path, mode):
        """
        Returns a numpy array (FH, FW, 3) uint8 representing the composed frame.
        mode: "fit" | "fill" | "blur_bg"
        """
        from PIL import Image as PILImage, ImageFilter

        img     = PILImage.open(image_path).convert("RGB")
        iw, ih  = img.size
        i_ratio = iw / ih
        f_ratio = FW / FH

        if mode == "fit":
            # Scale to fit entirely inside frame, letterbox/pillarbox with black
            if i_ratio > f_ratio:
                new_w = FW
                new_h = int(FW / i_ratio)
            else:
                new_h = FH
                new_w = int(FH * i_ratio)
            new_w = max(new_w, 1)
            new_h = max(new_h, 1)
            resized = img.resize((new_w, new_h), PILImage.LANCZOS)
            canvas  = PILImage.new("RGB", (FW, FH), (0, 0, 0))
            x0      = (FW - new_w) // 2
            y0      = (FH - new_h) // 2
            canvas.paste(resized, (x0, y0))
            return np.array(canvas, dtype=np.uint8)

        elif mode == "fill":
            # Scale to fill frame completely, crop excess
            if i_ratio > f_ratio:
                new_h = FH
                new_w = int(FH * i_ratio)
            else:
                new_w = FW
                new_h = int(FW / i_ratio)
            new_w   = max(new_w, 1)
            new_h   = max(new_h, 1)
            resized = img.resize((new_w, new_h), PILImage.LANCZOS)
            x0      = (new_w - FW) // 2
            y0      = (new_h - FH) // 2
            cropped = resized.crop((x0, y0, x0 + FW, y0 + FH))
            return np.array(cropped, dtype=np.uint8)

        elif mode == "blur_bg":
            # Background: fill + heavy blur
            if i_ratio > f_ratio:
                bg_h = FH
                bg_w = int(FH * i_ratio)
            else:
                bg_w = FW
                bg_h = int(FW / i_ratio)
            bg_w    = max(bg_w, 1)
            bg_h    = max(bg_h, 1)
            bg_img  = img.resize((bg_w, bg_h), PILImage.LANCZOS)
            x0      = (bg_w - FW) // 2
            y0      = (bg_h - FH) // 2
            bg_crop = bg_img.crop((x0, y0, x0 + FW, y0 + FH))
            bg_blur = bg_crop.filter(ImageFilter.GaussianBlur(radius=30))

            # Foreground: fit inside frame
            if i_ratio > f_ratio:
                fg_w = FW
                fg_h = int(FW / i_ratio)
            else:
                fg_h = FH
                fg_w = int(FH * i_ratio)
            fg_w    = max(fg_w, 1)
            fg_h    = max(fg_h, 1)
            fg_img  = img.resize((fg_w, fg_h), PILImage.LANCZOS)

            canvas  = bg_blur.copy()
            px      = (FW - fg_w) // 2
            py      = (FH - fg_h) // 2
            canvas.paste(fg_img, (px, py))
            return np.array(canvas, dtype=np.uint8)

        else:
            raise ValueError(f"Unknown layout_mode: {mode}")

    # ── Motion effects ────────────────────────────────────────────────────────
    def apply_motion(base_clip, clip_duration, motion, speed):
        from PIL import Image as PILImage

        w, h       = base_clip.size   # always FW, FH after layout compose
        BASE_ZOOM  = 0.08
        BASE_PAN   = 0.08
        zoom_delta = min(BASE_ZOOM * speed, 0.25)
        pan_delta  = min(BASE_PAN  * speed, 0.25)

        def progress(t):
            p = min(t / clip_duration, 1.0)
            return p * p * (3 - 2 * p)

        def _resize_crop_center(frame, factor):
            new_w, new_h = int(w * factor), int(h * factor)
            arr = np.array(
                PILImage.fromarray(frame).resize((new_w, new_h), PILImage.LANCZOS),
                dtype=np.uint8,
            )
            x0 = (new_w - w) // 2
            y0 = (new_h - h) // 2
            return arr[y0:y0 + h, x0:x0 + w]

        if motion == "zoom_in":
            def effect(get_frame, t):
                return _resize_crop_center(to_uint8(get_frame(t)), 1.0 + zoom_delta * progress(t))

        elif motion == "zoom_out":
            def effect(get_frame, t):
                return _resize_crop_center(to_uint8(get_frame(t)), 1.0 + zoom_delta * (1.0 - progress(t)))

        elif motion == "move_left":
            factor = 1.0 + pan_delta
            def effect(get_frame, t):
                frame        = to_uint8(get_frame(t))
                new_w, new_h = int(w * factor), int(h * factor)
                arr  = np.array(PILImage.fromarray(frame).resize((new_w, new_h), PILImage.LANCZOS), dtype=np.uint8)
                x0   = int((new_w - w) * (1.0 - progress(t)))
                y0   = (new_h - h) // 2
                return arr[y0:y0 + h, x0:x0 + w]

        elif motion == "move_right":
            factor = 1.0 + pan_delta
            def effect(get_frame, t):
                frame        = to_uint8(get_frame(t))
                new_w, new_h = int(w * factor), int(h * factor)
                arr  = np.array(PILImage.fromarray(frame).resize((new_w, new_h), PILImage.LANCZOS), dtype=np.uint8)
                x0   = int((new_w - w) * progress(t))
                y0   = (new_h - h) // 2
                return arr[y0:y0 + h, x0:x0 + w]

        else:
            return base_clip

        return base_clip.transform(effect, apply_to=["mask"])

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

        return CompositeVideoClip([clip_a.with_position(pos_a), clip_b.with_position(pos_b)], size=(w, h))

    def make_zoom_transition(clip_a, clip_b, tdur):
        w, h = clip_a.size

        def zoom_out_frame(get_frame, t):
            frame        = to_uint8(get_frame(t))
            factor       = 1.0 + 0.12 * min(t / tdur, 1.0)
            new_w, new_h = int(w * factor), int(h * factor)
            from PIL import Image
            arr = np.array(Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8)
            return arr[(new_h - h) // 2:(new_h + h) // 2, (new_w - w) // 2:(new_w + w) // 2]

        def zoom_in_frame(get_frame, t):
            frame        = to_uint8(get_frame(t))
            factor       = 1.12 - 0.12 * min(t / tdur, 1.0)
            new_w, new_h = int(w * factor), int(h * factor)
            from PIL import Image
            arr = np.array(Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8)
            return arr[(new_h - h) // 2:(new_h + h) // 2, (new_w - w) // 2:(new_w + w) // 2]

        return clip_a.transform(zoom_out_frame), clip_b.transform(zoom_in_frame)

    def make_blur_transition(clip_a, clip_b, tdur):
        from PIL import ImageFilter, Image

        def blur_out(get_frame, t):
            frame  = to_uint8(get_frame(t))
            radius = int(min(t / tdur, 1.0) * 18)
            if radius == 0:
                return frame
            return np.array(Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)

        def blur_in(get_frame, t):
            frame  = to_uint8(get_frame(t))
            radius = int((1.0 - min(t / tdur, 1.0)) * 18)
            if radius == 0:
                return frame
            return np.array(Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)

        return clip_a.transform(blur_out), clip_b.transform(blur_in)

    # ── Resolvers ─────────────────────────────────────────────────────────────
    def resolve_transition(cfg):
        t = cfg["transition"] if cfg["transition"] is not None else transition_type
        if t == "random":
            t = _random.choice(NON_RANDOM_TRANSITIONS)
        return t

    def resolve_motion(cfg):
        return cfg["motion"] if cfg["motion"] is not None else motion_type

    def resolve_speed(cfg):
        return cfg["motion_speed"] if cfg["motion_speed"] is not None else motion_speed

    # ── Build base clips ──────────────────────────────────────────────────────
    def make_base_clip(cfg):
        # 1. Compose layout → uniform FW×FH numpy frame
        composed = compose_frame(cfg["path"], layout_mode)

        # 2. Build ImageClip from composed frame (already correct size)
        clip = ImageClip(composed, duration=cfg["duration"])

        # 3. Apply motion
        clip = apply_motion(clip, cfg["duration"], resolve_motion(cfg), resolve_speed(cfg))

        return clip

    t0 = time.perf_counter()
    def _log(msg):
        if log_progress:
            elapsed = time.perf_counter() - t0
            print(f"[{elapsed:6.1f}s] {msg}")
        if progress_callback:
            progress_callback(msg)

    _log(f"Starting video build: {n} images, layout={layout_mode}, motion={motion_type}")
    raw_clips = [make_base_clip(c) for c in configs]
    _log("Base clips created (25%)")

    # ── Assemble sequence ─────────────────────────────────────────────────────
    def build_clip_sequence():
        processed             = []
        transition_types_used = []

        for i, clip in enumerate(raw_clips):
            is_last = (i == n - 1)

            if i == 0:
                next_chosen = resolve_transition(configs[0]) if n > 1 else "fade"
                processed.append(
                    clip
                    if next_chosen in slide_directions
                    else clip.with_effects([vfx.FadeIn(transition_duration)])
                )
                continue

            chosen = resolve_transition(configs[i - 1])
            transition_types_used.append(chosen)
            tdur = transition_duration

            prev_clip    = processed[-1]
            p_dur        = prev_clip.duration
            tail         = prev_clip.subclipped(max(0, p_dur - tdur), p_dur)
            trimmed_prev = prev_clip.subclipped(0, max(0, p_dur - tdur))
            head         = clip.subclipped(0, min(tdur, clip.duration))
            body         = clip.subclipped(min(tdur, clip.duration))

            if chosen in slide_directions:
                composite = make_slide_transition(tail, head, tdur, slide_directions[chosen])
                composite = composite.with_duration(tdur)
                processed[-1] = trimmed_prev
                processed.append(composite)
                processed.append(body.with_effects([vfx.FadeOut(tdur)]) if is_last else body)

            elif chosen == "zoom":
                zoom_tail, zoom_head = make_zoom_transition(tail, head, tdur)
                composite = CompositeVideoClip(
                    [
                        zoom_tail.with_effects([vfx.FadeOut(tdur)]),
                        zoom_head.with_effects([vfx.FadeIn(tdur)]),
                    ],
                    size=tail.size,
                ).with_duration(tdur)
                processed[-1] = trimmed_prev
                processed.append(composite)
                processed.append(body.with_effects([vfx.FadeOut(tdur)]) if is_last else body)

            elif chosen == "blur":
                blur_tail, blur_head = make_blur_transition(tail, head, tdur)
                composite = CompositeVideoClip(
                    [
                        blur_tail.with_effects([vfx.FadeOut(tdur)]),
                        blur_head.with_effects([vfx.FadeIn(tdur)]),
                    ],
                    size=tail.size,
                ).with_duration(tdur)
                processed[-1] = trimmed_prev
                processed.append(composite)
                processed.append(body.with_effects([vfx.FadeOut(tdur)]) if is_last else body)

            else:  # fade
                composite = CompositeVideoClip(
                    [
                        tail.with_effects([vfx.FadeOut(tdur)]),
                        head.with_effects([vfx.FadeIn(tdur)]),
                    ],
                    size=tail.size,
                ).with_duration(tdur)
                processed[-1] = trimmed_prev
                processed.append(composite)
                processed.append(body.with_effects([vfx.FadeOut(tdur)]) if is_last else body)

        last_chosen = resolve_transition(configs[-2]) if n > 1 else None
        if last_chosen not in slide_directions and processed:
            processed[-1] = processed[-1].with_effects([vfx.FadeOut(transition_duration)])

        return processed

    processed_clips = build_clip_sequence()
    _log("Transitions built (50%)")

    # ── Concatenate ───────────────────────────────────────────────────────────
    final_video = concatenate_videoclips(processed_clips, method="compose")
    final_video = final_video.subclipped(0, min(final_video.duration, video_duration))
    _log("Video concatenated and trimmed (75%)")

    # ── Audio ─────────────────────────────────────────────────────────────────
    actual_duration = final_video.duration
    if audio_duration < actual_duration:
        loops_needed = int(actual_duration // audio_duration) + 1
        audio_clip   = concatenate_audioclips([audio_clip] * loops_needed)
    audio_clip  = audio_clip.subclipped(0, actual_duration)
    final_video = final_video.with_audio(audio_clip)
    _log("Audio mixed (85%)")

    # ── Export ────────────────────────────────────────────────────────────────
    _log("Export started (90%)")

    class _CallbackLogger(ProgressBarLogger):
        def __init__(self, cb=None, log=False, start_time=None):
            super().__init__(min_time_interval=0.2)
            self._cb = cb
            self._log = log
            self._t0 = start_time

        def _emit(self, percent):
            msg = f"Rendering: {percent}%"
            if self._cb:
                self._cb(msg)
            elif self._log and self._t0 is not None:
                elapsed = time.perf_counter() - self._t0
                print(f"[{elapsed:6.1f}s] {msg}")

        def bars_callback(self, bar, attr, value, old_value=None):
            if attr != "index":
                return
            total = self.bars.get(bar, {}).get("total")
            if total:
                percent = int((value / total) * 100)
                self._emit(percent)

    final_video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=os.path.splitext(output_path)[0] + "_TEMP_AUDIO.m4a",
        remove_temp=False,
        logger=_CallbackLogger(progress_callback, log_progress, t0) if (progress_callback or log_progress) else None,
    )
    _log("Export finished (100%)")

    durations_summary   = [f"{c['duration']:.1f}s" for c in configs]
    transitions_summary = [(configs[i]["transition"] or transition_type) for i in range(n - 1)]
    motions_summary     = [f"{resolve_motion(c)}@{resolve_speed(c)}x" for c in configs]

    print(f"Video saved to: {output_path}")
    print(f"   Frame size  : {FW}x{FH}  |  Layout: {layout_mode}")
    print(f"   Images      : {n}")
    print(f"   Duration    : {actual_duration:.1f}s total")
    print(f"   Per-image   : {', '.join(durations_summary)}")
    print(f"   Transitions : {', '.join(transitions_summary)}")
    print(f"   Motions     : {', '.join(motions_summary)}")
    print(f"   Cinematic   : {cinematic}  |  Transition dur: {transition_duration}s")


# ── Examples ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # BASE = "/content/drive/MyDrive/editor"
    import os
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

    # ── Example 1: blur_bg layout (default, best for mixed aspect ratios) ────
    # create_video_from_image_and_audio(
    #     image_paths=[
    #         f"{BASE}/img.png",
    #         f"{BASE}/img2.png",
    #         f"{BASE}/img3.png",
    #     ],
    #     audio_path=f"{BASE}/bairan.mp3",
    #     output_path=f"{BASE}/blur_bg.mp4",
    #     transition_type="fade",
    #     motion_type="zoom_in",
    #     motion_speed=0.8,
    #     layout_mode="blur_bg",
    #     frame_size=(1080, 1920),
    # )

    # ── Example 2: fill layout ────────────────────────────────────────────────
    # create_video_from_image_and_audio(
    #     image_paths=[
    #         {"path": f"{BASE}/img.png",  "duration": 4, "transition": "slide_left",  "motion": "move_right", "motion_speed": 0.6},
    #         {"path": f"{BASE}/img2.png", "duration": 4, "transition": "zoom",         "motion": "zoom_in",    "motion_speed": 1.0},
    #         {"path": f"{BASE}/img3.png", "duration": 4, "transition": "slide_right",  "motion": "move_left",  "motion_speed": 0.6},
    #     ],
    #     audio_path=f"{BASE}/bairan.mp3",
    #     output_path=f"{BASE}/fill_layout.mp4",
    #     transition_type="fade",
    #     motion_type="zoom_in",
    #     motion_speed=1.0,
    #     layout_mode="fill",
    #     frame_size=(1080, 1920),
    # )

    # ── Example 3: fit layout (letterbox/pillarbox) ───────────────────────────
    def build_beat_sync_edit(beats, image_paths, audio_path, output_path, BASE):
        """
        Create beat-synced video config and call your existing video function.
        """

        # ── Step 1: Remove very close beats (<0.3s) ─────────────────────────────
        filtered = [beats[0]]
        for b in beats[1:]:
            if b - filtered[-1] > 0.3:
                filtered.append(b)

        print(f"Filtered beats: {len(filtered)}")

        # ── Step 2: Build clips using 2-beat grouping ───────────────────────────
        clips = []
        i = 0
        img_index = 0

        while i < len(filtered) - 2 and img_index < len(image_paths):
            start = filtered[i]
            end   = filtered[i + 2]   # jump 2 beats
            duration = end - start

            clips.append({
                "path": image_paths[img_index],
                "duration": round(duration, 2),
                "transition": "random",
                "motion":"zoom_in",
                "motion_speed": 0.6 + (img_index % 3) * 0.2
            })

            i += 2
            img_index += 1

        print(f"Generated {len(clips)} clips")

        # ── Step 3: Call your main function ─────────────────────────────────────
        create_video_from_image_and_audio(
            image_paths=clips,
            audio_path=audio_path,
            output_path=output_path,

            transition_type="random",
            motion_type="zoom_out",
            motion_speed=0.7,

            layout_mode="blur_bg",
            frame_size=(1080, 1920),
    )
    
    beats = [0.0, 0.035, 0.337, 0.685, 1.045, 1.405, 1.765, 2.113, 2.473, 2.833, 3.193, 3.541, 3.901, 4.261, 4.621, 4.969, 5.329, 5.689, 6.049, 6.397]

    images = [
        f"{BASE}/img5.png",
        f"{BASE}/img6.png",
        f"{BASE}/img7.png",
        f"{BASE}/img8.png",
        f"{BASE}/img.png",
        f"{BASE}/img2.png",
        f"{BASE}/img3.png",
        f"{BASE}/img4.png",
        # f"{BASE}/img5.png",
        # f"{BASE}/img6.png",
        # f"{BASE}/img7.png",
        # f"{BASE}/img8.png",
    ]

    # build_beat_sync_edit(
    #     beats=beats,
    #     image_paths=images,
    #     audio_path=f"{BASE}/shisha.mp3",
    #     output_path=f"{BASE}/beat_sync_final_shisha.mp4",
    #     BASE=BASE
    # )
    create_video_from_image_and_audio(
            image_paths=images,
            audio_path=f"{BASE}/shisha.mp3",
            output_path=f"{BASE}/beat_sync_final_shisha_2.mp4",

            transition_type="random",
            motion_type="zoom_in",
            motion_speed=2,
            duration=1.5,
            layout_mode="blur_bg",
            frame_size=(1080, 1920),
    )

