# !apt-get install -y python3-gi python3-gi-cairo gir1.2-pango-1.0 \
#                     libcairo2-dev libpango1.0-dev libpangocairo-1.0-0
# !pip install moviepy pillow pycairo PyGObject

# !wget -q -O /content/NotoSansDevanagari-Regular.ttf \
#     "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf"


def overlay_subtitles(video_path, captions, output_path, font_path=None):
    """
    Overlays Hindi/Devanagari/Punjabi/Gurmukhi subtitles onto a video.

    Uses GI (PyGObject) + Pango + cairo for correct OpenType shaping.
    No boxes. Works with all Unicode scripts.

    Colab setup (run once):
        !apt-get install -y python3-gi python3-gi-cairo gir1.2-pango-1.0 \
                            libcairo2-dev libpango1.0-dev libpangocairo-1.0-0
        !pip install moviepy pillow pycairo PyGObject

    Args:
        video_path  (str):      Path to the input video file.
        captions    (list):     [{"start": float, "end": float, "text": str}, ...]
        output_path (str):      Path to save the final subtitled MP4.
        font_path   (str|None): Path to a .ttf font (e.g. NotoSansDevanagari-Regular.ttf).
    """
    import os
    import re
    import textwrap
    import numpy as np
    from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

    # ── Validate ──────────────────────────────────────────────────────────────
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not captions:
        raise ValueError("Captions list is empty.")

    # ── Load video ────────────────────────────────────────────────────────────
    print(f"📹 Loading video: {video_path} ...")
    video        = VideoFileClip(video_path)
    vid_w, vid_h = video.size
    fps          = video.fps

    # ── Layout constants ──────────────────────────────────────────────────────
    FONT_SIZE      = max(int(vid_h * 0.048), 26)
    BOTTOM_MARGIN  = max(int(vid_h * 0.06), 40)
    # FIX: Hindi/Devanagari glyphs are ~1.6× wider than ASCII.
    # Using 38 caused lines to overflow; 24 keeps text within the safe width.
    MAX_CHARS_LINE = 24
    PAD_X          = 20
    PAD_Y          = 14
    LINE_SPACING   = max(int(FONT_SIZE * 0.3), 8)
    BG_ALPHA       = 0.68
    STROKE_WIDTH   = 2
    FADE_DUR       = 0.15

    # ── Register custom font with fontconfig ──────────────────────────────────
    font_family = "Noto Sans"
    if font_path and os.path.exists(font_path):
        font_dir  = os.path.dirname(os.path.abspath(font_path))
        conf_path = "/tmp/custom_fonts.conf"
        with open(conf_path, "w") as f:
            f.write(f"""<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig><dir>{font_dir}</dir></fontconfig>""")
        os.environ["FONTCONFIG_FILE"] = conf_path

        # "NotoSansDevanagari-Regular" → "Noto Sans Devanagari"
        raw         = os.path.splitext(os.path.basename(font_path))[0]
        raw         = re.sub(r'-(Regular|Bold|Light|Medium|Italic).*$', '', raw)
        font_family = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', raw)
        print(f"🔤 Font family: '{font_family}'  ({font_path})")
    else:
        print(f"🔤 Using system font: '{font_family}'")

    # ── Import GI / Pango / cairo ─────────────────────────────────────────────
    try:
        import gi
        gi.require_version("Pango", "1.0")
        gi.require_version("PangoCairo", "1.0")
        from gi.repository import Pango, PangoCairo
        import cairo
        USE_PANGO = True
        print("✅ Using GI Pango + pycairo for shaped text rendering.")
    except Exception as e:
        USE_PANGO = False
        print(f"⚠️  GI/Pango unavailable ({e}) — falling back to Pillow.")

    # ── Shared helper: compute clamped pill geometry ──────────────────────────
    def _pill_geometry(txt_w, txt_h, canvas_w, canvas_h):
        """
        Returns (pill_x, pill_y, pill_w, pill_h) guaranteed to stay
        fully within the canvas bounds.
        """
        SAFE_W = int(canvas_w * 0.90)           # hard max width (90 % of frame)

        pill_w = min(txt_w + PAD_X * 2, SAFE_W)
        pill_w = max(pill_w, 1)                  # never zero

        pill_h = int(txt_h + PAD_Y * 2)
        pill_h = min(pill_h, canvas_h - 8)       # never taller than frame

        # FIX: clamp x so pill never goes negative or past right edge
        pill_x = max((canvas_w - pill_w) // 2, 0)

        # FIX: clamp y so pill never goes above y=4 or past bottom edge
        pill_y_ideal = canvas_h - BOTTOM_MARGIN - pill_h
        pill_y = max(min(pill_y_ideal, canvas_h - pill_h - 4), 4)

        return pill_x, pill_y, pill_w, pill_h

    # ══════════════════════════════════════════════════════════════════════════
    #  RENDERER A: GI Pango + pycairo  (correct shaping — no boxes)
    # ══════════════════════════════════════════════════════════════════════════
    def render_pango(text, canvas_w, canvas_h):
        lines = textwrap.wrap(text, width=MAX_CHARS_LINE) or [text]

        def make_layout(cairo_ctx, line_text):
            layout = PangoCairo.create_layout(cairo_ctx)
            layout.set_text(line_text, -1)
            desc = Pango.FontDescription.from_string(
                f"{font_family} {FONT_SIZE}"
            )
            layout.set_font_description(desc)
            layout.set_alignment(Pango.Alignment.CENTER)
            return layout

        # Measure on a throw-away 1×1 surface
        dummy_surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        dummy_ctx  = cairo.Context(dummy_surf)
        line_dims  = []
        for line in lines:
            lo     = make_layout(dummy_ctx, line)
            pw, ph = lo.get_pixel_size()
            line_dims.append((pw, ph))

        txt_w = max(d[0] for d in line_dims)
        txt_h = sum(d[1] for d in line_dims) + LINE_SPACING * (len(lines) - 1)

        # FIX: use shared clamped geometry helper
        pill_x, pill_y, pill_w, pill_h = _pill_geometry(
            txt_w, txt_h, canvas_w, canvas_h
        )

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, canvas_w, canvas_h)
        ctx     = cairo.Context(surface)

        ctx.set_source_rgba(0, 0, 0, 0)
        ctx.paint()

        # Rounded rect pill background
        r  = 12.0
        x0, y0 = float(pill_x), float(pill_y)
        x1, y1 = x0 + pill_w,   y0 + pill_h
        ctx.new_sub_path()
        ctx.arc(x0 + r, y0 + r, r, np.pi,       1.5 * np.pi)
        ctx.arc(x1 - r, y0 + r, r, 1.5 * np.pi, 0)
        ctx.arc(x1 - r, y1 - r, r, 0,           0.5 * np.pi)
        ctx.arc(x0 + r, y1 - r, r, 0.5 * np.pi, np.pi)
        ctx.close_path()
        ctx.set_source_rgba(0, 0, 0, BG_ALPHA)
        ctx.fill()

        cursor_y = pill_y + PAD_Y
        for i, line in enumerate(lines):
            lw, lh = line_dims[i]
            tx = pill_x + (pill_w - lw) // 2

            # Black stroke — 8-direction
            lo = make_layout(ctx, line)
            for dx, dy in [(-STROKE_WIDTH, 0), (STROKE_WIDTH, 0),
                           (0, -STROKE_WIDTH), (0, STROKE_WIDTH),
                           (-STROKE_WIDTH, -STROKE_WIDTH), (STROKE_WIDTH, -STROKE_WIDTH),
                           (-STROKE_WIDTH,  STROKE_WIDTH), (STROKE_WIDTH,  STROKE_WIDTH)]:
                ctx.move_to(tx + dx, cursor_y + dy)
                ctx.set_source_rgba(0, 0, 0, 1)
                PangoCairo.show_layout(ctx, lo)

            # White main text
            ctx.move_to(tx, cursor_y)
            ctx.set_source_rgba(1, 1, 1, 1)
            PangoCairo.show_layout(ctx, lo)

            cursor_y += lh + LINE_SPACING

        buf  = surface.get_data()
        arr  = np.frombuffer(buf, dtype=np.uint8).reshape((canvas_h, canvas_w, 4))
        rgba = arr[:, :, [2, 1, 0, 3]].copy()
        return rgba

    # ══════════════════════════════════════════════════════════════════════════
    #  RENDERER B: Pillow fallback
    # ══════════════════════════════════════════════════════════════════════════
    def render_pillow(text, canvas_w, canvas_h):
        from PIL import Image, ImageDraw, ImageFont

        def _pil_font(size):
            for p in [font_path,
                      "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
                if p and os.path.exists(p):
                    try:
                        return ImageFont.truetype(p, size)
                    except Exception:
                        pass
            return ImageFont.load_default()

        pf    = _pil_font(FONT_SIZE)
        lines = textwrap.wrap(text, width=MAX_CHARS_LINE) or [text]

        line_dims = []
        for line in lines:
            try:
                bb = pf.getbbox(line)
                lw, lh = bb[2] - bb[0], bb[3] - bb[1]
            except Exception:
                dummy = Image.new("RGBA", (1, 1))
                lw, lh = ImageDraw.Draw(dummy).textsize(line, font=pf)
            try:
                asc, desc = pf.getmetrics()
                lh = max(lh, asc + desc)
            except Exception:
                pass
            line_dims.append((lw, lh))

        txt_w = max(d[0] for d in line_dims)
        txt_h = sum(d[1] for d in line_dims) + LINE_SPACING * (len(lines) - 1)

        # FIX: use shared clamped geometry helper
        pill_x, pill_y, pill_w, pill_h = _pill_geometry(
            txt_w, txt_h, canvas_w, canvas_h
        )

        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw   = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
            radius=12, fill=(0, 0, 0, int(BG_ALPHA * 255)),
        )
        cursor_y = pill_y + PAD_Y
        for i, line in enumerate(lines):
            lw, lh = line_dims[i]
            tx = pill_x + (pill_w - lw) // 2
            for dx, dy in [(-STROKE_WIDTH, 0), (STROKE_WIDTH, 0),
                           (0, -STROKE_WIDTH), (0, STROKE_WIDTH),
                           (-STROKE_WIDTH, -STROKE_WIDTH), (STROKE_WIDTH, -STROKE_WIDTH),
                           (-STROKE_WIDTH,  STROKE_WIDTH), (STROKE_WIDTH,  STROKE_WIDTH)]:
                draw.text((tx + dx, cursor_y + dy), line, font=pf, fill=(0, 0, 0, 255))
            draw.text((tx, cursor_y), line, font=pf, fill=(255, 255, 255, 255))
            cursor_y += lh + LINE_SPACING

        return np.array(canvas)

    # ── Select renderer ───────────────────────────────────────────────────────
    render_subtitle_image = render_pango if USE_PANGO else render_pillow

    # ── Build one ImageClip per caption ───────────────────────────────────────
    print(f"✍️  Building {len(captions)} subtitle clip(s) ...")
    subtitle_clips = []

    for cap in captions:
        start    = cap["start"]
        end      = cap["end"]
        text     = cap["text"].strip()
        duration = end - start

        if not text or duration <= 0:
            continue

        frame = render_subtitle_image(text, vid_w, vid_h)

        clip = (
            ImageClip(frame, ismask=False)
            .set_start(start)
            .set_duration(duration)
            .set_position((0, 0))
            .crossfadein(FADE_DUR)
            .crossfadeout(FADE_DUR)
        )
        subtitle_clips.append(clip)

    # ── Composite & export ────────────────────────────────────────────────────
    final = CompositeVideoClip([video, *subtitle_clips], size=(vid_w, vid_h))
    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    print(f"✅ Subtitled video saved to: {output_path}")


# ── Quick-start ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hindi_captions = [
        {"start": 0.0,  "end": 4.0,  "text": "खोया रहूँ तेरी याद करके, नादानियाँ बहुत मैंने"},
        {"start": 4.0,  "end": 8.0,  "text": "मैं रखूँ तेरी संभाल के निशानियाँ, तेरे बिना"},
        {"start": 8.0,  "end": 12.0, "text": "हाल मेरा ठीक नहीं होता, आज मुझे संभाल ले"},
    ]

    overlay_subtitles(
        video_path="slideshow.mp4",
        captions=hindi_captions,
        output_path="subtitled_hindi.mp4",
        font_path="/content/NotoSansDevanagari-Regular.ttf",
    )