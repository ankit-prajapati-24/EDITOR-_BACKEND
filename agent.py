import os
from pathlib import Path
from groq import Groq
from final.generate_video_final import create_video_from_image_and_audio
from pydub import AudioSegment
import json

# -------------------------
# CONFIG
# -------------------------
API_KEY = ""
client = Groq(api_key=API_KEY)

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------
# Beat detection helper (simple, optional)
# -------------------------
def detect_audio_beats(audio_path):
    # For simplicity: just get audio length in seconds
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000  # seconds

# -------------------------
# LLM Agent to plan video
# -------------------------
def ai_plan_video(images, audio_path):
    prompt = f"""
    You are a video editing assistant.
    Given {len(images)} images and audio {audio_path}, 
    suggest for each image:
    - duration (seconds)
    - transition type (fade, slide_left, slide_right, zoom, etc.)
    - motion type (zoom_in, move_left, move_right, etc.)
    - motion speed (0.5-1.5)
    Also suggest layout_mode (blur_bg, fill) and  motion_type for the video.
    Return output as JSON with keys "images" and "video_settings".
    The "images" value must be a list of dicts with keys:
    motion_type must be one of {'move_right', 'zoom_in', 'move_left', 'zoom_out'}
    available images:"img.png", "img2.png", "img3.png", "img4.png", "img5.png", "img6.png", "img7.png", "img8.png"
    - path
    - duration
    - transition
    - motion
    - motion_speed
    Return ONLY valid JSON. No extra text, no code fences.
    """
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=4096,
        top_p=1,
        reasoning_effort="medium",
        stream=False,
    )

    # Safely parse output (expect JSON)
    plan_str = (response.choices[0].message.content or "").strip()
    if not plan_str:
        raise ValueError(f"Empty model response. Full response: {response}")
    # Remove code fences if present
    if plan_str.startswith("```"):
        plan_str = plan_str.strip("`")
        if plan_str.lower().startswith("json"):
            plan_str = plan_str[4:].strip()
    # Extract JSON object if extra text slipped in
    if "{" in plan_str and "}" in plan_str:
        plan_str = plan_str[plan_str.find("{"):plan_str.rfind("}") + 1]
    plan = json.loads(plan_str)
    return plan

# -------------------------
# Main function
# -------------------------
def generate_video(images, audio_path):
    images = [str(p) for p in images]
    audio_path = str(audio_path)
    plan = ai_plan_video(images, audio_path)
    print("AI Plan:", json.dumps(plan, indent=2))
    # return
    
    output_path = OUTPUT_DIR / "final_output.mp4"
    
    create_video_from_image_and_audio(
        image_paths=plan["images"],
        audio_path=audio_path,
        output_path=str(output_path),
        **plan["video_settings"]
    )
    
    # Return local URL path
    return f"file://{output_path.resolve()}"

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    images = [
        ASSETS_DIR / "img.png",
        ASSETS_DIR / "img2.png",
        ASSETS_DIR / "img3.png",
    ]
    audio = ASSETS_DIR / "shisha.mp3"
    
    local_url = generate_video(images, audio)
    print("Video generated at:", local_url)
