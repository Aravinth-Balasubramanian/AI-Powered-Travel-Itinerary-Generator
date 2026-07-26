import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, concatenate_videoclips, AudioFileClip

OUTPUT_FOLDER = "videos"
MUSIC_PATH = "assets/music.mp3"
W, H = 1280, 720


def make_card(place_name, city, day, index):
    """Draw a single place card and return it as a numpy array."""

    # Dark blue background
    img = Image.new("RGB", (W, H), color=(15, 36, 71))
    draw = ImageDraw.Draw(img)

    # Amber left bar
    draw.rectangle([0, 0, 8, H], fill=(245, 158, 11))

    # Try to load a font, fallback to default
    try:
        font_big = ImageFont.truetype("arial.ttf", 70)
        font_med = ImageFont.truetype("arial.ttf", 36)
        font_sml = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font_big = font_med = font_sml = ImageFont.load_default()

    # Day badge
    draw.rounded_rectangle([40, 30, 170, 80], radius=12, fill=(245, 158, 11))
    draw.text((105, 55), f"DAY {day}", font=font_sml, fill="white", anchor="mm")

    # Place number circle
    draw.ellipse([W - 110, 20, W - 40, 90], fill=(245, 158, 11))
    draw.text((W - 75, 55), f"#{index}", font=font_sml, fill="white", anchor="mm")

    # Place name
    draw.text((60, H // 2 - 60), place_name, font=font_big, fill="white")

    # City
    draw.text((60, H // 2 + 40), f"📍  {city}", font=font_med, fill=(203, 213, 225))

    # Divider line
    draw.line([(60, H - 100), (W - 60, H - 100)], fill=(100, 120, 160), width=1)

    # Branding
    draw.text((W // 2, H - 50), "🧭  ItinerAI — AI Travel Planner",
              font=font_sml, fill=(203, 213, 225), anchor="mm")

    return np.array(img)


def generate_video(itinerary, city="India", days=3, output_path=None):
    """
    itinerary : list of dicts — each with keys: name, city, day
    city      : destination city name
    output_path: where to save the MP4
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(OUTPUT_FOLDER, "travel_video.mp4")

    clips = []
    for i, place in enumerate(itinerary, start=1):
        frame = make_card(
            place_name=place.get("name", "Unknown"),
            city=place.get("city", city),
            day=place.get("day", 1),
            index=i,
        )
        clips.append(ImageClip(frame).with_duration(3))   # 3 seconds per place

    video = concatenate_videoclips(clips, method="compose")

    # Add music if the file exists
    if os.path.exists(MUSIC_PATH):
        audio = AudioFileClip(MUSIC_PATH).subclipped(0, video.duration)
        video = video.with_audio(audio)

    video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
    print(f"✅ Video saved: {output_path}")
    return output_path


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [
        {"name": "India Gate",     "city": "Delhi", "day": 1},
        {"name": "Qutub Minar",    "city": "Delhi", "day": 1},
        {"name": "Humayun's Tomb", "city": "Delhi", "day": 1},
        {"name": "Red Fort",       "city": "Delhi", "day": 2},
        {"name": "Lotus Temple",   "city": "Delhi", "day": 2},
    ]
    generate_video(sample, city="Delhi", days=2)