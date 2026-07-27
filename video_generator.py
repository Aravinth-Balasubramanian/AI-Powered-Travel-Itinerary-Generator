import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# moviepy 1.0.3 compatible imports
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

# Ensure output folder exists at import time
os.makedirs("videos", exist_ok=True)

OUTPUT_FOLDER = "videos"
MUSIC_PATH    = "assets/music.mp3"
W, H, FPS, DURATION = 1280, 720, 24, 4

PALETTES = [
    ((10, 25, 60),  (30, 80, 160)),   # Deep navy
    ((40, 10, 60),  (120, 30, 140)),  # Royal purple
    ((10, 50, 40),  (20, 120, 90)),   # Forest teal
    ((60, 20, 10),  (160, 60, 20)),   # Burnt sienna
]
ACCENT = (250, 180, 50)


def gradient_bg(img, top, bot):
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        arr[y] = [int(top[c] + (bot[c] - top[c]) * t) for c in range(3)]
    img.paste(Image.fromarray(arr))


def load_fonts():
    """Try common font paths; fall back to PIL default."""
    font_paths = [
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
    ]
    for path in font_paths:
        try:
            return (
                ImageFont.truetype(path, 72),
                ImageFont.truetype(path, 34),
                ImageFont.truetype(path, 22),
            )
        except Exception:
            pass
    # PIL default bitmap font — no size arg supported
    d = ImageFont.load_default()
    return d, d, d


def make_card(place_name, city, day, index, total):
    """Render one gradient story-card as a numpy array."""
    img  = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img, "RGBA")   # RGBA mode so alpha fills work
    palette = PALETTES[index % len(PALETTES)]
    gradient_bg(img, palette[0], palette[1])

    # Diagonal accent stripe (bottom-left corner)
    draw.polygon([(0, H - 180), (0, H), (220, H)], fill=(*ACCENT, 180))

    # Progress bar
    progress = int((index / total) * (W - 80))
    draw.rectangle([40, H - 22, W - 40, H - 10], fill=(255, 255, 255, 40))
    draw.rectangle([40, H - 22, 40 + progress, H - 10], fill=(*ACCENT, 255))

    f_big, f_med, f_sml = load_fonts()

    # Day badge — pill shape
    draw.rounded_rectangle([40, 34, 160, 76], radius=20, fill=(*ACCENT, 255))
    draw.text((100, 55), f"DAY {day}", font=f_sml, fill=(20, 20, 20), anchor="mm")

    # Index counter — top-right circle
    draw.ellipse([W - 96, 26, W - 34, 88], outline=(*ACCENT, 255), width=3)
    draw.text((W - 65, 57), f"{index}/{total}", font=f_sml, fill="white", anchor="mm")

    # Place name with soft shadow
    draw.text((62, H // 2 - 68), place_name, font=f_big, fill=(0, 0, 0, 120))
    draw.text((60, H // 2 - 70), place_name, font=f_big, fill="white")

    # City line (emoji stripped to avoid font issues on servers)
    city_text = f"[ {city} ]"
    draw.text((62, H // 2 + 38), city_text, font=f_med, fill=(200, 220, 255))

    # Thin separator line
    draw.line(
        [(40, H // 2 + 90), (W - 40, H // 2 + 90)],
        fill=(255, 255, 255, 60),
        width=1,
    )

    # Branding footer
    draw.text(
        (W // 2, H - 40),
        "ItinerAI  —  AI Travel Planner",
        font=f_sml,
        fill=(180, 200, 240),
        anchor="mm",
    )

    # Convert back to plain RGB array for MoviePy
    return np.array(img.convert("RGB"))


def generate_video(itinerary, city="India", output_path=None):
    """
    Generate an MP4 highlight video from a list of place dicts.

    Each dict should have: name, city (optional), day (optional).
    Returns the output file path.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = output_path or os.path.join(OUTPUT_FOLDER, "travel_video.mp4")

    if not itinerary:
        raise ValueError("Itinerary is empty — cannot generate video.")

    total = len(itinerary)
    clips = []

    for i, p in enumerate(itinerary, 1):
        frame = make_card(
            place_name=str(p.get("name", "Unknown")),
            city=str(p.get("city", city)),
            day=p.get("day", 1),
            index=i,
            total=total,
        )
        # moviepy 1.0.3: set_duration (not with_duration)
        clip = ImageClip(frame).set_duration(DURATION)
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")

    # Add background music if available
    if os.path.exists(MUSIC_PATH):
        try:
            audio = AudioFileClip(MUSIC_PATH).subclip(0, video.duration)
            video = video.set_audio(audio)
        except Exception as e:
            print(f"⚠️  Music skipped: {e}")

    video.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    print(f"✅  Video saved → {output_path}")
    return output_path


if __name__ == "__main__":
    sample = [
        {"name": "India Gate",      "city": "Delhi", "day": 1},
        {"name": "Qutub Minar",     "city": "Delhi", "day": 1},
        {"name": "Humayun's Tomb",  "city": "Delhi", "day": 1},
        {"name": "Red Fort",        "city": "Delhi", "day": 2},
        {"name": "Lotus Temple",    "city": "Delhi", "day": 2},
    ]
    generate_video(sample, city="Delhi")