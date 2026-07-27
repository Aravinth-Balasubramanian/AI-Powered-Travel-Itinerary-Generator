import os, numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, concatenate_videoclips, AudioFileClip
import os
os.makedirs("videos", exist_ok=True)

OUTPUT_FOLDER, MUSIC_PATH, W, H, FPS, DURATION = "videos", "assets/music.mp3", 1280, 720, 24, 4

PALETTES = [
    ((10, 25, 60), (30, 80, 160)),    # Deep navy
    ((40, 10, 60), (120, 30, 140)),   # Royal purple
    ((10, 50, 40), (20, 120, 90)),    # Forest teal
    ((60, 20, 10), (160, 60, 20)),    # Burnt sienna
]
ACCENT = (250, 180, 50)


def gradient_bg(img, top, bot):
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        arr[y] = [int(top[c] + (bot[c] - top[c]) * t) for c in range(3)]
    img.paste(Image.fromarray(arr))


def load_fonts():
    for path in ["arial.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(path, 72), ImageFont.truetype(path, 34), ImageFont.truetype(path, 22)
        except Exception:
            pass
    d = ImageFont.load_default()
    return d, d, d


def make_card(place_name, city, day, index, total):
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    palette = PALETTES[index % len(PALETTES)]
    gradient_bg(img, palette[0], palette[1])

    # Diagonal accent stripe
    draw.polygon([(0, H - 180), (0, H), (220, H)], fill=(*ACCENT, 180))

    # Progress bar
    progress = int((index / total) * (W - 80))
    draw.rectangle([40, H - 22, W - 40, H - 10], fill=(255, 255, 255, 40))
    draw.rectangle([40, H - 22, 40 + progress, H - 10], fill=ACCENT)

    f_big, f_med, f_sml = load_fonts()

    # Day badge — pill shape
    draw.rounded_rectangle([40, 34, 160, 76], radius=20, fill=ACCENT)
    draw.text((100, 55), f"DAY {day}", font=f_sml, fill=(20, 20, 20), anchor="mm")

    # Index counter top-right
    draw.ellipse([W - 96, 26, W - 34, 88], outline=ACCENT, width=3)
    draw.text((W - 65, 57), f"{index}/{total}", font=f_sml, fill="white", anchor="mm")

    # Place name — with soft shadow
    draw.text((62, H // 2 - 68), place_name, font=f_big, fill=(0, 0, 0, 120))
    draw.text((60, H // 2 - 70), place_name, font=f_big, fill="white")

    # City line
    draw.text((62, H // 2 + 38), f"📍  {city}", font=f_med, fill=(200, 220, 255))

    # Thin separator
    draw.line([(40, H // 2 + 90), (W - 40, H // 2 + 90)], fill=(255, 255, 255, 60), width=1)

    # Branding footer
    draw.text((W // 2, H - 40), "✈  ItinerAI — AI Travel Planner", font=f_sml, fill=(180, 200, 240), anchor="mm")

    return np.array(img)


def generate_video(itinerary, city="India", output_path=None):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = output_path or os.path.join(OUTPUT_FOLDER, "travel_video.mp4")
    total = len(itinerary)
    clips = [
        ImageClip(make_card(p.get("name", "Unknown"), p.get("city", city),
                            p.get("day", 1), i, total)).with_duration(DURATION)
        for i, p in enumerate(itinerary, 1)
    ]
    video = concatenate_videoclips(clips, method="compose")
    if os.path.exists(MUSIC_PATH):
        video = video.with_audio(AudioFileClip(MUSIC_PATH).subclipped(0, video.duration))
    video.write_videofile(output_path, fps=FPS, codec="libx264", audio_codec="aac", logger=None)
    print(f"✅  Video saved → {output_path}")
    return output_path


if __name__ == "__main__":
    sample = [
        {"name": "India Gate",     "city": "Delhi", "day": 1},
        {"name": "Qutub Minar",    "city": "Delhi", "day": 1},
        {"name": "Humayun's Tomb", "city": "Delhi", "day": 1},
        {"name": "Red Fort",       "city": "Delhi", "day": 2},
        {"name": "Lotus Temple",   "city": "Delhi", "day": 2},
    ]
    generate_video(sample, city="Delhi")