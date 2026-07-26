# AI-Powered Travel Itinerary Generator

A final year project I built to learn how RAG actually works in practice — not just the theory. You give it a city, it gives you a day-wise travel itinerary, an interactive route map, and a ranked list of places scored by an XGBoost model I trained on Indian tourism data.

Live at `localhost:8501` when you run it. No hosted version yet.

---

## What it does

Type "Jaipur" → get a 3-day itinerary with places pulled from a ChromaDB vector store, routed via OSRM, geocoded with Nominatim, and ranked by a model that weighs rating, visit duration, and entry fee. The LLM (Llama 3.3 70B via Groq) only writes the prose — it can't invent places that aren't in the database.

There's also a video export tab that renders a short MP4 highlight reel of your trip, one card per place.

---

## Stack

| Layer | What I used |
|---|---|
| LLM | Llama 3.3 70B via `langchain-groq` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (local, persistent) |
| Ranking | XGBoost regression pipeline (`scikit-learn` + `xgboost`) |
| Routing | OSRM public API |
| Geocoding | Nominatim (OpenStreetMap) — 1s delay between calls to respect rate limits |
| Maps | Folium + `streamlit-folium` |
| Video | MoviePy + Pillow |
| History | SQLite (stores last 10 trips) |
| UI | Streamlit |

---

## Project structure

```
itinerai/
├── app.py                  # everything — UI, backend classes, RAG pipeline
├── data/
│   └── cleaned_places.csv  # 325 tourist places across India (not included)
├── vector_db/              # ChromaDB persists here after first run
├── videos/                 # generated MP4s go here
├── xgboost_model.pkl       # pre-trained ranker
├── .env                    # your API keys (not committed)
├── requirements.txt
└── notebooks/
    ├── preprocessing.ipynb
    ├── embedding.ipynb
    ├── model_training.ipynb
    ├── osrm_router.ipynb
    └── geogrpahy.ipynb
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/yourusername/itinerai.git
cd itinerai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

Torch takes a while. If you don't have a GPU, the CPU version is fine — embeddings are small.

**3. Add your API key**

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com). The free tier is enough for this.

**4. (Optional) Add the full dataset**

The app ships with 16 sample places hardcoded as a fallback. For all 325 places, put `cleaned_places.csv` inside a `data/` folder. The vector DB builds itself on first run — takes about 30 seconds.

**5. Run**

```bash
streamlit run app.py
```

---

## Training the ranker (optional)

If you want to retrain the XGBoost model on your own dataset:

```bash
# Open notebooks/model_training.ipynb
# Point it at your cleaned_places.csv
# Run all cells — saves xgboost_model.pkl automatically
```

The model predicts a `recommendation_score` built from a weighted combination of normalized rating (50%), visit duration (30%), and inverted entry fee (20%). RMSE on the test set was 0.02, which is fine for ranking purposes.

---

## Known issues

- **Geocoding is slow** — Nominatim enforces a 1 req/sec rate limit, so a 9-place trip takes ~10 seconds to geocode. I added `time.sleep(1.1)` between calls. Using a paid geocoding API would fix this.
- **Kerala / smaller cities** — If a city isn't in the dataset, the city filter returns zero places. The video tab falls back to the top RAG results in that case; the itinerary tab still generates prose from the LLM.
- **Font in videos** — The video generator tries `arial.ttf` first and falls back to PIL's default bitmap font if it's not found. On Linux servers, the default font looks rough. Drop any `.ttf` file in the project root and update the path in `generate_video()`.
- **ChromaDB cold start** — First run downloads the embedding model (~90MB). Subsequent runs load from cache.

---

## Notebooks

The `notebooks/` folder has the exploration work:

- `preprocessing.ipynb` — cleaning the raw Kaggle dataset, handling missing values, normalizing types
- `embedding.ipynb` — building the ChromaDB collection, testing retrieval quality
- `model_training.ipynb` — training and evaluating the XGBoost pipeline
- `osrm_router.ipynb` — testing the OSRM routing API
- `geogrpahy.ipynb` — early geocoding experiments with Mapbox (switched to Nominatim to avoid API costs)

---

## What I'd do differently

- Replace the hardcoded SAMPLE fallback with a proper seeding script
- Add caching for geocoding results so repeated queries don't hit Nominatim
- Use a streaming LLM response so the itinerary appears word by word instead of all at once

## 🌐 Live Demo

🔗 https://ai-powered-travel-itinerary-generator.streamlit.app/
- The video export is purely PIL — would be nicer with proper motion between cards

---

