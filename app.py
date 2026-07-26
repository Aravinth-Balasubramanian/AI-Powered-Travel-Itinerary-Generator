"""
ItinerAI — AI Travel Planner  |  streamlit run app.py
"""
import os, uuid, sqlite3, time
from datetime import datetime
from pathlib import Path
import joblib, pandas as pd, requests, streamlit as st, folium
from streamlit_folium import st_folium
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import chromadb
import torchvision

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="ItinerAI", page_icon="🌴", layout="wide", initial_sidebar_state="expanded")

_HERE      = Path(__file__).parent
DATA_PATH  = _HERE / "data" / "cleaned_places.csv"
CHROMA_DIR = str(_HERE / "vector_db")
DB_PATH    = str(_HERE / "itinerai_history.db")
MODEL_PATH = str(_HERE / "xgboost_model.pkl")

SAMPLE = pd.DataFrame([
    {"Zone":"Northern","State":"Rajasthan","City":"Jaipur","Name":"Amber Fort","Type":"Historical","Establishment Year":"1592","time needed to visit in hrs":3.0,"Google review rating":4.8,"Entrance Fee in INR":200,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Historical","DSLR Allowed":"Yes","Number of google review in lakhs":1.5,"Best Time to visit":"Morning"},
    {"Zone":"Northern","State":"Rajasthan","City":"Jaipur","Name":"Hawa Mahal","Type":"Historical","Establishment Year":"1799","time needed to visit in hrs":2.0,"Google review rating":4.7,"Entrance Fee in INR":50,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Historical","DSLR Allowed":"Yes","Number of google review in lakhs":1.2,"Best Time to visit":"Morning"},
    {"Zone":"Northern","State":"Rajasthan","City":"Jaipur","Name":"City Palace","Type":"Historical","Establishment Year":"1729","time needed to visit in hrs":2.5,"Google review rating":4.6,"Entrance Fee in INR":300,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Historical","DSLR Allowed":"Yes","Number of google review in lakhs":0.9,"Best Time to visit":"Afternoon"},
    {"Zone":"Northern","State":"Rajasthan","City":"Jaipur","Name":"Jal Mahal","Type":"Lake","Establishment Year":"1750","time needed to visit in hrs":1.0,"Google review rating":4.5,"Entrance Fee in INR":0,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Natural","DSLR Allowed":"Yes","Number of google review in lakhs":0.7,"Best Time to visit":"Evening"},
    {"Zone":"Northern","State":"Delhi","City":"Delhi","Name":"India Gate","Type":"War Memorial","Establishment Year":"1921","time needed to visit in hrs":0.5,"Google review rating":4.6,"Entrance Fee in INR":0,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Historical","DSLR Allowed":"Yes","Number of google review in lakhs":2.6,"Best Time to visit":"Evening"},
    {"Zone":"Northern","State":"Delhi","City":"Delhi","Name":"Humayun's Tomb","Type":"Tomb","Establishment Year":"1572","time needed to visit in hrs":2.0,"Google review rating":4.5,"Entrance Fee in INR":30,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Historical","DSLR Allowed":"Yes","Number of google review in lakhs":0.4,"Best Time to visit":"Afternoon"},
    {"Zone":"Northern","State":"Delhi","City":"Delhi","Name":"Qutub Minar","Type":"Historical","Establishment Year":"1193","time needed to visit in hrs":2.5,"Google review rating":4.5,"Entrance Fee in INR":35,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Historical","DSLR Allowed":"Yes","Number of google review in lakhs":0.8,"Best Time to visit":"Morning"},
    {"Zone":"Southern","State":"Karnataka","City":"Mysore","Name":"Mysore Palace","Type":"Historical","Establishment Year":"1912","time needed to visit in hrs":3.0,"Google review rating":4.7,"Entrance Fee in INR":100,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Historical","DSLR Allowed":"No","Number of google review in lakhs":2.0,"Best Time to visit":"Evening"},
    {"Zone":"Southern","State":"Karnataka","City":"Mysore","Name":"Chamundi Hills","Type":"Religious","Establishment Year":"1659","time needed to visit in hrs":2.0,"Google review rating":4.5,"Entrance Fee in INR":0,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Religious","DSLR Allowed":"Yes","Number of google review in lakhs":0.5,"Best Time to visit":"Morning"},
    {"Zone":"Western","State":"Goa","City":"Goa","Name":"Baga Beach","Type":"Beach","Establishment Year":"NA","time needed to visit in hrs":3.0,"Google review rating":4.3,"Entrance Fee in INR":0,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Natural","DSLR Allowed":"Yes","Number of google review in lakhs":0.6,"Best Time to visit":"Morning"},
    {"Zone":"Western","State":"Goa","City":"Goa","Name":"Basilica of Bom Jesus","Type":"Religious","Establishment Year":"1605","time needed to visit in hrs":1.5,"Google review rating":4.6,"Entrance Fee in INR":0,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Religious","DSLR Allowed":"Yes","Number of google review in lakhs":0.4,"Best Time to visit":"Morning"},
    {"Zone":"Southern","State":"Tamil Nadu","City":"Chennai","Name":"Marina Beach","Type":"Beach","Establishment Year":"NA","time needed to visit in hrs":2.0,"Google review rating":4.3,"Entrance Fee in INR":0,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Natural","DSLR Allowed":"Yes","Number of google review in lakhs":1.0,"Best Time to visit":"Evening"},
    {"Zone":"Eastern","State":"West Bengal","City":"Kolkata","Name":"Victoria Memorial","Type":"Historical","Establishment Year":"1921","time needed to visit in hrs":2.5,"Google review rating":4.6,"Entrance Fee in INR":30,"Airport with 50km Radius":"Yes","Weekly Off":"Monday","Significance":"Historical","DSLR Allowed":"Yes","Number of google review in lakhs":0.9,"Best Time to visit":"Morning"},
    {"Zone":"Western","State":"Maharashtra","City":"Mumbai","Name":"Gateway of India","Type":"Historical","Establishment Year":"1924","time needed to visit in hrs":1.0,"Google review rating":4.5,"Entrance Fee in INR":0,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Historical","DSLR Allowed":"Yes","Number of google review in lakhs":2.1,"Best Time to visit":"Evening"},
    {"Zone":"Northern","State":"Uttar Pradesh","City":"Agra","Name":"Taj Mahal","Type":"Historical","Establishment Year":"1653","time needed to visit in hrs":3.0,"Google review rating":4.9,"Entrance Fee in INR":50,"Airport with 50km Radius":"Yes","Weekly Off":"Friday","Significance":"Historical","DSLR Allowed":"Yes","Number of google review in lakhs":3.5,"Best Time to visit":"Morning"},
    {"Zone":"Northern","State":"Uttar Pradesh","City":"Agra","Name":"Agra Fort","Type":"Historical","Establishment Year":"1573","time needed to visit in hrs":2.5,"Google review rating":4.4,"Entrance Fee in INR":40,"Airport with 50km Radius":"Yes","Weekly Off":"Open Daily","Significance":"Historical","DSLR Allowed":"Yes","Number of google review in lakhs":0.6,"Best Time to visit":"Morning"},
])

# ── CSS ───────────────────────────────────────────────────────────────────────
# Palette: Deep Navy #0D1B2A · Warm Gold #D4A843 · Cream #FBF7F0 · Slate #8A9BB0
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#FBF7F0;}
.block-container{padding:0 2rem 3rem !important;}

/* Sidebar */
section[data-testid="stSidebar"]{background:#0D1B2A !important;border-right:2px solid #1E3A5F !important;}
section[data-testid="stSidebar"] *{color:#E2E8F0 !important;}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"]>div,
section[data-testid="stSidebar"] div[data-baseweb="base-input"]{
  background:#162338 !important;border:1px solid #1E3A5F !important;
  border-radius:8px !important;color:#E2E8F0 !important;}
section[data-testid="stSidebar"] input::placeholder{color:#4A6580 !important;}
section[data-testid="stSidebar"] .stButton>button{
  background:#D4A843 !important;color:#0D1B2A !important;
  border:none;border-radius:10px;font-weight:700;
  padding:.7rem 1rem;width:100%;font-size:.95rem;letter-spacing:.02em;}
section[data-testid="stSidebar"] .stButton>button:hover{background:#C49535 !important;}

/* Hero */
.hero{background:#0D1B2A;padding:4rem 3rem 3.5rem;text-align:center;margin:0 -2rem 2rem;
  border-bottom:3px solid #D4A843;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 65% 55% at 50% 105%,rgba(212,168,67,.1),transparent);pointer-events:none;}
.hero-emoji{font-size:3rem;margin-bottom:.5rem;}
.hero h1{font-family:'DM Serif Display',serif;color:#FBF7F0;font-size:3rem;line-height:1.1;margin:.4rem 0 .8rem;}
.hero h1 em{color:#D4A843;font-style:italic;}
.hero-sub{color:#8A9BB0;font-size:1rem;max-width:520px;margin:0 auto 1.6rem;line-height:1.65;}
.hero-badges{display:flex;flex-wrap:wrap;gap:.4rem;justify-content:center;}
.badge{background:rgba(212,168,67,.12);color:#D4A843;border:1px solid rgba(212,168,67,.25);
  padding:.25rem .7rem;border-radius:999px;font-size:.68rem;font-weight:600;letter-spacing:.05em;}

/* Stat bar */
.stat-bar{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:#E8E0D0;
  border-radius:14px;overflow:hidden;margin-bottom:2rem;}
.stat-cell{background:#fff;padding:1rem .6rem;text-align:center;}
.stat-cell .sv{font-size:1.2rem;font-weight:700;color:#0D1B2A;line-height:1;}
.stat-cell .sk{font-size:.58rem;color:#8A9BB0;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-top:.3rem;}

/* Section header */
.sec-head{display:flex;align-items:center;gap:.75rem;margin:0 0 1.2rem;}
.sec-head h2{font-family:'DM Serif Display',serif;font-size:1.4rem;color:#0D1B2A;margin:0;white-space:nowrap;font-weight:400;}
.sec-line{flex:1;height:1px;background:linear-gradient(90deg,#D4A843,transparent);}

/* Itinerary box */
.itin-box{background:#fff;border-radius:14px;padding:2rem;line-height:1.8;color:#1e293b;
  font-size:.95rem;border:1px solid #E8E0D0;border-top:3px solid #D4A843;}

/* Place card */
.pcard{background:#fff;border-radius:12px;padding:1rem 1.3rem;margin-bottom:.7rem;
  border-left:4px solid #D4A843;border-top:1px solid #EEE8DC;
  border-right:1px solid #EEE8DC;border-bottom:1px solid #EEE8DC;
  display:flex;align-items:flex-start;gap:1rem;}
.pnum{background:#0D1B2A;color:#D4A843;font-weight:700;width:32px;height:32px;
  border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:.85rem;}
.pcard h4{margin:0 0 .35rem;color:#0D1B2A;font-size:.95rem;font-weight:600;}
.tag{display:inline-block;padding:.11rem .48rem;border-radius:999px;font-size:.63rem;font-weight:700;margin:.1rem .12rem .1rem 0;}
.t-type{background:#FEF3C7;color:#78350F;}.t-rat{background:#ECFDF5;color:#064E3B;}
.t-fee{background:#EFF6FF;color:#1E3A8A;}.t-dur{background:#F5F3FF;color:#4C1D95;}
.t-day{background:#0D1B2A;color:#D4A843;}.t-ml{background:#FFF7ED;color:#9A3412;}

/* Route strip */
.route-strip{background:#0D1B2A;border-radius:12px;padding:.9rem 1.4rem;
  display:flex;gap:2.5rem;margin-bottom:1.2rem;align-items:center;
  border-bottom:2px solid #D4A843;}
.route-strip .rv{color:#D4A843;font-size:1.25rem;font-weight:700;}
.route-strip .rk{color:#4A6580;font-size:.62rem;text-transform:uppercase;font-weight:600;letter-spacing:.08em;}

/* Empty state */
.empty{text-align:center;padding:5rem 1rem;color:#8A9BB0;}
.empty .ei{font-size:3.5rem;margin-bottom:1rem;}
.empty h3{color:#4A6580;font-size:1.05rem;margin:0 0 .4rem;}
.empty p{font-size:.88rem;margin:0;}

/* Footer */
.footer{text-align:center;padding:2rem 1rem 1rem;color:#8A9BB0;font-size:.72rem;
  border-top:1px solid #E8E0D0;margin-top:3rem;}
.footer b{color:#D4A843;}

/* Video section */
.vid-box{background:#fff;border-radius:14px;padding:1.5rem;border:1px solid #E8E0D0;border-top:3px solid #D4A843;}
</style>
""", unsafe_allow_html=True)

# ── BACKEND ───────────────────────────────────────────────────────────────────
class DataIngestion:
    def load(self):
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            for c in ["Google review rating","Entrance Fee in INR","time needed to visit in hrs"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            return df
        st.toast("Using sample data — add data/cleaned_places.csv for full dataset.", icon="ℹ️")
        return SAMPLE.copy()
    def to_docs(self, df):
        return [Document(
            page_content=(f"Place:{r['Name']} City:{r['City']} State:{r['State']} Type:{r['Type']} "
                          f"Significance:{r['Significance']} BestTime:{r['Best Time to visit']} "
                          f"Duration:{r['time needed to visit in hrs']}hrs Rating:{r['Google review rating']} "
                          f"Fee:{r['Entrance Fee in INR']}INR"),
            metadata={"name":str(r["Name"]),"city":str(r["City"]),"state":str(r["State"]),
                      "type":str(r["Type"]),"rating":float(r["Google review rating"]),
                      "duration":float(r["time needed to visit in hrs"]),
                      "fee":float(r["Entrance Fee in INR"]),"best_time":str(r["Best Time to visit"])})
            for _, r in df.iterrows()]

class EmbeddingModel:
    def __init__(self): self.m = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    def embed_docs(self, docs): return self.m.embed_documents([d.page_content for d in docs])
    def embed_query(self, q):   return self.m.embed_query(q)

class VectorStore:
    def __init__(self):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.col = chromadb.PersistentClient(path=CHROMA_DIR).get_or_create_collection("tourist_places")
    def empty(self): return self.col.count() == 0
    def add(self, docs, embs):
        self.col.add(ids=[str(uuid.uuid4()) for _ in docs],
                     documents=[d.page_content for d in docs],
                     embeddings=embs, metadatas=[d.metadata for d in docs])
    def search(self, emb, k=15):
        k = min(k, max(self.col.count(), 1))
        return self.col.query(query_embeddings=[emb], n_results=k)

class RAGPipeline:
    _PROMPT = ChatPromptTemplate.from_template("""You are ItinerAI, an expert Indian travel planner.
Generate a day-wise itinerary using ONLY the places listed below. Never invent places.
Format: **Day N — Theme** header, then bullet points: • Place (Xh) — one-line reason.
End with a 3-bullet Travel Tips section (best season, transport, food tip).
Retrieved places:\n{context}\n\nRequest: {question}""")
    def __init__(self, vs, em):
        self.vs, self.em = vs, em
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    def ask(self, q, k=15):
        r = self.vs.search(self.em.embed_query(q), k)
        ctx = "\n\n".join(r["documents"][0])
        return {"answer": self.llm.invoke(self._PROMPT.format(context=ctx, question=q)).content,
                "metadata": r["metadatas"][0]}

class Geocoder:
    def __init__(self): self.s = requests.Session(); self.s.headers["User-Agent"] = "ItinerAI/2.0"
    def get(self, name, city=""):
        try:
            d = self.s.get("https://nominatim.openstreetmap.org/search",
                           params={"q":f"{name},{city},India","format":"json","limit":1},timeout=6).json()
            if d: return float(d[0]["lat"]), float(d[0]["lon"])
        except: pass
        return None, None

class MLRanker:
    def __init__(self): self.model = joblib.load(MODEL_PATH) if Path(MODEL_PATH).exists() else None
    def score(self, places):
        if not self.model: return [{**p,"ml_score":None} for p in places]
        rows = pd.DataFrame([{"Google review rating":float(p.get("rating",0) or 0),
            "time needed to visit in hrs":float(p.get("duration",0) or 0),
            "Entrance Fee in INR":float(p.get("fee",0) or 0),
            "Type":p.get("type","?"),"City":p.get("city","?"),"State":p.get("state","?")} for p in places])
        try:
            preds = self.model.predict(rows)
            return [{**p,"ml_score":round(float(s),4)} for p,s in zip(places,preds)]
        except: return [{**p,"ml_score":None} for p in places]

class OSRMRouter:
    def route(self, a, b):
        try:
            d=requests.get(f"https://router.project-osrm.org/route/v1/driving/{a[1]},{a[0]};{b[1]},{b[0]}",
                           params={"overview":"full","geometries":"geojson"},timeout=8).json()
            if d.get("code")=="Ok":
                rt=d["routes"][0]; return {"km":round(rt["distance"]/1000,1),"min":round(rt["duration"]/60,0),"coords":rt["geometry"]["coordinates"]}
        except: pass
        return None

def recommend(places, city, budget, interest, days):
    r = [p for p in places if p.get("city","").lower()==city.lower()]
    if interest and interest!="Any": r=[p for p in r if interest.lower() in p.get("type","").lower()]
    if budget and budget<100000:     r=[p for p in r if float(p.get("fee",0) or 0)<=budget]
    r=[p for p in r if isinstance(p.get("duration"),(int,float))]
    return r[:days*3]

def build_map(places, segs):
    valid=[p for p in places if p.get("lat") and p.get("lon")]
    if not valid: return None
    m=folium.Map([valid[0]["lat"],valid[0]["lon"]],zoom_start=12,tiles="cartodbpositron")
    colors=["#5b6ef5","#8b5cf6","#ec4899","#10b981","#f59e0b","#ef4444","#14b8a6"]
    for i,p in enumerate(valid,1):
        c=colors[(i-1)%len(colors)]
        folium.CircleMarker([p["lat"],p["lon"]],radius=13,color="#1a1a2e",weight=2,fill=True,fill_color=c,fill_opacity=.9,
            popup=folium.Popup(f"<b>{i}. {p['name']}</b><br>⭐{p.get('rating','?')} · ₹{int(float(p.get('fee',0) or 0))} · {p.get('duration','?')}h",max_width=200),
            tooltip=f"{i}. {p['name']}").add_to(m)
        folium.Marker([p["lat"],p["lon"]],icon=folium.DivIcon(
            html=f'<div style="color:#fff;font-weight:800;font-size:9px;text-align:center;margin-top:5px">{i}</div>',
            icon_size=(26,26),icon_anchor=(13,13))).add_to(m)
    for s in segs:
        if s: folium.PolyLine([[lat,lon] for lon,lat in s["coords"]],color="#5b6ef5",weight=4,opacity=.8,dash_array="6 4").add_to(m)
    m.fit_bounds([[p["lat"],p["lon"]] for p in valid])
    return m

def generate_video(places, city, output_path="videos/travel_video.mp4"):
    # Guard: need at least 1 place
    places = [p for p in places if p.get("name")]
    if not places:
        return None, "No places available to render. Generate an itinerary with matched places first."
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from moviepy import ImageClip, concatenate_videoclips
        os.makedirs("videos", exist_ok=True)
        W, H = 1280, 720
        # Navy/gold palette matching the app
        NAV  = (13, 27, 42)       # #0D1B2A
        GOLD = (212, 168, 67)     # #D4A843
        GOLD_DIM = (160, 124, 44) # dimmer gold for divider
        SLATE = (138, 155, 176)   # #8A9BB0
        WHITE = (251, 247, 240)   # #FBF7F0
        clips = []
        for i, p in enumerate(places, 1):
            img = Image.new("RGB", (W, H), color=NAV)
            draw = ImageDraw.Draw(img)
            # Gold left accent bar
            draw.rectangle([0, 0, 6, H], fill=GOLD)
            # Bottom gold rule
            draw.line([(0, H-70),(W, H-70)], fill=GOLD_DIM, width=1)
            try:
                fb = ImageFont.truetype("arial.ttf", 62)
                fm = ImageFont.truetype("arial.ttf", 30)
                fs = ImageFont.truetype("arial.ttf", 20)
            except:
                fb = fm = fs = ImageFont.load_default()
            # Day badge (gold pill)
            draw.rounded_rectangle([36, 28, 170, 72], radius=10, fill=GOLD)
            draw.text((103, 50), f"DAY  {p.get('day',1)}", font=fs, fill=NAV, anchor="mm")
            # Place number circle (gold outline)
            draw.ellipse([W-108, 18, W-42, 84], outline=GOLD, width=2)
            draw.text((W-75, 51), f"#{i}", font=fm, fill=GOLD, anchor="mm")
            # Place name
            name = p.get("name","?")
            draw.text((48, H//2-60), name, font=fb, fill=WHITE)
            # City
            draw.text((48, H//2+30), f"  {p.get('city', city)}", font=fm, fill=SLATE)
            # Rating + fee
            rating = p.get("rating","N/A")
            fee = int(float(p.get("fee", 0) or 0))
            draw.text((48, H//2+76), f"  {rating}     Rs.{fee} entry", font=fm, fill=GOLD)
            # Branding
            draw.text((W//2, H-36), "ItinerAI  —  AI Travel Planner", font=fs, fill=GOLD_DIM, anchor="mm")
            clips.append(ImageClip(np.array(img)).with_duration(3))
        video = concatenate_videoclips(clips, method="compose")
        video.write_videofile(output_path, fps=24, codec="libx264", logger=None)
        return output_path
    except Exception as e:
        return None, str(e)

# ── DB ────────────────────────────────────────────────────────────────────────
def init_db():
    c=sqlite3.connect(DB_PATH); c.execute("CREATE TABLE IF NOT EXISTS h(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT,city TEXT,days INT,budget INT,interest TEXT,itin TEXT)"); c.commit(); c.close()
def save_h(city,days,budget,interest,itin):
    c=sqlite3.connect(DB_PATH); c.execute("INSERT INTO h(ts,city,days,budget,interest,itin) VALUES(?,?,?,?,?,?)",(datetime.now().strftime("%d %b %H:%M"),city,days,budget,interest,itin)); c.commit(); c.close()
def load_h():
    try: c=sqlite3.connect(DB_PATH); df=pd.read_sql("SELECT * FROM h ORDER BY id DESC LIMIT 10",c); c.close(); return df
    except: return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_pipeline():
    di=DataIngestion(); df=di.load(); em=EmbeddingModel(); vs=VectorStore()
    if vs.empty():
        docs=di.to_docs(df); vs.add(docs, em.embed_docs(docs))
    return RAGPipeline(vs,em), MLRanker(), OSRMRouter(), Geocoder()

init_db()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-emoji">🌴</div>
  <h1>Plan your dream trip<br>with <em>ItinerAI</em></h1>
  <p class="hero-sub">Describe where you want to go. Get a personalized day-wise itinerary,
  an interactive route map, and AI-ranked places — instantly.</p>
  <div class="hero-badges">
    <span class="badge">RAG · LangChain</span>
    <span class="badge">Llama 3.3 70B</span>
    <span class="badge">XGBoost Ranking</span>
    <span class="badge">OSRM Routing</span>
    <span class="badge">ChromaDB</span>
    <span class="badge">Folium Maps</span>
    <span class="badge">Video Export</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Plan Your Trip")
    st.caption("Fill in your preferences below")
    st.divider()
    city     = st.text_input("Destination City", placeholder="Jaipur, Delhi, Mysore, Goa…")
    days     = st.slider("Trip Duration (days)", 1, 7, 3)
    budget   = st.selectbox("Budget per Attraction",
                 [200,500,1000,2000,5000,100000], index=3,
                 format_func=lambda x: f"Up to ₹{x:,}" if x<100000 else "No limit")
    interest = st.selectbox("Primary Interest",
                 ["Any","Historical","Religious","Nature","Beach",
                  "Wildlife","Museum","Recreational","War Memorial","Tomb","Lake"])
    st.divider()
    generate = st.button("✨ Generate Itinerary", use_container_width=True)
    st.divider()
    st.markdown("**🕓 Recent Trips**")
    hdf = load_h()
    if not hdf.empty:
        for _, row in hdf.head(5).iterrows():
            st.caption(f"📍 **{row['city']}** · {row['days']}d · {row['ts']}")
    else:
        st.caption("No trips yet.")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📝 Itinerary", "🗺️ Route Map", "📊 Ranked Places", "🎬 Video"])

# ── GENERATE ──────────────────────────────────────────────────────────────────
if generate:
    city = city.strip()
    if not city:
        st.warning("Please enter a destination city.")
    else:
        with st.spinner("Loading AI pipeline…"):
            rag, ranker, router, geo = load_pipeline()
        with st.spinner(f"Generating itinerary for **{city}** with Llama 3.3 70B…"):
            try:
                q = f"{days}-day trip to {city}, interest:{interest}, budget-friendly"
                res = rag.ask(q, k=15)
            except Exception as e:
                st.error(f"LLM error: {e}"); st.stop()
        recs = recommend(res["metadata"], city, budget if budget<100000 else None, interest, days)
        recs = ranker.score(recs)
        recs.sort(key=lambda x:(x.get("ml_score") is None, -(x.get("ml_score") or 0)))
        with st.spinner("Geocoding places & calculating routes…"):
            for i, p in enumerate(recs):
                lat, lon = geo.get(p.get("name",""), p.get("city",""))
                p["lat"], p["lon"], p["day"] = lat, lon, (i//3)+1
                time.sleep(1.1)
            segs=[]
            for i in range(len(recs)-1):
                a,b=recs[i],recs[i+1]
                segs.append(router.route((a.get("lat"),a.get("lon")),(b.get("lat"),b.get("lon")))
                            if a.get("lat") and b.get("lat") else None)
        save_h(city, days, budget, interest, res["answer"])
        st.session_state.update({"res":res,"recs":recs,"segs":segs,"city":city,"days":days,"video_path":None})
        st.rerun()

# ── TAB 1 — ITINERARY ─────────────────────────────────────────────────────────
with tab1:
    if "res" in st.session_state:
        recs=st.session_state["recs"]; city_=st.session_state["city"]; days_=st.session_state["days"]
        total_fee=sum(float(p.get("fee",0)or 0) for p in recs)
        total_hrs=sum(float(p.get("duration",0)or 0) for p in recs)
        avg_rat  =(sum(float(p.get("rating",0)or 0) for p in recs)/len(recs)) if recs else 0
        st.markdown(f"""
        <div class="stat-bar">
          <div class="stat-cell"><div class="sv">🏙️</div><div class="sk">{city_}</div></div>
          <div class="stat-cell"><div class="sv">📅 {days_}d</div><div class="sk">Duration</div></div>
          <div class="stat-cell"><div class="sv">📍 {len(recs)}</div><div class="sk">Places</div></div>
          <div class="stat-cell"><div class="sv">⭐ {avg_rat:.1f}</div><div class="sk">Avg Rating</div></div>
          <div class="stat-cell"><div class="sv">₹{int(total_fee):,}</div><div class="sk">Total Fees</div></div>
          <div class="stat-cell"><div class="sv">⏱ {total_hrs:.0f}h</div><div class="sk">Visit Time</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><h2>Your Itinerary</h2><div class="sec-line"></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="itin-box">{st.session_state["res"]["answer"].replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""<div class="empty"><div class="ei">🧳</div><h3>No itinerary yet</h3>
        <p>Enter a destination and click <b>✨ Generate Itinerary</b></p></div>""", unsafe_allow_html=True)

# ── TAB 2 — MAP ───────────────────────────────────────────────────────────────
with tab2:
    if "recs" in st.session_state:
        recs=st.session_state["recs"]; segs=st.session_state["segs"]
        st.markdown('<div class="sec-head"><h2>Interactive Route Map</h2><div class="sec-line"></div></div>', unsafe_allow_html=True)
        valid_segs=[s for s in segs if s]
        if valid_segs:
            total_km=sum(s["km"] for s in valid_segs); total_min=sum(s["min"] for s in valid_segs)
            st.markdown(f"""<div class="route-strip">
              <div><div class="rv">{total_km:.1f} km</div><div class="rk">Total Distance</div></div>
              <div><div class="rv">{int(total_min)} min</div><div class="rk">Drive Time</div></div>
              <div><div class="rv">{len(recs)}</div><div class="rk">Stops</div></div>
            </div>""", unsafe_allow_html=True)
        m = build_map(recs, segs)
        if m: st_folium(m, width=None, height=520, returned_objects=[])
        else: st.info("No coordinates found. Try a different city or check your connection.")
    else:
        st.markdown("""<div class="empty"><div class="ei">🗺️</div><h3>Map not generated yet</h3>
        <p>Generate an itinerary first to see your route here.</p></div>""", unsafe_allow_html=True)

# ── TAB 3 — PLACES ────────────────────────────────────────────────────────────
with tab3:
    if "recs" in st.session_state:
        recs=st.session_state["recs"]
        st.markdown('<div class="sec-head"><h2>Matched & Ranked Places</h2><div class="sec-line"></div></div>', unsafe_allow_html=True)
        if not recs:
            st.info("No places matched your filters. Try relaxing the budget or interest.")
        for i,p in enumerate(recs,1):
            ml=p.get("ml_score"); s_ml=f'<span class="tag t-ml">🤖 {ml:.3f}</span>' if ml else ""
            lat=p.get("lat"); s_ll=f'<span class="tag t-fee">📍 {lat:.3f},{p["lon"]:.3f}</span>' if lat else ""
            st.markdown(f"""
            <div class="pcard">
              <div class="pnum">{i}</div>
              <div class="place-body">
                <h4>{p.get("name","?")} — {p.get("city","?")}, {p.get("state","?")}</h4>
                <span class="tag t-day">Day {p.get("day","?")}</span>
                <span class="tag t-type">{p.get("type","N/A")}</span>
                <span class="tag t-rat">⭐ {p.get("rating","N/A")}</span>
                <span class="tag t-fee">₹{int(float(p.get("fee",0)or 0)):,}</span>
                <span class="tag t-dur">⏱ {p.get("duration","N/A")}h</span>
                <span class="tag t-type">{p.get("best_time","")}</span>
                {s_ml}{s_ll}
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="empty"><div class="ei">📊</div><h3>No places yet</h3>
        <p>Generate an itinerary to see XGBoost-ranked place cards.</p></div>""", unsafe_allow_html=True)

# ── TAB 4 — VIDEO ─────────────────────────────────────────────────────────────
with tab4:
    if "res" in st.session_state:
        city_ = st.session_state["city"]
        recs  = st.session_state["recs"]
        # Use all RAG metadata as fallback if city filter returned no places
        vid_places = recs if recs else st.session_state["res"]["metadata"]
        # Assign day numbers if missing (fallback list won't have them)
        for idx, p in enumerate(vid_places):
            if "day" not in p:
                p["day"] = (idx // 3) + 1

        st.markdown('<div class="sec-head"><h2>Trip Highlight Video</h2><div class="sec-line"></div></div>', unsafe_allow_html=True)

        vid_path = st.session_state.get("video_path")

        if vid_path and Path(vid_path).exists():
            st.video(vid_path)
            n = len(vid_places)
            st.caption(f"🎬 {n} place{'s' if n!=1 else ''} · {n*3}s · {city_}")
            col_dl, col_re = st.columns([1,1])
            with col_dl:
                with open(vid_path, "rb") as f:
                    st.download_button("⬇️ Download MP4", f,
                                       file_name=f"itinerai_{city_.lower().replace(' ','_')}.mp4",
                                       mime="video/mp4", use_container_width=True)
            with col_re:
                if st.button("🔄 Regenerate", use_container_width=True):
                    st.session_state["video_path"] = None
                    st.rerun()
        else:
            st.markdown('<div class="vid-box">', unsafe_allow_html=True)
            n = len(vid_places)
            if not recs:
                st.info(f"No places matched your city filter — video will use the {n} closest RAG results instead.")
            st.markdown("#### 🎬 Generate a trip highlight video")
            st.caption(f"Renders a {n*3}s MP4 ({n} place{'s' if n!=1 else ''}) · plays directly in the browser, no download needed.")
            col_info, col_btn = st.columns([3, 2])
            with col_info:
                st.markdown("""
**Each card shows:**
- Place name & city
- Day number badge
- Google rating & entry fee
- ItinerAI branding
                """)
            with col_btn:
                st.write("")
                if st.button("▶️ Generate & Preview Video", use_container_width=True):
                    with st.spinner(f"Rendering {n} cards… (~{n*2}s)"):
                        result = generate_video(vid_places, city_)
                    if isinstance(result, str) and Path(result).exists():
                        st.session_state["video_path"] = result
                        st.rerun()
                    else:
                        err = result[1] if isinstance(result, tuple) else str(result)
                        st.error(f"Generation failed: {err}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('''<div class="empty"><div class="ei">🎬</div><h3>No video yet</h3>
        <p>Generate an itinerary first, then come back here to preview your trip video.</p></div>''', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built by <b>Aravinth B</b> · ECE, SASTRA University ·
  Stack: LangChain · ChromaDB · ChatGroq · XGBoost · OSRM · Folium · MoviePy · Streamlit
</div>""", unsafe_allow_html=True)