
        import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import datetime
import warnings
warnings.filterwarnings('ignore')

from deep_translator import GoogleTranslator
from langdetect  import detect

st.set_page_config(
    page_title = "HealthBridge",
    page_icon  = "🏥",
    layout     = "centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    color: #1a5276;
    margin-bottom: 0rem;
}
.sub-title {
    text-align: center;
    font-size: 1rem;
    color: #5d6d7e;
    margin-bottom: 1.5rem;
}
.badge {
    display: inline-block;
    background: #eaf4fb;
    color: #1a5276;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 2px;
    font-weight: 600;
}
.section-card {
    background: #f8f9fa;
    border-left: 4px solid #1a5276;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.emergency-box {
    background: #fdecea;
    border-left: 5px solid #e74c3c;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.urgent-box {
    background: #fef9e7;
    border-left: 5px solid #f39c12;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.mild-box {
    background: #eafaf1;
    border-left: 5px solid #27ae60;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.hospital-card {
    background: #ffffff;
    border: 1px solid #d5d8dc;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.disease-bar-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #1a5276;
}
.tip-item {
    padding: 4px 0;
    font-size: 0.93rem;
}
.footer-note {
    text-align: center;
    font-size: 0.8rem;
    color: #aab7b8;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    rf_model        = joblib.load("healthbridge_model.pkl")
    le              = joblib.load("label_encoder.pkl")
    symptom_columns = joblib.load("symptom_columns.pkl")
    return rf_model, le, symptom_columns

@st.cache_data
def load_datasets():
    desc_df      = pd.read_csv("symptom_Description.csv")
    prec_df      = pd.read_csv("symptom_precaution.csv")
    sev_df       = pd.read_csv("Symptom-severity.csv")
    hospitals_df = pd.read_csv("tamil_nadu_hospitals.csv")
    desc_df.columns      = desc_df.columns.str.strip()
    prec_df.columns      = prec_df.columns.str.strip()
    sev_df.columns       = sev_df.columns.str.strip()
    return desc_df, prec_df, sev_df, hospitals_df

try:
    rf_model, le, symptom_columns = load_models()
    desc_df, prec_df, sev_df, hospitals_df = load_datasets()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

translator = GoogleTranslator(source="ta", target="en")

tamil_symptom_map = {
    "காய்ச்சல்": "high_fever", "kaichal": "high_fever", "kaayshal": "high_fever",
    "juram": "high_fever", "adhika kaichal": "high_fever", "அதிக காய்ச்சல்": "high_fever",
    "manda kaichal": "mild_fever", "தலைவலி": "headache", "thalai vali": "headache",
    "talaivali": "headache", "head pain": "headache", "வாந்தி": "vomiting",
    "vaanthi": "vomiting", "vandhi": "vomiting", "குமட்டல்": "nausea",
    "kumattal": "nausea", "சோர்வு": "fatigue", "sorvu": "fatigue",
    "tiredness": "fatigue", "weakness": "fatigue", "உடல் வலி": "muscle_pain",
    "udal vali": "muscle_pain", "body pain": "muscle_pain", "body ache": "muscle_pain",
    "மூட்டு வலி": "joint_pain", "moottu vali": "joint_pain", "joint pain": "joint_pain",
    "நடுக்கம்": "chills", "nadukkam": "chills", "வியர்வை": "sweating",
    "viyarvai": "sweating", "இருமல்": "cough", "irumal": "cough",
    "மூச்சு திணறல்": "breathlessness", "moochu thinaral": "breathlessness",
    "breathing problem": "breathlessness", "மார்பு வலி": "chest_pain",
    "maarbu vali": "chest_pain", "chest pain": "chest_pain",
    "வயிற்று வலி": "stomach_pain", "vayiru vali": "stomach_pain",
    "stomach pain": "stomach_pain", "வயிற்றுப்போக்கு": "diarrhoea",
    "vayiru pokku": "diarrhoea", "loose motion": "diarrhoea", "loose motions": "diarrhoea",
    "மலச்சிக்கல்": "constipation", "malasikkal": "constipation",
    "தோல் தடிப்பு": "skin_rash", "tol thadippu": "skin_rash", "rash": "skin_rash",
    "அரிப்பு": "itching", "arippu": "itching", "itchy": "itching",
    "எடை இழப்பு": "weight_loss", "edai izhapu": "weight_loss", "weight loss": "weight_loss",
    "பசியின்மை": "loss_of_appetite", "pasi illaamai": "loss_of_appetite",
    "no appetite": "loss_of_appetite", "மஞ்சள் காமாலை": "yellowish_skin",
    "manjal kamalai": "yellowish_skin", "yellow skin": "yellowish_skin",
    "yellow eyes": "yellowing_of_eyes", "மங்கலான பார்வை": "blurred_and_distorted_vision",
    "mangalana parvai": "blurred_and_distorted_vision", "blurred vision": "blurred_and_distorted_vision",
    "படபடப்பு": "restlessness", "padapadappu": "restlessness",
    "முதுகு வலி": "back_pain", "mudugu vali": "back_pain", "back pain": "back_pain",
    "தலைசுற்றல்": "dizziness", "thalai sutral": "dizziness", "dizzy": "dizziness",
    "தாகம்": "dehydration", "thaagam": "dehydration", "very thirsty": "dehydration",
    "கழுத்து வலி": "neck_pain", "kazhuththu vali": "neck_pain", "neck pain": "neck_pain",
}

translation_fixes = {
    "trembling": "chills", "shaking": "chills", "shiver": "chills",
    "vomit": "vomiting", "fever": "high fever", "temperature": "high fever",
    "stomach ache": "stomach pain", "belly ache": "stomach pain",
    "loose stool": "diarrhoea", "loose stools": "diarrhoea",
    "perspiration": "sweating", "tired": "fatigue",
    "no hunger": "loss of appetite", "breathing problem": "breathlessness",
    "shortness of breath": "breathlessness", "blurry vision": "blurred and distorted vision",
    "joint ache": "joint pain", "muscle ache": "muscle pain",
    "back ache": "back pain", "head ache": "headache",
}

severity_dict = {}
if models_loaded:
    for _, row in sev_df.iterrows():
        symptom = str(row[sev_df.columns[0]]).strip().lower().replace(' ', '_')
        weight  = int(row[sev_df.columns[1]])
        severity_dict[symptom] = weight

critical_diseases = [
    "Malaria","Dengue","Typhoid","Tuberculosis","Hepatitis B","Hepatitis C",
    "Hepatitis D","Hepatitis E","hepatitis A","Jaundice","Diabetes","Pneumonia",
    "Heart attack","Paralysis (brain hemorrhage)","Drug Reaction","AIDS",
    "Chronic cholestasis","Alcoholic hepatitis",
]
urgent_diseases = [
    "Chicken pox","Urinary tract infection","Gastroenteritis","Migraine",
    "Allergy","Arthritis","Bronchial Asthma","Hypertension","Hypothyroidism",
    "Hyperthyroidism","Hypoglycemia","Osteoarthristis","Cervical spondylosis",
    "Varicose veins","Psoriasis","Fungal infection","Peptic ulcer diseae",
    "Dimorphic hemmorhoids(piles)","(vertigo) Paroymsal  Positional Vertigo",
]
mild_diseases = ["Common Cold","Acne","Impetigo","GERD"]

seasonal_diseases = {
    "Malaria"        : [6,7,8,9,10],
    "Dengue"         : [7,8,9,10,11],
    "Typhoid"        : [5,6,7,8,9],
    "Gastroenteritis": [5,6,7,8],
    "Common Cold"    : [11,12,1,2],
    "Pneumonia"      : [11,12,1,2],
    "Allergy"        : [2,3,4],
    "Chicken pox"    : [2,3,4,5],
}

urgency_tamil = {
    "EMERGENCY": ("🔴", "அவசரம் (EMERGENCY)", "உடனே மருத்துவமனை செல்லுங்கள்!", "Go to hospital RIGHT NOW. Do not wait.", "emergency-box"),
    "URGENT"   : ("🟡", "கவனம் தேவை (URGENT)", "இன்றே மருத்துவரை சந்தியுங்கள்", "Visit a doctor or health centre TODAY.", "urgent-box"),
    "MILD"     : ("🟢", "சாதாரணம் (MILD)", "வீட்டில் ஓய்வு எடுங்கள்", "Rest at home. Visit doctor if symptoms worsen.", "mild-box"),
}

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def normalize_input(user_text):
    text = user_text.strip().lower()
    lang = detect_language(text)
    if lang == 'ta':
        try:
            text = translator.translate(text).lower()
        except:
            pass
    for wrong, correct in translation_fixes.items():
        if wrong in text:
            text = text.replace(wrong, correct)
    for tamil_word, english_word in tamil_symptom_map.items():
        if tamil_word.lower() in text:
            text = text.replace(tamil_word.lower(), english_word.replace('_', ' '))
    return text

def extract_symptoms(user_text):
    normalized   = normalize_input(user_text)
    input_vector = pd.DataFrame([{col: 0 for col in symptom_columns}])
    matched = []
    for col in symptom_columns:
        readable = col.replace('_', ' ').lower()
        if readable in normalized or col.lower() in normalized:
            input_vector[col] = 1
            matched.append(col)
    return input_vector, matched, normalized

def get_severity_score(matched_symptoms):
    return sum(severity_dict.get(s, 3) for s in matched_symptoms)

def get_urgency_from_severity(score, num_symptoms):
    avg = score / max(num_symptoms, 1)
    if avg >= 5 or score >= 20:
        return "EMERGENCY"
    elif avg >= 3 or score >= 10:
        return "URGENT"
    return "MILD"

def get_urgency(disease_name, matched_symptoms, days):
    if disease_name in critical_diseases:
        base_level = "EMERGENCY"
    elif disease_name in urgent_diseases:
        base_level = "URGENT"
    elif disease_name in mild_diseases:
        base_level = "MILD"
    else:
        score      = get_severity_score(matched_symptoms)
        base_level = get_urgency_from_severity(score, len(matched_symptoms))
    final_level = base_level
    if base_level == "MILD"   and days > 3:
        final_level = "URGENT"
    if base_level == "URGENT" and days > 5:
        final_level = "EMERGENCY"
    return final_level

def get_first_aid(disease_name):
    row = prec_df[prec_df.iloc[:, 0].str.strip().str.lower() == disease_name.strip().lower()]
    if len(row) == 0:
        return ["Rest and stay hydrated.", "Avoid self-medication.",
                "Visit the nearest health centre as soon as possible."]
    tips = []
    for col in prec_df.columns[1:]:
        tip = str(row.iloc[0][col]).strip()
        if tip and tip.lower() != 'nan':
            tips.append(tip.capitalize())
    return tips if tips else ["Rest and visit nearest health centre."]

def get_disease_description(disease_name):
    row = desc_df[desc_df.iloc[:, 0].str.strip().str.lower() == disease_name.strip().lower()]
    if len(row) == 0:
        return "Please consult a doctor for proper diagnosis."
    return str(row.iloc[0].iloc[1]).strip()

def haversine_distance(lat1, lon1, lat2, lon2):
    R     = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a     = (math.sin(d_lat/2)**2 +
             math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) *
             math.sin(d_lon/2)**2)
    return round(6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 2)

def find_nearest_hospitals(user_lat, user_lon, top_n=3):
    hospitals_df['distance_km'] = hospitals_df.apply(
        lambda row: haversine_distance(user_lat, user_lon, row['latitude'], row['longitude']), axis=1
    )
    nearest = hospitals_df.sort_values('distance_km').head(top_n)
    results = []
    for _, row in nearest.iterrows():
        results.append({
            "name"       : row['hospital_name'],
            "type"       : row['type'],
            "district"   : row['district'],
            "distance_km": row['distance_km'],
            "maps_link"  : f"https://maps.google.com/?q={row['latitude']},{row['longitude']}"
        })
    return results


st.markdown('<div class="main-title">🏥 HealthBridge</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Symptom Checker & Hospital Navigator for Rural Tamil Nadu</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-bottom:1.5rem;">
  <span class="badge">🗣️ Tamil</span>
  <span class="badge">✍️ Tanglish</span>
  <span class="badge">🔤 English</span>
  <span class="badge">🏥 42 Diseases</span>
  <span class="badge">💊 132 Symptoms</span>
  <span class="badge">📍 GPS Navigation</span>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"Model files not found. Please make sure all .pkl and .csv files are in the same folder as app.py.\n\nError: {load_error}")
    st.stop()

st.markdown("### 📝 Enter Your Symptoms")
st.markdown("Type in **Tamil** (காய்ச்சல்), **Tanglish** (kaichal thalai vali), or **English** (fever headache)")

user_text = st.text_area(
    label       = "Symptoms",
    placeholder = "e.g.  kaichal thalai vali vaanthi   OR   fever headache vomiting   OR   காய்ச்சல் தலைவலி",
    height      = 100,
    label_visibility = "collapsed"
)

col1, col2 = st.columns(2)
with col1:
    days = st.number_input("How many days have you had these symptoms?", min_value=1, max_value=30, value=2)
with col2:
    use_location = st.checkbox("Find nearest hospitals (enter GPS coordinates)", value=False)

user_lat = user_lon = None
search_radius = 5
hospital_types = []
if use_location:
    col3, col4 = st.columns(2)
    with col3:
        user_lat = st.number_input("Latitude",  value=13.0827, format="%.4f")
    with col4:
        user_lon = st.number_input("Longitude", value=80.2707, format="%.4f")

    st.markdown("**🔘 Search Radius**")
    search_radius = st.radio(
        "Select radius",
        options=[1, 3, 5],
        format_func=lambda x: f"{x} km — {'Very nearby' if x==1 else 'Nearby' if x==3 else 'Wider area'}",
        horizontal=True,
        label_visibility="collapsed"
    )

    hospital_types = ["District Hospital", "PHC", "CHC", "Government"]

predict_btn = st.button("🔍 Check My Symptoms", use_container_width=True, type="primary")

if predict_btn:
    if not user_text.strip():
        st.warning("Please enter your symptoms first.")
    else:
        with st.spinner("Analysing your symptoms..."):

            input_vector, matched, normalized = extract_symptoms(user_text)

            if not matched:
                st.error("No symptoms detected. Please try again.")
                st.info("Example: **kaichal thalai vali** or **fever headache vomiting** or **காய்ச்சல் தலைவலி**")
                st.stop()

            if len(matched) < 5:
                st.warning(f"⚠️ You entered only **{len(matched)} symptom(s)**. Please enter **at least 5 symptoms** for accurate prediction.")
                st.info(f"✅ Symptoms detected so far: **{', '.join([s.replace('_',' ') for s in matched])}**\n\nAdd more symptoms like: headache, vomiting, chills, fatigue, body pain, sweating, nausea, dizziness")
                st.stop()

            probs    = rf_model.predict_proba(input_vector)[0]
            top5_idx = np.argsort(probs)[::-1][:5]
            raw_preds = {le.classes_[i]: round(float(probs[i]), 3) for i in top5_idx}

            month   = datetime.datetime.now().month
            boosted = {}
            for disease, prob in raw_preds.items():
                if disease in seasonal_diseases and month in seasonal_diseases[disease]:
                    boosted[disease] = min(prob + 0.15, 1.0)
                else:
                    boosted[disease] = prob
            total     = sum(boosted.values())
            adj_preds = dict(sorted(
                {d: round(p/total, 3) for d, p in boosted.items()}.items(),
                key=lambda x: x[1], reverse=True
            ))

            top_disease = list(adj_preds.keys())[0]
            top_confidence = list(adj_preds.values())[0]

            if top_confidence < 0.50:
                st.warning(f"⚠️ Low confidence prediction — **{int(top_confidence*100)}%**. Please enter more specific symptoms for better accuracy.")
                st.info("Add more symptoms to improve prediction accuracy.")

            urgency_level = get_urgency(top_disease, matched, days)
            icon, tamil_label, tamil_msg, eng_msg, box_class = urgency_tamil[urgency_level]
            tips        = get_first_aid(top_disease)
            description = get_disease_description(top_disease)

        st.markdown("---")
        st.markdown("## 📋 Your Health Report")

        st.markdown("#### 🦠 Predicted Disease")
        top_list = list(adj_preds.items())[:3]
        for rank, (disease, prob) in enumerate(top_list, 1):
            pct = int(prob * 100)
            label = f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉'} {disease}"
            st.progress(pct, text=f"{label} — **{pct}%**")

        st.markdown(f"#### 📖 About {top_disease}")
        st.markdown(f'<div class="section-card">{description}</div>', unsafe_allow_html=True)

        st.markdown("#### ⚡ Urgency Level")
        st.markdown(f"""
        <div class="{box_class}">
            <div style="font-size:1.4rem; font-weight:700;">{icon} {tamil_label}</div>
            <div style="font-size:1.05rem; color:#555; margin-top:4px;">{tamil_msg}</div>
            <div style="font-size:0.95rem; color:#777; margin-top:2px;">{eng_msg}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🩹 First Aid Steps")
        tips_html = "".join([f'<div class="tip-item">✅ {tip}</div>' for tip in tips])
        st.markdown(f'<div class="section-card">{tips_html}</div>', unsafe_allow_html=True)

        st.markdown("#### 🔍 Symptoms Detected")
        sym_badges = " ".join([f'<span class="badge">{s.replace("_"," ")}</span>' for s in matched])
        st.markdown(sym_badges, unsafe_allow_html=True)

        if use_location and user_lat and user_lon:
            st.markdown("#### 🗺️ Nearest Hospitals")

            hospitals_df['distance_km'] = hospitals_df.apply(
                lambda row: haversine_distance(user_lat, user_lon, row['latitude'], row['longitude']), axis=1
            )

            filtered = hospitals_df[hospitals_df['distance_km'] <= search_radius]

            if hospital_types:
                type_filter = hospitals_df['type'].str.contains('|'.join(hospital_types), case=False, na=False)
                filtered = hospitals_df[type_filter & (hospitals_df['distance_km'] <= search_radius)]

            filtered = filtered.sort_values('distance_km').head(3)

            if len(filtered) == 0:
                st.warning(f"⚠️ No hospitals found within {search_radius} km. Try increasing the search radius.")
                nearest = hospitals_df.sort_values('distance_km').head(3)
                st.info("Showing 3 nearest hospitals from Tamil Nadu dataset:")
                for i, (_, row) in enumerate(nearest.iterrows(), 1):
                    maps_link = f"https://maps.google.com/?q={row['latitude']},{row['longitude']}"
                    st.markdown(f"""
                    <div class="hospital-card">
                        <b>{i}. {row['hospital_name']}</b><br>
                        <span style="color:#5d6d7e;">🏥 {row['type']} &nbsp;|&nbsp; 📍 {row['district']} &nbsp;|&nbsp; 📏 {round(row['distance_km'],2)} km away</span><br>
                        <a href="{maps_link}" target="_blank" style="color:#1a5276; font-weight:600;">📌 Open in Google Maps →</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success(f"✅ Found {len(filtered)} hospital(s) within {search_radius} km")
                for i, (_, row) in enumerate(filtered.iterrows(), 1):
                    maps_link = f"https://maps.google.com/?q={row['latitude']},{row['longitude']}"
                    st.markdown(f"""
                    <div class="hospital-card">
                        <b>{i}. {row['hospital_name']}</b><br>
                        <span style="color:#5d6d7e;">🏥 {row['type']} &nbsp;|&nbsp; 📍 {row['district']} &nbsp;|&nbsp; 📏 {round(row['distance_km'],2)} km away</span><br>
                        <a href="{maps_link}" target="_blank" style="color:#1a5276; font-weight:600;">📌 Open in Google Maps →</a>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('<div class="footer-note">⚠️ HealthBridge provides guidance only — not a medical diagnosis. Always consult a qualified doctor.</div>', unsafe_allow_html=True)
