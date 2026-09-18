import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
# Assuming 'pipeline' module and necessary files (data/, models/) exist
from pipeline import clean_football_data, prepare_features_streamlined_fifa, load_fifa_ratings_streamlined, create_enhanced_team_mapping
import re
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide"
)

# --- Helper Function for Logos ---
# Clubs whose logo file is not simply their slugified name.
LOGO_ALIASES = {
    "Man United": "man_utd",
}


def get_logo_path(team_name, logo_dir="logos"):
    """Converts a team name to a logo file path."""
    if team_name in LOGO_ALIASES:
        return os.path.join(logo_dir, LOGO_ALIASES[team_name] + ".png")
    # Apostrophes are dropped rather than replaced, so "Nott'm Forest" resolves
    # to nottm_forest.png; only spaces become underscores.
    filename = team_name.lower().replace("'", "")
    filename = re.sub(r"[ ]", '_', filename)
    filename = re.sub(r"[^a-z0-9_]", '', filename) + ".png"
    return os.path.join(logo_dir, filename)

# --- Custom CSS for FIFA-style UI (Logo size fixed, Home/Away labels styled) ---
st.markdown("""
<style>

/* === GLOBAL BACKGROUND (EA FC STYLE) === */
.stApp {
    background: linear-gradient(145deg, #0a0d1a 0%, #0d1021 60%, #090b18 100%);
    color: #fff;
    font-family: 'Poppins', sans-serif;
    overflow-x: hidden;
}

/* Subtle animated glowing background */
.stApp::before {
    content: "";
    background: radial-gradient(circle at 20% 20%, rgba(0,255,135,0.15) 0%, transparent 70%),
                radial-gradient(circle at 80% 80%, rgba(56,189,248,0.15) 0%, transparent 70%);
    position: fixed;
    inset: 0;
    z-index: -1;
    animation: glowPulse 8s ease-in-out infinite alternate;
}
@keyframes glowPulse {
    0% {opacity: 0.4;}
    100% {opacity: 0.8;}
}

/* === HEADERS === */
h1, .stTitle {
    text-align: center !important;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 2px;
    color: #00ff87;
    text-shadow: 0 0 15px rgba(0,255,135,0.6);
    margin-bottom: 40px !important;
    animation: fadeIn 1.2s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* === TEAM PANELS (EA 3D Style) === */
.team-panel {
    background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
    border-radius: 18px;
    box-shadow: inset 0 0 25px rgba(255,255,255,0.05), 0 6px 25px rgba(0,0,0,0.5);
    transition: all 0.4s ease;
    padding: 25px 20px;
    position: relative;
    overflow: hidden;
}
.team-panel::after {
    content: "";
    position: absolute;
    top: -30%;
    left: -30%;
    width: 160%;
    height: 160%;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    transform: rotate(30deg);
    z-index: 0;
}
.team-panel:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 8px 30px rgba(0,255,135,0.25);
}

/* Home / Away borders */
.home-panel {
    border: 2px solid rgba(0,255,135,0.5);
    box-shadow: 0 0 25px rgba(0,255,135,0.3);
}
.away-panel {
    border: 2px solid rgba(56,189,248,0.5);
    box-shadow: 0 0 25px rgba(56,189,248,0.3);
}

/* === TEAM LOGOS ===
   Not scoped to .team-panel: those wrapper divs are not emitted, so a scoped
   rule would never match and logos would render at their native size. */
div[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}
div[data-testid="stImage"] img {
    max-height: 130px;
    width: auto !important;
    object-fit: contain;
    filter: drop-shadow(0 0 20px rgba(255,255,255,0.4));
    transition: transform 0.4s ease;
}
div[data-testid="stImage"] img:hover {
    transform: scale(1.1);
}

/* === TEAM LABELS === */
.team-selector h2 {
    font-family: 'Orbitron', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 1.4em;
}
span[style*="HOME"], span[style*="AWAY"] {
    letter-spacing: 1px;
    font-size: 0.8em;
}

/* === PREDICT BUTTON (EA Neon Animated) === */
.vs-column .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #00ff87, #38bdf8, #e90052);
    background-size: 300% 300%;
    color: #0a0c1a;
    font-weight: bold;
    font-size: 1.1em;
    padding: 12px 32px;
    border-radius: 10px;
    border: none;
    animation: neonShift 4s linear infinite;
    box-shadow: 0 0 20px rgba(0,255,135,0.5);
    transition: transform 0.2s ease;
}
@keyframes neonShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.vs-column .stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(0,255,135,0.8);
}

/* === RESULTS BOX (Animated Reveal) === */
.results-box {
    background: linear-gradient(135deg, rgba(0,255,135,0.08), rgba(233,0,82,0.08));
    backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 25px rgba(255,255,255,0.15);
    text-align: center;
    animation: fadeUp 0.6s ease-out;
}
@keyframes fadeUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* === METRICS === */
.team-stats .stMetric > label, .results-box .stMetric > label {
    font-size: 0.8em;
    text-transform: uppercase;
    font-weight: 700;
    color: #00ff87 !important;
}
.team-stats .stMetric > div, .results-box .stMetric > div {
    font-size: 1.5em;
    font-weight: 700;
    text-shadow: 0 0 8px rgba(255,255,255,0.3);
}

/* === VS SECTION === */
.vs-column h1 {
    font-size: 4em;
    color: #e90052;
    text-shadow: 0 0 25px rgba(233,0,82,0.8);
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 3px;
}

/* === SCOREBOARD GLOW === */
.results-box h2 {
    color: #00ff87;
    text-shadow: 0 0 15px rgba(0,255,135,0.7);
    font-family: 'Orbitron', sans-serif;
}
.results-box h3, .results-box h4 {
    font-weight: 600;
    color: #fff;
    text-shadow: 0 0 10px rgba(255,255,255,0.3);
}

</style>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Poppins:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# --- Caching Functions ---
@st.cache_resource
def load_model():
    model_path = 'models/xgb_model.joblib'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    model = joblib.load(model_path)
    return model

@st.cache_data
def load_all_data():
    """Loads all data and FIFA stats/mapping just once."""
    paths = [
        'data/pl-22.csv', 'data/pl-23.csv', 
        'data/pl-24.csv', 'data/pl-25.csv'
    ]
    fifa_path = 'data/tbl_team.csv'
    
    # 1. Load historical data
    try:
        # Assuming clean_football_data is robust
        all_dfs = [clean_football_data(pd.read_csv(p)) for p in paths]
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure your 'data' folder contains all CSV files.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error processing data files: {e}")
        return None, None, None, None, None

    historical_df = pd.concat(all_dfs).sort_values('Date').reset_index(drop=True)

    # 2. Load FIFA data
    if not os.path.exists(fifa_path):
        st.error(f"FIFA data file not found at {fifa_path}")
        return historical_df, None, None, None, None
        
    fifa_ratings = load_fifa_ratings_streamlined(fifa_path, "Premier")
    
    # 3. Create Team List and Mapping
    team_list = sorted(list(pd.concat([
        historical_df['HomeTeam'], 
        historical_df['AwayTeam']
    ]).unique()))
    
    match_teams = set(team_list)
    fifa_teams = set(fifa_ratings.keys())
    team_mapping = create_enhanced_team_mapping(match_teams, fifa_teams)
    
    return historical_df, fifa_path, team_list, fifa_ratings, team_mapping

# --- Helper function to get stats ---
def get_team_stats(team_name, fifa_ratings, team_mapping):
    """Gets the ATT, MID, DEF stats for a given team."""
    mapped_name = team_mapping.get(team_name, team_name)
    stats = fifa_ratings.get(mapped_name)
    
    if stats:
        return {
            "ATT": stats.get('attack', 0),
            "MID": stats.get('midfield', 0),
            "DEF": stats.get('defence', 0),
            "rated": True
        }
    # The bundled FIFA table predates several current Premier League clubs, so
    # some teams genuinely have no ratings. Show that rather than a fake "0".
    return {"ATT": "—", "MID": "—", "DEF": "—", "rated": False}

# --- Team Cycling Callback ---
def change_team(side, direction, team_list):
    """Callback to cycle to the next or previous team."""
    current_team = st.session_state[side]
    try:
        current_index = team_list.index(current_team)
    except ValueError:
        current_index = 0 # Default to first team if not found
    
    if direction == "next":
        new_index = (current_index + 1) % len(team_list)
    else: # "prev"
        new_index = (current_index - 1) % len(team_list)
        
    st.session_state[side] = team_list[new_index]
    # When changing team, reset prediction state
    st.session_state.prediction_made = False




# --- Load Model & Data ---
model = load_model()
all_data = load_all_data()

if all_data and all_data[0] is not None and all_data[1] is not None:
    historical_df, fifa_path, team_list, fifa_ratings, team_mapping = all_data
else:
    # Stop app if essential data loading failed (error already shown in load_all_data)
    st.stop() 

# --- Initialize Session State ---
if 'home_team' not in st.session_state or st.session_state.home_team not in team_list:
    st.session_state.home_team = 'Man City'
if 'away_team' not in st.session_state or st.session_state.away_team not in team_list:
    st.session_state.away_team = 'Arsenal'
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# --- UI Layout ---
col_title1, col_title2 = st.columns([1, 9])
with col_title1:
    pl_logo_path = "logos/pl_logo.png"
    if os.path.exists(pl_logo_path):
        st.image(pl_logo_path, width=100)
    else:
        st.image("https://placehold.co/100x100/ffffff/000000?text=PL", width=100)
with col_title2:
    st.title('Premier League Match Predictor')




# --- Get current team data ---
home_stats = get_team_stats(st.session_state.home_team, fifa_ratings, team_mapping)
away_stats = get_team_stats(st.session_state.away_team, fifa_ratings, team_mapping)

home_logo_path = get_logo_path(st.session_state.home_team)
away_logo_path = get_logo_path(st.session_state.away_team)

# Use placeholder image URLs if logos not found
home_logo = home_logo_path if os.path.exists(home_logo_path) else "https://placehold.co/150x50/4f46e5/FFF?text=?"
away_logo = away_logo_path if os.path.exists(away_logo_path) else "https://placehold.co/150x50/004d80/FFF?text=?"

# --- MAIN MATCHUP UI ---
# --- MAIN MATCHUP UI (with side spacing columns added) ---
# Layout: [blank_left, home, vs, away, blank_right]
blank_left, col_home, col_vs, col_away, blank_right = st.columns([0.2, 0.3, 0.1, 0.3, 0.2])

# --- HOME PANEL ---
with col_home:
    # st.markdown('<div class="team-panel home-panel">', unsafe_allow_html=True)
    # st.markdown('<div class="team-selector">', unsafe_allow_html=True)
    c_prev, c_name, c_next = st.columns([1, 4, 1])
    with c_prev:
        st.button("❮", key="home_prev", on_click=change_team, args=('home_team', 'prev', team_list))
    with c_name:
        st.markdown(f"""
            <div style="text-align: center;">
                <span style="color: #00ff87; font-size: 0.9em; font-weight: 600; text-shadow: 0 0 5px rgba(0,255,135,0.5);">HOME</span>
                <h2 style="margin: 0; padding: 0;">{st.session_state.home_team}</h2>
            </div>
        """, unsafe_allow_html=True)
    with c_next:
        st.button("❯", key="home_next", on_click=change_team, args=('home_team', 'next', team_list))
    st.markdown('</div>', unsafe_allow_html=True)
    st.image(home_logo)
    st.markdown('<div class="team-stats">', unsafe_allow_html=True)
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("ATT", home_stats["ATT"])
    s_col2.metric("MID", home_stats["MID"])
    s_col3.metric("DEF", home_stats["DEF"])
    if not home_stats["rated"]:
        st.caption("No FIFA rating in the bundled dataset for this club.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- VS & PREDICT BUTTON ---
with col_vs:
    st.markdown('<div class="vs-column">', unsafe_allow_html=True)
    st.markdown('<h1>VS</h1>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top: 50px;'>", unsafe_allow_html=True)
    if st.button('Predict', type="primary", key="predict"):
        st.session_state.prediction_made = True
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- AWAY PANEL ---
with col_away:
    # st.markdown('<div class="team-panel away-panel">', unsafe_allow_html=True)
    # st.markdown('<div class="team-selector">', unsafe_allow_html=True)
    c_prev, c_name, c_next = st.columns([1, 4, 1])
    with c_prev:
        st.button("❮", key="away_prev", on_click=change_team, args=('away_team', 'prev', team_list))
    with c_name:
        st.markdown(f"""
            <div style="text-align: center;">
                <span style="color: #38bdf8; font-size: 0.9em; font-weight: 600; text-shadow: 0 0 5px rgba(56,189,248,0.5);">AWAY</span>
                <h2 style="margin: 0; padding: 0;">{st.session_state.away_team}</h2>
            </div>
        """, unsafe_allow_html=True)
    with c_next:
        st.button("❯", key="away_next", on_click=change_team, args=('away_team', 'next', team_list))
    st.markdown('</div>', unsafe_allow_html=True)
    st.image(away_logo)
    st.markdown('<div class="team-stats">', unsafe_allow_html=True)
    s_col4, s_col5, s_col6 = st.columns(3)
    s_col4.metric("ATT", away_stats["ATT"])
    s_col5.metric("MID", away_stats["MID"])
    s_col6.metric("DEF", away_stats["DEF"])
    if not away_stats["rated"]:
        st.caption("No FIFA rating in the bundled dataset for this club.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- PREDICTION LOGIC & RESULTS ---
if st.session_state.prediction_made:
    if st.session_state.home_team == st.session_state.away_team:
        st.error('Home and Away teams must be different.')
        st.session_state.prediction_made = False # Reset state
    else:
        with st.spinner(f'Calculating features for {st.session_state.home_team} vs {st.session_state.away_team}...'):
            # 1. Create a new match DataFrame
            new_match = pd.DataFrame({
                'Date': [pd.to_datetime(datetime.now())],
                'HomeTeam': [st.session_state.home_team],
                'AwayTeam': [st.session_state.away_team],
                'FTHG': [0], 'FTAG': [0], 'FTR': ['D'] # Dummy values
            })
            
            # 2. Combine with historical data
            combined_df = pd.concat([historical_df, new_match], ignore_index=True)
            
            # 3. Run the full feature preparation pipeline
            try:
                X_features, _ = prepare_features_streamlined_fifa(
                    df=combined_df, 
                    fifa_path=fifa_path, 
                    target_type='result', 
                    league_filter="Premier"
                )
            except Exception as e:
                st.error(f"Feature engineering failed. Error: {e}")
                st.session_state.prediction_made = False
                st.stop()
            
            # 4. Get the last row (our new match)
            X_pred = X_features.tail(1)
            
            # 5. Make prediction
            prediction_proba = model.predict_proba(X_pred)[0]
            prediction = np.argmax(prediction_proba)
            
            class_map = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}
        
        # --- Display Results ---
        with st.container():
            st.markdown('<div class="results-box">', unsafe_allow_html=True)
            st.subheader(f'Prediction: {st.session_state.home_team} vs. {st.session_state.away_team}')
            
            result_text = class_map[prediction]
            if prediction == 2:
                st.success(f'**Result: {result_text} ({st.session_state.home_team})**')
                st.balloons()
            elif prediction == 0:
                st.success(f'**Result: {result_text} ({st.session_state.away_team})**')
                st.balloons()
            else:
                st.info(f'**Result: {result_text}**')

            st.write("### Probability Breakdown")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(
                label=f"{st.session_state.home_team} (Home Win)", 
                value=f"{prediction_proba[2]:.1%}"
            )
            res_col2.metric(
                label="Draw", 
                value=f"{prediction_proba[1]:.1%}"
            )
            res_col3.metric(
                label=f"{st.session_state.away_team} (Away Win)", 
                value=f"{prediction_proba[0]:.1%}"
            )
            
            with st.expander("Show Features Used for Prediction"):
                st.dataframe(X_pred)
            
            st.markdown('</div>', unsafe_allow_html=True)

