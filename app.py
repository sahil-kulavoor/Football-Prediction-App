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

# --- Custom CSS ---
# Selectors target Streamlit's real DOM: `data-testid` attributes and the
# `st-key-<key>` class Streamlit puts on any keyed widget or container. A bare
# <div> emitted through st.markdown does NOT wrap the widgets that follow it,
# so class names invented that way never match anything.
st.markdown("""
<style>

/* === GLOBAL BACKGROUND === */
.stApp {
    background: linear-gradient(145deg, #0a0d1a 0%, #0d1021 60%, #090b18 100%);
    color: #fff;
    font-family: 'Poppins', sans-serif;
}
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

/* === TITLE === */
h1, .stTitle {
    text-align: center !important;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 2px;
    color: #00ff87;
    text-shadow: 0 0 15px rgba(0,255,135,0.6);
    animation: fadeIn 1.2s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Header crest: kept crisp, no glow. */
.st-key-brand_mark [data-testid="stImage"] img {
    filter: none;
}

/* === TEAM PANELS === */
.st-key-home_panel, .st-key-away_panel {
    background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
    border-radius: 18px;
    padding: 22px 18px 10px 18px;
    transition: transform 0.35s ease, box-shadow 0.35s ease;
}
.st-key-home_panel {
    border: 2px solid rgba(0,255,135,0.45);
    box-shadow: 0 0 25px rgba(0,255,135,0.22);
}
.st-key-away_panel {
    border: 2px solid rgba(56,189,248,0.45);
    box-shadow: 0 0 25px rgba(56,189,248,0.22);
}
.st-key-home_panel:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 34px rgba(0,255,135,0.34);
}
.st-key-away_panel:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 34px rgba(56,189,248,0.34);
}

/* === TEAM NAME ROW === */
.team-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.team-name {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1rem, 2vw, 1.55rem);
    font-weight: 700;
    margin: 2px 0 0 0;
    line-height: 1.2;
}

/* Crest sits in a fixed-height box so both panels' stats line up. */
/* Streamlit sizes the image's whole wrapper chain to the bitmap's width, so a
   plain width:100% resolves against that same narrow box. Force every wrapper
   to fill the panel, then centre inside it. */
.st-key-home_panel [data-testid="stElementContainer"]:has([data-testid="stImage"]),
.st-key-away_panel [data-testid="stElementContainer"]:has([data-testid="stImage"]),
.st-key-home_panel [data-testid="stFullScreenFrame"],
.st-key-away_panel [data-testid="stFullScreenFrame"],
.st-key-home_panel [data-testid="stFullScreenFrame"] > div,
.st-key-away_panel [data-testid="stFullScreenFrame"] > div,
.st-key-home_panel [data-testid="stImageContainer"],
.st-key-away_panel [data-testid="stImageContainer"] {
    width: 100% !important;
}
.st-key-home_panel [data-testid="stImageContainer"],
.st-key-away_panel [data-testid="stImageContainer"] {
    justify-content: center;
}
.st-key-home_panel [data-testid="stImage"],
.st-key-away_panel [data-testid="stImage"] {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100% !important;
    height: 150px;
}
.st-key-home_panel [data-testid="stImage"] img,
.st-key-away_panel [data-testid="stImage"] img {
    max-height: 130px;
    max-width: 100%;
    width: auto !important;
    object-fit: contain;
    filter: drop-shadow(0 0 18px rgba(255,255,255,0.35));
    transition: transform 0.35s ease;
}
.st-key-home_panel [data-testid="stImage"] img:hover,
.st-key-away_panel [data-testid="stImage"] img:hover {
    transform: scale(1.08);
}

/* === TEAM CYCLING ARROWS === */
.st-key-home_prev button, .st-key-home_next button,
.st-key-away_prev button, .st-key-away_next button {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.18);
    color: #fff;
    border-radius: 10px;
    width: 100%;
    transition: all 0.2s ease;
}
.st-key-home_prev button:hover, .st-key-home_next button:hover {
    border-color: rgba(0,255,135,0.8);
    color: #00ff87;
    box-shadow: 0 0 12px rgba(0,255,135,0.4);
}
.st-key-away_prev button:hover, .st-key-away_next button:hover {
    border-color: rgba(56,189,248,0.8);
    color: #38bdf8;
    box-shadow: 0 0 12px rgba(56,189,248,0.4);
}

/* === METRICS === */
[data-testid="stMetric"] {
    text-align: center;
}
/* Streamlit renders the label as a grid whose inner track is shrink-wrapped to
   the left, so grid alignment alone will not centre it. Make every wrapper
   full-width and centre the text instead. */
[data-testid="stMetricLabel"] {
    display: block !important;
    width: 100%;
    text-align: center;
}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"],
[data-testid="stMetricLabel"] p {
    width: 100%;
    text-align: center;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1px;
    color: #00ff87 !important;
}
.st-key-away_panel [data-testid="stMetricLabel"] p {
    color: #38bdf8 !important;
}
[data-testid="stMetricValue"] {
    justify-content: center;
    font-size: clamp(1.1rem, 2.1vw, 1.65rem) !important;
    font-weight: 700;
    text-shadow: 0 0 8px rgba(255,255,255,0.3);
}

/* === VS + PREDICT === */
.vs-mark {
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.7rem, 3.6vw, 3rem);
    font-weight: 700;
    color: #e90052;
    text-shadow: 0 0 25px rgba(233,0,82,0.8);
    line-height: 1;
    margin-bottom: 14px;
}

.st-key-predict button {
    background: linear-gradient(90deg, #00ff87, #38bdf8, #e90052);
    background-size: 300% 300%;
    color: #0a0c1a !important;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 12px 8px;
    width: 100%;
    border-radius: 12px;
    border: none;
    white-space: nowrap;
    animation: neonShift 4s linear infinite;
    box-shadow: 0 0 20px rgba(0,255,135,0.5);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
@keyframes neonShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.st-key-predict button:hover {
    transform: scale(1.04);
    box-shadow: 0 0 28px rgba(0,255,135,0.85);
}
.st-key-predict button:focus:not(:active) {
    color: #0a0c1a !important;
    border: none;
}

/* === RESULTS === */
.st-key-results {
    background: linear-gradient(135deg, rgba(0,255,135,0.08), rgba(233,0,82,0.08));
    border-radius: 20px;
    padding: 22px 26px;
    margin-top: 26px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 0 25px rgba(255,255,255,0.10);
    animation: fadeUp 0.6s ease-out;
}
@keyframes fadeUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}
.result-headline {
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.15rem, 2.8vw, 1.9rem);
    font-weight: 700;
    margin: 4px 0;
    color: #00ff87;
    text-shadow: 0 0 15px rgba(0,255,135,0.6);
}
.result-sub {
    text-align: center;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 18px;
}
.prob-caption {
    text-align: center;
    font-size: 0.78rem;
    opacity: 0.6;
    margin-top: 14px;
}

/* === SMALL SCREENS === */
@media (max-width: 640px) {
    /* Streamlit stacks columns below this width by forcing a full-width
       min-width. The home/vs/away split SHOULD stack, but the selector arrows
       and the ATT/MID/DEF row must stay on one line, so override them here. */
    .st-key-home_panel [data-testid="stHorizontalBlock"],
    .st-key-away_panel [data-testid="stHorizontalBlock"],
    .st-key-results [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        gap: 0.4rem !important;
    }
    .st-key-home_panel [data-testid="stColumn"],
    .st-key-away_panel [data-testid="stColumn"],
    .st-key-results [data-testid="stColumn"] {
        min-width: 0 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
    }

    .st-key-home_panel, .st-key-away_panel {
        padding: 16px 12px 6px 12px;
    }
    .st-key-home_panel [data-testid="stImage"],
    .st-key-away_panel [data-testid="stImage"] {
        height: 96px;
    }
    .st-key-home_panel [data-testid="stImage"] img,
    .st-key-away_panel [data-testid="stImage"] img {
        max-height: 80px;
    }
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
col_brand, col_heading = st.columns([1, 9], vertical_alignment="center")
with col_brand:
    with st.container(key="brand_mark"):
        pl_logo_path = "logos/pl_logo.png"
        if os.path.exists(pl_logo_path):
            st.image(pl_logo_path, width=84)
        else:
            st.image("https://placehold.co/84x84/ffffff/000000?text=PL", width=84)
with col_heading:
    st.title('Premier League Match Predictor')


# --- Get current team data ---
home_stats = get_team_stats(st.session_state.home_team, fifa_ratings, team_mapping)
away_stats = get_team_stats(st.session_state.away_team, fifa_ratings, team_mapping)

home_logo_path = get_logo_path(st.session_state.home_team)
away_logo_path = get_logo_path(st.session_state.away_team)

# Use placeholder image URLs if logos not found
home_logo = home_logo_path if os.path.exists(home_logo_path) else "https://placehold.co/150x50/4f46e5/FFF?text=?"
away_logo = away_logo_path if os.path.exists(away_logo_path) else "https://placehold.co/150x50/004d80/FFF?text=?"


def render_team_panel(side, team, logo, stats, label, accent, container_key):
    """One team card: selector arrows, crest, and FIFA ratings."""
    with st.container(key=container_key):
        col_prev, col_name, col_next = st.columns([1, 5, 1], vertical_alignment="center")
        with col_prev:
            st.button("❮", key=f"{side}_prev", on_click=change_team,
                      args=(f"{side}_team", 'prev', team_list))
        with col_name:
            st.markdown(
                f'<div style="text-align:center;">'
                f'<div class="team-label" style="color:{accent};">{label}</div>'
                f'<div class="team-name">{team}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_next:
            st.button("❯", key=f"{side}_next", on_click=change_team,
                      args=(f"{side}_team", 'next', team_list))

        st.image(logo)

        stat_cols = st.columns(3)
        stat_cols[0].metric("ATT", stats["ATT"])
        stat_cols[1].metric("MID", stats["MID"])
        stat_cols[2].metric("DEF", stats["DEF"])
        if not stats["rated"]:
            st.caption("No FIFA rating in the bundled dataset for this club.")


# --- MAIN MATCHUP UI ---
col_home, col_vs, col_away = st.columns([1, 0.45, 1], vertical_alignment="center")

with col_home:
    render_team_panel("home", st.session_state.home_team, home_logo, home_stats,
                      "Home", "#00ff87", "home_panel")

with col_vs:
    st.markdown('<div class="vs-mark">VS</div>', unsafe_allow_html=True)
    if st.button('Predict', type="primary", key="predict", use_container_width=True):
        st.session_state.prediction_made = True

with col_away:
    render_team_panel("away", st.session_state.away_team, away_logo, away_stats,
                      "Away", "#38bdf8", "away_panel")

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
        result_text = class_map[prediction]
        if prediction == 2:
            headline = f'{result_text} — {st.session_state.home_team}'
        elif prediction == 0:
            headline = f'{result_text} — {st.session_state.away_team}'
        else:
            headline = result_text

        with st.container(key="results"):
            st.markdown(
                f'<div class="result-sub">'
                f'{st.session_state.home_team} vs {st.session_state.away_team}</div>'
                f'<div class="result-headline">{headline}</div>',
                unsafe_allow_html=True,
            )

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

            st.progress(float(prediction_proba[2]), text="Home")
            st.progress(float(prediction_proba[1]), text="Draw")
            st.progress(float(prediction_proba[0]), text="Away")

            st.markdown(
                '<div class="prob-caption">Bookmaker odds are unavailable for an '
                'unplayed fixture, so these probabilities sit closer to even than '
                'the model\'s training scores suggest. See the README.</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Show features used for this prediction"):
                st.dataframe(X_pred, use_container_width=True)

        # Celebrate once per new prediction, not on every rerun.
        if prediction != 1 and st.session_state.get('celebrated_for') != (
            st.session_state.home_team, st.session_state.away_team
        ):
            st.session_state.celebrated_for = (
                st.session_state.home_team, st.session_state.away_team
            )
            st.balloons()

