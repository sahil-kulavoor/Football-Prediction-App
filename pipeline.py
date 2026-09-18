"""Data cleaning and pre-match feature engineering for Premier League match prediction.

All feature engineering here is leakage-safe: every feature for a given match is
computed using only information available *before* that match is played.
"""

from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def _prior_window_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Sum of the previous up-to-`window` entries, excluding the current one.

    Equivalent to Series.shift(1).rolling(window, min_periods=1).sum().fillna(0),
    computed with a running total so each element costs O(1) instead of O(window).
    """
    values = np.asarray(values, dtype=float)
    running = np.concatenate(([0.0], np.cumsum(values)))
    positions = np.arange(values.size)
    lower = np.maximum(0, positions - window)
    return running[positions] - running[lower]


def _prior_window_count(size: int, window: int) -> np.ndarray:
    """How many prior entries each position averages over (capped at `window`)."""
    return np.minimum(np.arange(size), window).astype(float)


def _prior_window_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the previous up-to-`window` entries; 0.0 where there are none."""
    counts = _prior_window_count(np.asarray(values).size, window)
    totals = _prior_window_sum(values, window)
    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)


def load_football_data(paths: List[str]) -> List[pd.DataFrame]:
    return [pd.read_csv(path) for path in paths]

def clean_football_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean football match data - keep only pre-match features + target"""
    df = df.copy()

    # Convert Date column
    if 'Date' in df.columns:
        # Source CSVs (football-data.co.uk) use DD/MM/YYYY. Without dayfirst=True
        # pandas infers the format from the first rows, which are ambiguous for
        # seasons opening on a day <= 12 -- that silently swapped day/month on
        # some rows and turned the rest into NaT, dropping them below.
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['Date'])

    # Ensure essential columns exist
    essential_columns = ['HomeTeam', 'AwayTeam', 'FTR']
    for col in essential_columns:
        if col not in df.columns:
            raise ValueError(f"Missing essential column: {col}")

    # Keep odds columns only (pre-match information)
    odds_columns = [col for col in df.columns
                    if any(book in col for book in ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'VC'])]

    keep_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG'] + odds_columns
    df = df[keep_columns].dropna(subset=['FTR'])

    # Normalize odds → numeric
    for col in odds_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    return df


def engineer_football_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer safe pre-match features (no leakage)."""
    df = df.copy()

    # Odds-based features
    home_odds_cols = [c for c in df.columns if c.startswith("B365H") or c.startswith("BWH") or c.startswith("IWH")]
    draw_odds_cols = [c for c in df.columns if c.startswith("B365D") or c.startswith("BWD") or c.startswith("IWD")]
    away_odds_cols = [c for c in df.columns if c.startswith("B365A") or c.startswith("BWA") or c.startswith("IWA")]

    if home_odds_cols:
        df['Avg_Home_Odds'] = df[home_odds_cols].mean(axis=1)
        df['Home_Implied_Prob'] = 1 / df['Avg_Home_Odds']

    if draw_odds_cols:
        df['Avg_Draw_Odds'] = df[draw_odds_cols].mean(axis=1)
        df['Draw_Implied_Prob'] = 1 / df['Avg_Draw_Odds']

    if away_odds_cols:
        df['Avg_Away_Odds'] = df[away_odds_cols].mean(axis=1)
        df['Away_Implied_Prob'] = 1 / df['Avg_Away_Odds']

    # Encode categorical teams
    df['HomeTeam_Encoded'] = pd.Categorical(df['HomeTeam']).codes
    df['AwayTeam_Encoded'] = pd.Categorical(df['AwayTeam']).codes

    # Date-based features
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    return df


# Cell 6: Elo ratings calculation (separate)

def calculate_elo_ratings(df, k=20, home_advantage=100):
    """
    Calculates Elo ratings for each match.
    Updates ratings after each game sequentially (safe, no leakage).
    """
    df = df.copy()
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    elo = {team: 1500 for team in teams}  # initialize all teams at 1500

    home_elos, away_elos = [], []

    for _, row in df.sort_values('Date').iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        home_rating, away_rating = elo[home], elo[away]

        # Expected probabilities
        exp_home = 1 / (1 + 10 ** (-(home_rating + home_advantage - away_rating)/400))
        exp_away = 1 - exp_home

        # Actual result
        if row['FTHG'] > row['FTAG']:
            score_home, score_away = 1, 0
        elif row['FTHG'] < row['FTAG']:
            score_home, score_away = 0, 1
        else:
            score_home, score_away = 0.5, 0.5

        # Update ratings
        elo[home] += k * (score_home - exp_home)
        elo[away] += k * (score_away - exp_away)

        # Save pre-match Elo (before update)
        home_elos.append(home_rating)
        away_elos.append(away_rating)

    df['HomeElo'] = home_elos
    df['AwayElo'] = away_elos
    return df

# Cell 7: Rest days calculation

def add_rest_days(df):
    """
    Calculates days since last match for each team.
    """
    df = df.copy()

    # Melt format (team, match)
    home_df = df[['Date','HomeTeam']].rename(columns={'HomeTeam':'Team'})
    away_df = df[['Date','AwayTeam']].rename(columns={'AwayTeam':'Team'})
    long_df = pd.concat([home_df, away_df], axis=0).sort_values(['Team','Date'])

    # Calculate rest days
    long_df['RestDays'] = long_df.groupby('Team')['Date'].diff().dt.days

    # Merge back into original df
    df = df.merge(long_df[['Date','Team','RestDays']],
                  left_on=['Date','HomeTeam'], right_on=['Date','Team'], how='left') \
           .rename(columns={'RestDays':'HomeRestDays'}).drop(columns=['Team'])

    df = df.merge(long_df[['Date','Team','RestDays']],
                  left_on=['Date','AwayTeam'], right_on=['Date','Team'], how='left') \
           .rename(columns={'RestDays':'AwayRestDays'}).drop(columns=['Team'])

    return df


def add_goal_difference_features(df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
    """
    Adds rolling Goal Difference (GD = goals for - goals against) features
    over the last `window_size` matches for both Home and Away teams.
    """
    df = df.copy()
    df = df.sort_values("Date")

    # Initialize new columns
    df['HomeGD_LastN'] = 0.0
    df['AwayGD_LastN'] = 0.0

    # Process each team separately
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

    # Vectorised per team. The mean of a team's goal difference over its previous
    # up-to-`window_size` games is a shifted rolling mean, so the per-match rescan
    # of every earlier game is unnecessary.
    home_names = df['HomeTeam'].to_numpy()
    away_names = df['AwayTeam'].to_numpy()
    goals_for_home = df['FTHG'].to_numpy()
    goals_for_away = df['FTAG'].to_numpy()
    labels = df.index.to_numpy()

    for team in teams:
        # Positions of this team's matches, in the frame's (date-sorted) order.
        pos = np.flatnonzero((home_names == team) | (away_names == team))
        if pos.size == 0:
            continue

        played_at_home = home_names[pos] == team
        goal_diff = np.where(
            played_at_home,
            goals_for_home[pos] - goals_for_away[pos],
            goals_for_away[pos] - goals_for_home[pos],
        ).astype(float)

        # Excludes the current match from its own feature (no leakage).
        prior_mean = _prior_window_mean(goal_diff, window_size)

        team_labels = labels[pos]
        df.loc[team_labels[played_at_home], 'HomeGD_LastN'] = prior_mean[played_at_home]
        df.loc[team_labels[~played_at_home], 'AwayGD_LastN'] = prior_mean[~played_at_home]

    return df


# Cell 8: Add bookmaker odds (if available in dataset)

def add_odds_features(df):
    """
    Extracts odds features (if columns available).
    Many datasets use B365H, B365D, B365A (Bet365 odds for Home/Draw/Away).
    """
    df = df.copy()
    odds_cols = ['B365H','B365D','B365A']
    if all(col in df.columns for col in odds_cols):
        # Normalize odds → implied probabilities
        df['ImpliedHomeProb'] = 1 / df['B365H']
        df['ImpliedDrawProb'] = 1 / df['B365D']
        df['ImpliedAwayProb'] = 1 / df['B365A']
        total = df[['ImpliedHomeProb','ImpliedDrawProb','ImpliedAwayProb']].sum(axis=1)
        for col in ['ImpliedHomeProb','ImpliedDrawProb','ImpliedAwayProb']:
            df[col] = df[col] / total  # normalize
    return df


def add_h2h_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Adds Head-to-Head (H2H) historical features for each matchup.
    Calculates rolling averages of goals, points, wins/losses over last N matches.
    """
    df = df.copy()
    df = df.sort_values("Date")  # Ensure chronological order

    # Initialize new columns
    df['H2H_HomeWins'] = 0.0
    df['H2H_AwayWins'] = 0.0
    df['H2H_HomeGoals'] = 0.0
    df['H2H_AwayGoals'] = 0.0
    df['H2H_Meetings'] = 0.0

    home_names = df['HomeTeam'].to_numpy()
    away_names = df['AwayTeam'].to_numpy()
    results = df['FTR'].to_numpy()
    goals_for_home = df['FTHG'].to_numpy()
    goals_for_away = df['FTAG'].to_numpy()
    labels = df.index.to_numpy()

    n_rows = len(df)
    out_home_wins = np.zeros(n_rows)
    out_away_wins = np.zeros(n_rows)
    out_home_goals = np.zeros(n_rows)
    out_away_goals = np.zeros(n_rows)
    out_meetings = np.zeros(n_rows)

    # Group by unique matchup pair (order-independent). Both (A,B) and (B,A)
    # select the same set of rows and write to it; the later group wins, exactly
    # as before -- the grouping order is unchanged, so the outcome is unchanged.
    for (home, away), group in df.groupby(['HomeTeam', 'AwayTeam']):
        # Matches between these two teams in either home/away order.
        pos = np.flatnonzero(
            ((home_names == home) & (away_names == away))
            | ((home_names == away) & (away_names == home))
        )
        if pos.size == 0:
            continue

        # Orientation of each meeting relative to this group's "home" team.
        home_was_at_home = home_names[pos] == home
        meeting_results = results[pos]

        home_won = ((home_was_at_home) & (meeting_results == 'H')) | \
                   ((~home_was_at_home) & (meeting_results == 'A'))
        away_won = ((~home_was_at_home) & (meeting_results == 'H')) | \
                   ((home_was_at_home) & (meeting_results == 'A'))

        home_goals = np.where(home_was_at_home, goals_for_home[pos], goals_for_away[pos])
        away_goals = np.where(home_was_at_home, goals_for_away[pos], goals_for_home[pos])

        out_home_wins[pos] = _prior_window_sum(home_won, window_size)
        out_away_wins[pos] = _prior_window_sum(away_won, window_size)
        out_home_goals[pos] = _prior_window_sum(home_goals, window_size)
        out_away_goals[pos] = _prior_window_sum(away_goals, window_size)
        # Number of prior meetings counted, capped at the window.
        out_meetings[pos] = np.minimum(np.arange(pos.size), window_size)

    df['H2H_HomeWins'] = out_home_wins
    df['H2H_AwayWins'] = out_away_wins
    df['H2H_HomeGoals'] = out_home_goals
    df['H2H_AwayGoals'] = out_away_goals
    df['H2H_Meetings'] = out_meetings

    return df


def add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds streak features for both teams:
    - Current winning streak
    - Current losing streak
    before each match (no leakage).
    """
    df = df.copy()
    df = df.sort_values("Date")

    # Initialize columns
    df['HomeWinStreak'] = 0
    df['HomeLoseStreak'] = 0
    df['AwayWinStreak'] = 0
    df['AwayLoseStreak'] = 0

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

    for team in teams:
        # Extract games involving this team
        team_games = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values("Date")

        win_streak, lose_streak = 0, 0
        streaks = []

        for _, match in team_games.iterrows():
            # Assign streak before match
            streaks.append((win_streak, lose_streak))

            # Update streak based on match result
            if match['HomeTeam'] == team:
                if match['FTR'] == 'H':  # win
                    win_streak += 1
                    lose_streak = 0
                elif match['FTR'] == 'A':  # loss
                    lose_streak += 1
                    win_streak = 0
                else:  # draw resets both
                    win_streak, lose_streak = 0, 0
            else:  # team played away
                if match['FTR'] == 'A':
                    win_streak += 1
                    lose_streak = 0
                elif match['FTR'] == 'H':
                    lose_streak += 1
                    win_streak = 0
                else:
                    win_streak, lose_streak = 0, 0

        # Assign streak values back to df
        for i, (ws, ls) in enumerate(streaks):
            idx = team_games.index[i]
            if team_games.loc[idx, 'HomeTeam'] == team:
                df.at[idx, 'HomeWinStreak'] = ws
                df.at[idx, 'HomeLoseStreak'] = ls
            else:
                df.at[idx, 'AwayWinStreak'] = ws
                df.at[idx, 'AwayLoseStreak'] = ls

    return df


def add_points_per_game_features(df: pd.DataFrame, windows=[5, 10, 15]) -> pd.DataFrame:
    """
    Adds rolling Points Per Game (PPG) features for Home & Away teams
    over given window sizes. (Shifted to avoid leakage)
    """
    df = df.copy()
    df = df.sort_values("Date")

    # Compute points from results
    df['HomePoints'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    df['AwayPoints'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

    # Vectorised per team: PPG before a match is the mean of that team's points
    # over its previous up-to-`w` games, i.e. a shifted rolling mean.
    home_names = df['HomeTeam'].to_numpy()
    away_names = df['AwayTeam'].to_numpy()
    points_as_home = df['HomePoints'].to_numpy()
    points_as_away = df['AwayPoints'].to_numpy()

    n_rows = len(df)
    columns = {}
    for w in windows:
        columns[f'HomePPG_{w}'] = np.full(n_rows, np.nan)
        columns[f'AwayPPG_{w}'] = np.full(n_rows, np.nan)

    for team in teams:
        pos = np.flatnonzero((home_names == team) | (away_names == team))
        if pos.size == 0:
            continue

        played_at_home = home_names[pos] == team
        points = np.where(played_at_home, points_as_home[pos], points_as_away[pos]).astype(float)

        for w in windows:
            ppg = _prior_window_mean(points, w)
            columns[f'HomePPG_{w}'][pos[played_at_home]] = ppg[played_at_home]
            columns[f'AwayPPG_{w}'][pos[~played_at_home]] = ppg[~played_at_home]

    for name, values in columns.items():
        df[name] = values

    # Drop helper columns
    df = df.drop(columns=['HomePoints','AwayPoints'])
    return df


def create_team_form_features(df, rolling_window=5):
    """
    Creates rolling form features for each team using only past games.
    - Points: 3 for win, 1 for draw, 0 for loss
    - Goals scored & conceded
    - Shots taken & allowed (if available in data)
    """

    # Copy to avoid overwriting
    df = df.copy()

    # Assign match points for each team
    df['HomePoints'] = df.apply(lambda x: 3 if x['FTHG'] > x['FTAG']
                                else 1 if x['FTHG'] == x['FTAG'] else 0, axis=1)
    df['AwayPoints'] = df.apply(lambda x: 3 if x['FTAG'] > x['FTHG']
                                else 1 if x['FTHG'] == x['FTAG'] else 0, axis=1)

    # Melt dataframe so each row is a (team, match)
    home_df = df[['Date','HomeTeam','FTHG','FTAG','HomePoints']].rename(
        columns={'HomeTeam':'Team','FTHG':'GoalsFor','FTAG':'GoalsAgainst','HomePoints':'Points'})
    away_df = df[['Date','AwayTeam','FTAG','FTHG','AwayPoints']].rename(
        columns={'AwayTeam':'Team','FTAG':'GoalsFor','FTHG':'GoalsAgainst','AwayPoints':'Points'})

    long_df = pd.concat([home_df, away_df], axis=0).sort_values(['Team','Date'])

    # Rolling averages (safe: shift ensures only past games are used)
    for col in ['Points','GoalsFor','GoalsAgainst']:
        long_df[f'{col}_Rolling{rolling_window}'] = (
            long_df.groupby('Team')[col]
            .transform(lambda x: x.shift().rolling(rolling_window, min_periods=1).mean())
        )

    # Merge back into main dataframe (both home & away sides)
    df = df.merge(long_df[['Date','Team','Points_Rolling5','GoalsFor_Rolling5','GoalsAgainst_Rolling5']],
                  left_on=['Date','HomeTeam'], right_on=['Date','Team'], how='left') \
           .rename(columns={
               'Points_Rolling5':'HomeFormPoints',
               'GoalsFor_Rolling5':'HomeFormGF',
               'GoalsAgainst_Rolling5':'HomeFormGA'
           }).drop(columns=['Team'])

    df = df.merge(long_df[['Date','Team','Points_Rolling5','GoalsFor_Rolling5','GoalsAgainst_Rolling5']],
                  left_on=['Date','AwayTeam'], right_on=['Date','Team'], how='left') \
           .rename(columns={
               'Points_Rolling5':'AwayFormPoints',
               'GoalsFor_Rolling5':'AwayFormGF',
               'GoalsAgainst_Rolling5':'AwayFormGA'
           }).drop(columns=['Team'])

    return df


def examine_fifa_data(fifa_path: str):
    """Examine FIFA data structure to understand the format"""
    try:
        fifa_df = pd.read_csv(fifa_path)
        print(f"FIFA data shape: {fifa_df.shape}")
        print(f"Columns: {fifa_df.columns.tolist()}")
        print("\nFirst few rows:")
        print(fifa_df.head())
        print(f"\nUnique leagues: {fifa_df['str_league'].unique()}")
        print(f"\nPremier League teams:")
        pl_teams = fifa_df[fifa_df['str_league'].str.contains('Premier', case=False, na=False)]
        print(pl_teams[['str_team_name', 'str_league']].head(10))
        return fifa_df
    except Exception as e:
        print(f"Error examining FIFA data: {e}")
        return None
    


def load_fifa_ratings_streamlined(fifa_path: str, league_filter: str = None) -> Dict[str, Dict]:
    """
    Load only the 4 core FIFA ratings: overall, attack, midfield, defence
    """
    try:
        fifa_df = pd.read_csv(fifa_path)
        print(f"Loaded FIFA CSV with {len(fifa_df)} rows")

        # Filter by league if specified
        if league_filter:
            filtered = fifa_df[fifa_df['str_league'].str.contains(league_filter, case=False, na=False)]
            if len(filtered) == 0:
                print(f"No teams found for league filter '{league_filter}', using all teams")
                filtered = fifa_df
            fifa_df = filtered
            print(f"After filtering: {len(fifa_df)} teams")

        # Create mapping with only 4 core features
        fifa_ratings = {}
        for _, row in fifa_df.iterrows():
            team_name = row['str_team_name']
            fifa_ratings[team_name] = {
                'overall': row['int_overall'],
                'attack': row['int_attack'],     # <-- ADDED
                'midfield': row['int_midfield'],   # <-- ADDED
                'defence': row['int_defence']    # <-- ADDED
            }

        print(f"Successfully loaded FIFA ratings for {len(fifa_ratings)} teams")
        return fifa_ratings

    except Exception as e:
        print(f"Error loading FIFA ratings: {e}")
        return {}
    
    
def create_enhanced_team_mapping(match_data_teams: set, fifa_teams: set) -> Dict[str, str]:
    """
    Enhanced team name mapping to handle Premier League team name variations.
    """
    mapping = {}

    # Direct matches first
    for match_team in match_data_teams:
        if match_team in fifa_teams:
            mapping[match_team] = match_team

    # Enhanced Premier League team name variations
    name_variations = {
        # Common abbreviations and variations
        'Man United': ['Manchester United', 'Manchester Utd'],
        'Man City': ['Manchester City', 'Manchester'],
        'Tottenham': ['Tottenham Hotspur', 'Spurs'],
        'Brighton': ['Brighton & Hove Albion', 'Brighton & Hove'],
        'Wolves': ['Wolverhampton Wanderers', 'Wolverhampton'],
        'Newcastle': ['Newcastle United', 'Newcastle Utd'],
        'West Ham': ['West Ham United', 'West Ham Utd'],
        'Leicester': ['Leicester City', 'Leicester City FC'],
        'Crystal Palace': ['Crystal Palace FC'],
        'Sheffield United': ['Sheffield United FC', 'Sheffield Utd'],
        'Norwich': ['Norwich City', 'Norwich City FC'],
        "Nott'm Forest": ['Nottingham Forest', 'Nottingham Forest FC'],
        'Southampton': ['Southampton FC'],
        'Aston Villa': ['Aston Villa FC'],
        'Bournemouth': ['AFC Bournemouth', 'Bournemouth FC'],
        'Burnley': ['Burnley FC'],
        'Everton': ['Everton FC'],
        'Fulham': ['Fulham FC'],
        'Leeds': ['Leeds United', 'Leeds United FC'],
        'Liverpool': ['Liverpool FC'],
        'Arsenal': ['Arsenal FC'],
        'Chelsea': ['Chelsea FC'],
        'Brentford': ['Brentford FC'],
        'Luton': ['Luton Town', 'Luton Town FC'],
        'Ipswich': ['Ipswich Town', 'Ipswich Town FC'],
        'Bournemouth': ['AFC Bournemouth'],
        'Brentford': ['Brentford FC'],
        "Nott'm Forest": ['Nottingham Forest'],
        'Luton': ['Luton Town']
    }

    # Apply variations - try multiple possible FIFA names for each match team
    for match_team in match_data_teams:
        if match_team not in mapping:  # Not directly matched
            possible_fifa_names = name_variations.get(match_team, [match_team])
            for fifa_name in possible_fifa_names:
                if fifa_name in fifa_teams:
                    mapping[match_team] = fifa_name
                    break

    return mapping


def add_fifa_features_streamlined(df: pd.DataFrame, fifa_ratings: Dict[str, Dict],
                                 team_mapping: Dict[str, str] = None) -> pd.DataFrame:
    """
    Add only the 4 core FIFA team ratings as features.
    """
    df = df.copy()

    # Initialize FIFA feature columns (only 4 core features)
    fifa_features = ['overall', 'attack', 'midfield', 'defence']

    for prefix in ['Home', 'Away']:
        for feature in fifa_features:
            df[f'{prefix}FIFA_{feature}'] = 0

    # Apply team name mapping if provided
    effective_mapping = team_mapping if team_mapping else {}

    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Map team names if mapping exists
        fifa_home_team = effective_mapping.get(home_team, home_team)
        fifa_away_team = effective_mapping.get(away_team, away_team)

        # Add home team FIFA ratings
        if fifa_home_team in fifa_ratings:
            home_ratings = fifa_ratings[fifa_home_team]
            for feature in fifa_features:
                df.at[idx, f'HomeFIFA_{feature}'] = home_ratings[feature]

        # Add away team FIFA ratings
        if fifa_away_team in fifa_ratings:
            away_ratings = fifa_ratings[fifa_away_team]
            for feature in fifa_features:
                df.at[idx, f'AwayFIFA_{feature}'] = away_ratings[feature]

    # Create relative FIFA features (Home - Away) - these are often more predictive
    for feature in fifa_features:
        df[f'FIFA_{feature}_diff'] = df[f'HomeFIFA_{feature}'] - df[f'AwayFIFA_{feature}']

    return df

def prepare_features_streamlined_fifa(df: pd.DataFrame, fifa_path: str = None,
                                     target_type: str = 'result',
                                     league_filter: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Feature preparation with streamlined FIFA integration (4 features only).
    """

    # Load FIFA ratings if path provided
    fifa_ratings = {}
    team_mapping = {}
    if fifa_path:
        try:
            fifa_ratings = load_fifa_ratings_streamlined(fifa_path, league_filter)

            if fifa_ratings:
                # Get unique team names from match data
                match_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
                fifa_teams = set(fifa_ratings.keys())

                # Create enhanced team name mapping
                team_mapping = create_enhanced_team_mapping(match_teams, fifa_teams)

                print(f"Successfully mapped {len(team_mapping)} teams")

                # Show unmapped teams
                unmapped = match_teams - set(team_mapping.keys())
                if unmapped:
                    print(f"Unmapped teams: {unmapped}")
                else:
                    print("All teams successfully mapped!")

        except Exception as e:
            print(f"Error loading FIFA ratings: {e}")
            print("Continuing without FIFA features...")

    # Apply all existing feature engineering functions
    df = create_team_form_features(df)
    df = calculate_elo_ratings(df)
    df = add_rest_days(df)
    df = add_odds_features(df)
    df = add_h2h_features(df)
    df = add_goal_difference_features(df, window_size=10)
    df = add_streak_features(df)
    df = add_points_per_game_features(df, windows=[5,10,15])

    # Add streamlined FIFA features if available
    if fifa_ratings:
        df = add_fifa_features_streamlined(df, fifa_ratings, team_mapping)

    # Define feature columns
    feature_columns = []

    # Existing features (same as before)
    team_features = ['HomeTeam_Encoded', 'AwayTeam_Encoded']
    feature_columns.extend([col for col in team_features if col in df.columns])

    date_features = ['Month', 'DayOfWeek', 'Is_Weekend']
    feature_columns.extend([col for col in date_features if col in df.columns])

    form_features = [
        'HomeFormPoints','HomeFormGF','HomeFormGA',
        'AwayFormPoints','AwayFormGF','AwayFormGA'
    ]
    feature_columns.extend([col for col in form_features if col in df.columns])

    elo_features = ['HomeElo', 'AwayElo']
    feature_columns.extend([col for col in elo_features if col in df.columns])

    rest_features = ['HomeRestDays', 'AwayRestDays']
    feature_columns.extend([col for col in rest_features if col in df.columns])

    odds_features = [
        'ImpliedHomeProb','ImpliedDrawProb','ImpliedAwayProb',
        'Avg_Home_Odds','Avg_Draw_Odds','Avg_Away_Odds',
        'Home_Implied_Prob','Draw_Implied_Prob','Away_Implied_Prob'
    ]
    feature_columns.extend([col for col in odds_features if col in df.columns])

    h2h_features = ['H2H_HomeWins','H2H_AwayWins','H2H_HomeGoals','H2H_AwayGoals','H2H_Meetings']
    feature_columns.extend([col for col in h2h_features if col in df.columns])

    gd_features = ['HomeGD_LastN', 'AwayGD_LastN']
    feature_columns.extend([col for col in gd_features if col in df.columns])

    streak_features = ['HomeWinStreak','HomeLoseStreak','AwayWinStreak','AwayLoseStreak']
    feature_columns.extend([col for col in streak_features if col in df.columns])

    ppg_features = [
        'HomePPG_5','HomePPG_10','HomePPG_15',
        'AwayPPG_5','AwayPPG_10','AwayPPG_15'
    ]
    feature_columns.extend([col for col in ppg_features if col in df.columns])

    # Streamlined FIFA features (only 4 core features × 2 teams + 4 difference features = 12 total)
    if fifa_ratings:
        fifa_base_features = ['overall', 'attack', 'midfield', 'defence']

        # Individual team FIFA features
        for prefix in ['Home', 'Away']:
            for feature in fifa_base_features:
                col_name = f'{prefix}FIFA_{feature}'
                if col_name in df.columns:
                    feature_columns.append(col_name)

        # FIFA difference features (often more predictive)
        for feature in fifa_base_features:
            col_name = f'FIFA_{feature}_diff'
            if col_name in df.columns:
                feature_columns.append(col_name)

    # Build feature matrix
    X = df[feature_columns].fillna(0)

    # Target variable
    if target_type == 'result':
        if 'FTR' in df.columns:
            class_mapping = {'H': 2, 'D': 1, 'A': 0}
            y = df['FTR'].map(class_mapping)
        else:
            raise ValueError("FTR column not found for result prediction")
    elif target_type == 'total_goals':
        if all(col in df.columns for col in ['FTHG', 'FTAG']):
            y = df['FTHG'] + df['FTAG']
        else:
            raise ValueError("Goal columns not found for total goals prediction")
    elif target_type == 'over_2_5':
        if all(col in df.columns for col in ['FTHG', 'FTAG']):
            y = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
        else:
            raise ValueError("Goal columns not found for over/under prediction")
    else:
        raise ValueError(f"Unsupported target_type: {target_type}")

    return X, y
