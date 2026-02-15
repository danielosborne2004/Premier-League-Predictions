from scipy.stats import skellam
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

from xgboost import XGBRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def poisson_u25(mu, U25_prob):
    """
    Function used for solving mu from the under 2.5 goals probability.

    Arguments:
        mu (float): expected total goals in the match.
        U25_prob (float): implied probability of under 2.5 goals.

    Returns:
        float: difference between Poisson under 2.5 probability and U25_prob.
    """
    return np.exp(-mu) * (0.5*mu**2 + mu + 1) - U25_prob


def skellam_loss(phi, mu, draw_prob, win_prob, loss_prob):
    """
    Loss function used to fit phi using a Skellam goal difference model.

    Arguments:
        phi (float): proportion of total goals allocated to the home team.
        mu (float): expected total goals in the match.
        draw_prob (float): implied probability of a draw.
        win_prob (float): implied probability of a home win.
        loss_prob (float): implied probability of an away win.

    Returns:
        float: squared error loss between Skellam probabilities and market probabilities.
    """
    lambda_h = mu * phi
    lambda_a = mu * (1-phi)
    
    draw_loss = (skellam.pmf(0, lambda_h, lambda_a) - draw_prob) ** 2
    win_loss = ((1 - skellam.cdf(0, lambda_h, lambda_a)) - win_prob) ** 2
    loss_loss = (skellam.cdf(-1, lambda_h, lambda_a) - loss_prob) ** 2
    return draw_loss + win_loss + loss_loss


def target_vector_construction(games_original):
    """
    Append expected home goals and expected away goals to a match dataframe.

    For each match, the function uses under/over odds to solve for total expected goals,
    then uses 1X2 odds to split that total into home and away expected goals.

    Arguments:
        games_original (pd.DataFrame): match dataframe containing odds columns.

    Returns:
        pd.DataFrame: copy of input with added columns ['XGHome', 'XGAway'].
    """
    games = games_original.copy()
    xg_home = []
    xg_away = []

    for index, game in games.iterrows():

        # Work out implied under 2.5 probability from odds
        if pd.isna(game['P<2.5']) or pd.isna(game['P>2.5']):
            U25 = 1/game['Avg<2.5'] / (1/game['Avg>2.5'] + 1/game['Avg<2.5'])
        else:
            U25 = 1/game['P<2.5'] / (1/game['P<2.5'] + 1/game['P>2.5'])
        
        # Work out implied match outcome probabilities from odds
        draw_prob = 1/game['PSD'] / (1/game['PSD'] + 1/game['PSH'] + 1/game['PSA'])
        win_prob = 1/game['PSH'] / (1/game['PSD'] + 1/game['PSH'] + 1/game['PSA'])
        loss_prob = 1/game['PSA'] / (1/game['PSD'] + 1/game['PSH'] + 1/game['PSA'])
    
        # Solve for mu, the total expected goals
        u25_solution = root_scalar(poisson_u25, args=(U25, ), bracket=[0, 20], method='brentq')
        mu = u25_solution.root

        # Solve for phi, the home share of total expected goals
        skellam_solution = minimize_scalar(skellam_loss, args=(mu, draw_prob, win_prob, loss_prob), bounds=[0.001, 0.999], method='bounded')
        phi = skellam_solution.x

        # Save expected goals for home and away
        xg_home.append(phi * mu)
        xg_away.append((1-phi) * mu)

    games['XGHome'] = xg_home
    games['XGAway'] = xg_away

    return games


def version1_dataset(games_uncomplete):
    """
    Convert match-level data into a team-level dataset with one row per team per match.

    Creates a home view and away view of each match so every match becomes two rows.
    Also renames match statistics so they are from the perspective of 'Team'.

    Arguments:
        games_uncomplete (pd.DataFrame): raw match dataframe (one row per match).

    Returns:
        pd.DataFrame: cleaned team-level dataset sorted by date and time.
    """

    # Compute expected goals targets from betting markets
    games =  target_vector_construction(games_uncomplete)
    
    # Copy dataframes for home and away perspectives
    home_games = games.copy()
    away_games = games.copy()

    # Rename columns for the home team perspective
    home_games.rename(columns = {"HomeTeam": "Team", "AwayTeam": "Opponent", "XGHome":"PredictedXG", 'FTHG': 'Team Goals',
                                'FTAG': 'Opponent Goals', 'HS': 'Team Shots', 'AS': 'Opponent Shots', 'HST': 'Team SOT', 
                                'AST': 'Opponent SOT', 'HC': 'Team Corners', 'AC': 'Opponent Corners', 'HF': 'Team Fouls',
                                'AF': 'Opponent Fouls', 'HY': 'Team Yellow Cards', 'AY': 'Opponent Yellow Cards', 
                                'HR': 'Team Red Cards', 'AR': 'Opponent Red Cards'}, inplace=True)
    home_games.drop(columns={"XGAway"}, inplace=True)
    home_games['Location'] = "Home"

    # Rename columns for the away team perspective
    away_games.rename(columns = {"HomeTeam": "Opponent", "AwayTeam": "Team", "XGAway":"PredictedXG", 'FTHG': 'Opponent Goals',
                                'FTAG': 'Team Goals', 'HS': 'Opponent Shots', 'AS': 'Team Shots', 'HST': 'Opponent SOT', 
                                'AST': 'Team SOT', 'HC': 'Opponent Corners', 'AC': 'Team Corners', 'HF': 'Opponent Fouls',
                                'AF': 'Team Fouls', 'HY': 'Opponent Yellow Cards', 'AY': 'Team Yellow Cards', 
                                'HR': 'Opponent Red Cards', 'AR': 'Team Red Cards'}, inplace=True)
    away_games.drop(columns={"XGHome"}, inplace=True)
    away_games['Location'] = "Away"

    # Combine home and away views into one dataset
    cleaned_df = pd.concat([home_games, away_games])
    cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"], format="%d/%m/%Y")
    cleaned_df["Weekday"] = cleaned_df["Date"].dt.day_name()
    cleaned_df["Time"] = pd.to_datetime(cleaned_df["Time"]).dt.hour
    cleaned_df = cleaned_df.sort_values(by=["Date", "Time"], ascending=[True, True]).reset_index()
    
    return cleaned_df.loc[:, ["Date", "Weekday", "Time", "Season", "Team", "Opponent", "Location", "Team Goals", "Opponent Goals", "Team Shots", "Opponent Shots", 
                              "Team SOT", "Opponent SOT", "Team Corners", "Opponent Corners", 'Team Fouls', 'Opponent Fouls', 'Team Yellow Cards',
                              'Opponent Yellow Cards', 'Team Red Cards', 'Opponent Red Cards', "PredictedXG"]]


def get_schedule(current_season):
    """
    Load schedule.csv and return only fixtures after the current season data.
    Output is team-level (two rows per fixture).
    """
    max_date = current_season["Date"].max()

    schedule = pd.read_csv("Data/schedule.csv")

    dt = pd.to_datetime(schedule["Date"], dayfirst=True, errors="coerce")
    schedule["Time"] = dt.dt.strftime("%H:%M")
    schedule["Date"] = dt.dt.normalize()

    schedule = schedule[schedule["Date"] > max_date]

    home_df = schedule.copy()
    away_df = schedule.copy()

    home_df["Location"] = "Home"
    home_df = home_df.rename(columns={"Home Team": "Team", "Away Team": "Opponent"})
    home_df = home_df.drop(columns=["Match Number", "Round Number", "Result"])

    away_df["Location"] = "Away"
    away_df = away_df.rename(columns={"Home Team": "Opponent", "Away Team": "Team"})
    away_df = away_df.drop(columns=["Match Number", "Round Number", "Result"])

    output = pd.concat([home_df, away_df]).sort_values(by="Date", ascending=True).reset_index(drop=True)

    output.loc[output["Team"] == "Man Utd", "Team"] = "Man United"
    output.loc[output["Team"] == "Spurs", "Team"] = "Tottenham"
    output.loc[output["Opponent"] == "Man Utd", "Opponent"] = "Man United"
    output.loc[output["Opponent"] == "Spurs", "Opponent"] = "Tottenham"

    return output


def rolling_features(df, stats_window, betting_window):
    """
    Create rolling features for match statistics and strength metrics.

    All rolling features are shifted by one match so they only use past information.

    Arguments:
        df (pd.DataFrame): team-level dataset.
        stats_window (int): rolling window length for match-stat features.
        betting_window (int): rolling window length for PredictedXG strength features.

    Returns:
        pd.DataFrame: dataframe with rolling feature columns added and rows with missing values removed.
    """
    attacking_cols = {
        'Team Goals': 'att_goals',
        'Team Shots': 'att_shots',
        'Team SOT': 'att_sot',
        'Team Corners': 'att_corners',
    }

    # Rolling attacking stats for the team
    for col, new_name in attacking_cols.items():
        df[new_name] = (
            df.groupby(['Season', 'Team'])[col]
              .transform(lambda x: x.shift(1).rolling(stats_window).mean())
        )

    defensive_cols = {
        'Opponent Goals': 'def_goals_conceded',
        'Opponent Shots': 'def_shots_conceded',
        'Opponent SOT': 'def_sot_conceded',
        'Opponent Corners': 'def_corners_conceded'
    }

    # Rolling defensive stats for the team, which will be merged back as opponent features
    for col, new_name in defensive_cols.items():
        df[new_name] = (
            df.groupby(['Season', 'Team'])[col]
              .transform(lambda x: x.shift(1).rolling(stats_window).mean())
        )

    def_features = list(defensive_cols.values())

    # Rename the defensive features so they attach to the opponent
    defensive_df = (
        df[['Season', 'Date', 'Team'] + def_features]
        .rename(columns={'Team': 'Opponent'})
        .rename(columns={f: f.replace('def_', 'opp_def_') for f in def_features})
    )

    # Merge opponent defensive features onto the current row
    df = df.merge(
        defensive_df,
        on=['Season', 'Date', 'Opponent'],
        how='left'
    )

    # Remove the temporary team defensive columns
    df = df.drop(columns=def_features)

    # League average PredictedXG used as a baseline
    league_avg = (
        df.groupby('Season')['PredictedXG']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Rolling attack strength for the team
    df['attack_strength'] = (
        df.groupby('Team')['PredictedXG']
        .transform(lambda x: x.shift(1).rolling(betting_window, min_periods=10).mean())
        / league_avg
    )

    # Rolling defence strength for the opponent
    df['defence_strength'] = (
        df.groupby('Opponent')['PredictedXG']
        .transform(lambda x: x.shift(1).rolling(betting_window, min_periods=10).mean())
        / league_avg
    ) 

    weights = {'fouls': 1.0, 'yellow': 3.0, 'red': 6.0}

    # Rolling aggression score for the team
    df['agg_for'] = (
        weights['fouls'] * df.groupby(['Season', 'Team'])['Team Fouls']
            .transform(lambda x: x.shift(1).rolling(stats_window).mean())
        + weights['yellow'] * df.groupby(['Season', 'Team'])['Team Yellow Cards']
            .transform(lambda x: x.shift(1).rolling(stats_window).mean())
        + weights['red'] * df.groupby(['Season', 'Team'])['Team Red Cards']
            .transform(lambda x: x.shift(1).rolling(stats_window).mean())
    )

    # Copy team aggression score onto the opponent for the current row
    opp_agg_df = (
        df[['Season', 'Date', 'Team', 'agg_for']]
        .rename(columns={'Team': 'Opponent', 'agg_for': 'opp_agg_for'})
    )

    df = df.merge(
        opp_agg_df,
        on=['Season', 'Date', 'Opponent'],
        how='left'
    )

    # Drop early-season rows where rolling windows do not exist yet
    df = df.dropna().reset_index(drop=True)

    return df


def enriching_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add season-level promoted and Europe flags for both Team and Opponent.

    Arguments:
        df (pd.DataFrame): dataframe containing columns ['Season', 'Team', 'Opponent'].

    Returns:
        pd.DataFrame: copy of df with added columns
        ['TeamPromoted', 'OppPromoted', 'TeamInEurope', 'OppInEurope'].
    """
    df = df.copy()

    promoted_by_season = {
        '20-21': ['Leeds', 'West Brom', 'Fulham'],
        '21-22': ['Norwich', 'Watford', 'Brentford'],
        '22-23': ['Fulham', 'Bournemouth', "Nott'm Forest"],
        '23-24': ['Burnley', 'Sheffield United', 'Luton'],
        '24-25': ['Leicester', 'Ipswich', 'Southampton'],
        '25-26': ['Sunderland', 'Burnley', 'Leeds'],
    }

    europe_by_season = {
        '20-21': ['Man City', 'Liverpool', 'Chelsea', 'Man United', 'Tottenham', 'Arsenal', 'Leicester'],
        '21-22': ['Man City', 'Liverpool', 'Chelsea', 'Man United', 'Leicester', 'West Ham', 'Tottenham'],
        '22-23': ['Man City', 'Liverpool', 'Chelsea', 'Tottenham', 'Arsenal', 'Man United', 'West Ham'],
        '23-24': ['Man City', 'Arsenal', 'Man United', 'Newcastle', 'Liverpool', 'Brighton', 'West Ham', 'Aston Villa'],
        '24-25': ['Man City', 'Arsenal', 'Liverpool', 'Aston Villa', 'Tottenham', 'Chelsea', 'Man United'],
        '25-26': ['Liverpool', 'Arsenal', 'Man City', 'Chelsea', 'Newcastle United', 'Tottenham Hotspur',
                  'Aston Villa', 'Crystal Palace', 'Nottingham Forest'],
    }

    # Build a season-club lookup table so Team and Opponent can use the same flags
    all_season_teams = (
        pd.concat([
            df[['Season', 'Team']].rename(columns={'Team': 'Club'}),
            df[['Season', 'Opponent']].rename(columns={'Opponent': 'Club'})
        ])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    promoted_sets = {s: set(v) for s, v in promoted_by_season.items()}
    europe_sets = {s: set(v) for s, v in europe_by_season.items()}

    # Mark promoted and Europe teams for each season
    all_season_teams['Promoted'] = all_season_teams.apply(
        lambda r: int(r['Club'] in promoted_sets.get(r['Season'], set())),
        axis=1
    )
    all_season_teams['InEurope'] = all_season_teams.apply(
        lambda r: int(r['Club'] in europe_sets.get(r['Season'], set())),
        axis=1
    )

    # Merge flags for Team
    df = df.merge(
        all_season_teams.rename(columns={'Club': 'Team', 'Promoted': 'TeamPromoted', 'InEurope': 'TeamInEurope'}),
        on=['Season', 'Team'],
        how='left'
    )

    # Merge flags for Opponent
    df = df.merge(
        all_season_teams.rename(columns={'Club': 'Opponent', 'Promoted': 'OppPromoted', 'InEurope': 'OppInEurope'}),
        on=['Season', 'Opponent'],
        how='left'
    )

    # Fill missing values for clubs not in the lookup lists
    for c in ['TeamPromoted', 'OppPromoted', 'TeamInEurope', 'OppInEurope']:
        df[c] = df[c].fillna(0).astype(int)

    return df


def prepare_data(stats_window, betting_window):
    """
    Load season CSV files and build the final modelling dataframe.

    This creates the team-level dataset, adds rolling features, adds enrichment flags,
    and encodes categorical columns as integer codes.

    Arguments:
        stats_window (int): rolling window length for match-stat features.
        betting_window (int): rolling window length for PredictedXG strength features.

    Returns:
        pd.DataFrame: final dataframe used for model training.
    """

    # Read in the season CSV files
    df20 = pd.read_csv('Data/PL2020.csv')
    df20['Season'] = '20-21'
    df21 = pd.read_csv('Data/PL2021.csv')
    df21['Season'] = '21-22'
    df22 = pd.read_csv('Data/PL2022.csv')
    df22['Season'] = '22-23'
    df23 = pd.read_csv('Data/PL2023.csv')
    df23['Season'] = '23-24'
    df24 = pd.read_csv('Data/PL2024.csv')
    df24['Season'] = '24-25'
    df25 = pd.read_csv('Data/PL2025.csv')
    df25['Season'] = '25-26'

    # Concatenate all seasons and construct the dataset
    df = version1_dataset(pd.concat([df20, df21, df22, df23, df24, df25], ignore_index=True))

    # Add rolling features and enrichment flags
    df = rolling_features(df, stats_window, betting_window)
    df = enriching_features(df)

    # Encode categorical columns
    df['location_code'] = df['Location'].astype('category').cat.codes
    df['opp_code'] = df['Opponent'].astype('category').cat.codes
    df['team_code'] = df['Team'].astype('category').cat.codes
    df['season_code'] = df['Season'].astype('category').cat.codes
    df['weekday_code'] = df['Weekday'].astype('category').cat.codes

    return df

def expanding_window_len(horizon, start_window, max_window, growth_rate):
    """
    Returns the rolling window length to use for a given horizon.

    Arguments:
        horizon (int): 1 = next match, 2 = second match ahead, etc.
        start_window (int): window length used for the next match.
        max_window (int): maximum window length allowed.
        growth_rate (float): controls how quickly the window expands toward max_window.

    Returns:
        window (int): rolling window length to use.
    """
    window = start_window + (max_window - start_window) * (1 - np.exp(-growth_rate * (horizon - 1)))
    window = int(np.ceil(window))
    return min(window, max_window)


def get_historic_schedule(season_raw_df, cutoff_date):
    """
    Creates the same style output as get_schedule() but for historic seasons, using the raw match list.

    Arguments:
        season_raw_df (pd.DataFrame): raw match-level dataframe for a season (one row per match).
        cutoff_date (str|pd.Timestamp): date separating played vs future fixtures.

    Returns:
        pd.DataFrame: team-level future fixtures with columns:
                      ['Date', 'Time'(optional), 'Team', 'Opponent', 'Location']
    """
    df = season_raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.normalize()
    cutoff = pd.to_datetime(cutoff_date).normalize()

    df_future = df[df["Date"] > cutoff].sort_values(["Date"]).copy()

    home_rows = pd.DataFrame({
        "Date": df_future["Date"].values,
        "Team": df_future["HomeTeam"].values,
        "Opponent": df_future["AwayTeam"].values,
        "Location": "Home",
    })

    away_rows = pd.DataFrame({
        "Date": df_future["Date"].values,
        "Team": df_future["AwayTeam"].values,
        "Opponent": df_future["HomeTeam"].values,
        "Location": "Away",
    })

    if "Time" in df_future.columns:
        home_rows["Time"] = df_future["Time"].values
        away_rows["Time"] = df_future["Time"].values

    schedule = pd.concat([home_rows, away_rows], ignore_index=True)

    sort_cols = ["Date", "Team"]
    if "Time" in schedule.columns:
        sort_cols = ["Date", "Time", "Team"]

    schedule = schedule.sort_values(sort_cols).reset_index(drop=True)
    return schedule


def prepare_data_future(stats_start, stats_max, stats_growth, bet_start, bet_max, bet_growth, cutoff_date=None):
    """
    Build the modelling dataframe for future fixtures by dynamically populating rolling features
    using only information available up to a cutoff date.

    Key modelling choices:
    - Horizon is defined per team for future fixtures: 1 = next match after cutoff, 2 = second, etc.
    - Window lengths for a fixture are determined by that row's horizon (h_team).
    - Opponent uses the SAME window lengths (W_stats, W_bet) for that fixture (symmetry).
    - defence_strength is computed to MATCH rolling_features(): it uses a series grouped by 'Opponent'
      (not by 'Team'), so we build an opp_bet_cache for that.

    Arguments:
        stats_start (int): rolling window length for next match stats (horizon=1).
        stats_max (int): maximum rolling window length for stats.
        stats_growth (float): expansion rate for stats window.
        bet_start (int): rolling window length for next match betting features (horizon=1).
        bet_max (int): maximum rolling window length for betting window.
        bet_growth (float): expansion rate for betting window.
        cutoff_date (str|pd.Timestamp|None): if None, uses current point in 25-26 season.
                                            if provided, builds future fixtures from that point in the matching season.

    Returns:
        pd.DataFrame: future fixtures dataframe with populated features and codes.
    """

    df20 = pd.read_csv("Data/PL2020.csv"); df20["Season"] = "20-21"
    df21 = pd.read_csv("Data/PL2021.csv"); df21["Season"] = "21-22"
    df22 = pd.read_csv("Data/PL2022.csv"); df22["Season"] = "22-23"
    df23 = pd.read_csv("Data/PL2023.csv"); df23["Season"] = "23-24"
    df24 = pd.read_csv("Data/PL2024.csv"); df24["Season"] = "24-25"
    df25 = pd.read_csv("Data/PL2025.csv"); df25["Season"] = "25-26"

    raw_all = pd.concat([df20, df21, df22, df23, df24, df25], ignore_index=True)
    raw_all["Date"] = pd.to_datetime(raw_all["Date"], dayfirst=True, errors="coerce").dt.normalize()

    df = version1_dataset(raw_all.copy())
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

    if cutoff_date is None:
        season_to_use = "25-26"
        cutoff = df[df["Season"] == season_to_use]["Date"].max()
    else:
        cutoff = pd.to_datetime(cutoff_date).normalize()
        season_ranges = df.groupby("Season")["Date"].agg(["min", "max"]).reset_index()
        season_match = season_ranges[(season_ranges["min"] <= cutoff) & (cutoff <= season_ranges["max"])]

        if len(season_match) == 0:
            raise ValueError("cutoff_date does not fall inside any season date range in your data.")

        season_to_use = season_match.iloc[0]["Season"]

    season_df = df[df["Season"] == season_to_use].copy()
    season_so_far = season_df[season_df["Date"] <= cutoff].copy()
    season_so_far = season_so_far.sort_values(["Date", "Time"]).reset_index(drop=True)

    if cutoff_date is None:
        future = get_schedule(season_so_far).copy()
        future["Season"] = season_to_use
    else:
        season_raw = raw_all[raw_all["Season"] == season_to_use].copy()
        future = get_historic_schedule(season_raw, cutoff).copy()
        future["Season"] = season_to_use

    future["Date"] = pd.to_datetime(future["Date"], dayfirst=True, errors="coerce").dt.normalize()
    future["Weekday"] = future["Date"].dt.day_name()

    if "Time" in future.columns:
        future["Time"] = pd.to_datetime(future["Time"], errors="coerce").dt.hour
    else:
        future["Time"] = np.nan

    future = future.sort_values(["Date", "Time", "Team"]).reset_index(drop=True)

    future["h_team"] = future.groupby(["Season", "Team"]).cumcount() + 1

    hist_stats = season_so_far.copy()
    hist_bet = df[df["Date"] <= cutoff].copy()
    hist_bet = hist_bet.sort_values(["Date", "Time"]).reset_index(drop=True)

    if len(hist_stats) <= 1:
        league_avg_pxg = np.nan
    else:
        league_avg_pxg = hist_stats["PredictedXG"].iloc[:-1].mean()

    weights = {"fouls": 1.0, "yellow": 3.0, "red": 6.0}

    def last_w_mean(values, w, min_periods=None):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if min_periods is not None and len(values) < min_periods:
            return np.nan
        if len(values) < w:
            return np.nan
        return values[-w:].mean()

    team_stats_cache = {t: g.sort_values(["Date", "Time"]) for t, g in hist_stats.groupby("Team")}
    team_bet_cache = {t: g.sort_values(["Date", "Time"]) for t, g in hist_bet.groupby("Team")}
    opp_bet_cache  = {o: g.sort_values(["Date", "Time"]) for o, g in hist_bet.groupby("Opponent")}

    att_goals, att_shots, att_sot, att_corners = [], [], [], []
    opp_def_goals, opp_def_shots, opp_def_sot, opp_def_corners = [], [], [], []
    attack_strength, defence_strength = [], []
    agg_for, opp_agg_for = [], []

    for r in future.itertuples(index=False):
        W_stats = expanding_window_len(int(r.h_team), stats_start, stats_max, stats_growth)
        W_bet = expanding_window_len(int(r.h_team), bet_start, bet_max, bet_growth)

        tg = team_stats_cache.get(r.Team)
        og = team_stats_cache.get(r.Opponent)

        tb = team_bet_cache.get(r.Team)
        ob_def = opp_bet_cache.get(r.Opponent)

        if tg is None or og is None or tb is None or ob_def is None:
            att_goals.append(np.nan); att_shots.append(np.nan); att_sot.append(np.nan); att_corners.append(np.nan)
            opp_def_goals.append(np.nan); opp_def_shots.append(np.nan); opp_def_sot.append(np.nan); opp_def_corners.append(np.nan)
            attack_strength.append(np.nan); defence_strength.append(np.nan)
            agg_for.append(np.nan); opp_agg_for.append(np.nan)
            continue

        att_goals.append(last_w_mean(tg["Team Goals"].to_numpy(), W_stats))
        att_shots.append(last_w_mean(tg["Team Shots"].to_numpy(), W_stats))
        att_sot.append(last_w_mean(tg["Team SOT"].to_numpy(), W_stats))
        att_corners.append(last_w_mean(tg["Team Corners"].to_numpy(), W_stats))

        opp_def_goals.append(last_w_mean(og["Opponent Goals"].to_numpy(), W_stats))
        opp_def_shots.append(last_w_mean(og["Opponent Shots"].to_numpy(), W_stats))
        opp_def_sot.append(last_w_mean(og["Opponent SOT"].to_numpy(), W_stats))
        opp_def_corners.append(last_w_mean(og["Opponent Corners"].to_numpy(), W_stats))

        team_pxg = last_w_mean(tb["PredictedXG"].to_numpy(), W_bet, min_periods=10)
        opp_pxg_def = last_w_mean(ob_def["PredictedXG"].to_numpy(), W_bet, min_periods=10)

        if np.isnan(league_avg_pxg) or league_avg_pxg == 0:
            attack_strength.append(np.nan)
            defence_strength.append(np.nan)
        else:
            attack_strength.append(team_pxg / league_avg_pxg)
            defence_strength.append(opp_pxg_def / league_avg_pxg)

        fouls = last_w_mean(tg["Team Fouls"].to_numpy(), W_stats)
        yell = last_w_mean(tg["Team Yellow Cards"].to_numpy(), W_stats)
        red = last_w_mean(tg["Team Red Cards"].to_numpy(), W_stats)
        agg_for.append(weights["fouls"] * fouls + weights["yellow"] * yell + weights["red"] * red)

        ofouls = last_w_mean(og["Team Fouls"].to_numpy(), W_stats)
        oyell = last_w_mean(og["Team Yellow Cards"].to_numpy(), W_stats)
        ored = last_w_mean(og["Team Red Cards"].to_numpy(), W_stats)
        opp_agg_for.append(weights["fouls"] * ofouls + weights["yellow"] * oyell + weights["red"] * ored)

    future["att_goals"] = att_goals
    future["att_shots"] = att_shots
    future["att_sot"] = att_sot
    future["att_corners"] = att_corners

    future["opp_def_goals_conceded"] = opp_def_goals
    future["opp_def_shots_conceded"] = opp_def_shots
    future["opp_def_sot_conceded"] = opp_def_sot
    future["opp_def_corners_conceded"] = opp_def_corners

    future["attack_strength"] = attack_strength
    future["defence_strength"] = defence_strength

    future["agg_for"] = agg_for
    future["opp_agg_for"] = opp_agg_for

    future = enriching_features(future)

    future["location_code"] = future["Location"].astype("category").cat.codes
    future["opp_code"] = future["Opponent"].astype("category").cat.codes
    future["team_code"] = future["Team"].astype("category").cat.codes
    future["season_code"] = future["Season"].astype("category").cat.codes
    future["weekday_code"] = future["Weekday"].astype("category").cat.codes

    future = future.dropna().reset_index(drop=True)
    return future


def get_current_table():
    """
    Construct the current Premier League table from the 2025–26 results dataset.

    The function loads match results, computes points from match outcomes,
    aggregates goals for/against and points by team, and returns the league
    table sorted by standard ranking rules (Points, Goal Difference, Goals For).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by Team containing:
        - GF : Goals For
        - GA : Goals Against
        - GD : Goal Difference
        - Pts : Total Points
    """
    df25 = pd.read_csv('Data/PL2025.csv')
    df25['Season'] = '25-26'
    data_all = version1_dataset(df25)

    data_all["Pts"] = np.select(
    [
        data_all["Team Goals"] > data_all["Opponent Goals"],
        data_all["Team Goals"] == data_all["Opponent Goals"]
    ], [3, 1],default=0)

    table = (
        data_all.groupby("Team")
        .agg(
            GF=("Team Goals", "sum"),
            GA=("Opponent Goals", "sum"), 
            Pts=("Pts", "sum")))

    table["GD"] = table["GF"] - table["GA"]

    table = table.sort_values(["Pts", "GD", "GF"],ascending=False)

    return table[['GF', 'GA', 'GD', 'Pts']]

def future_xg_model():
    """
    Train the expected goals (xG) model and generate predictions for future fixtures.

    The function trains an XGBoost regression model on historical match data to
    predict expected goals, then applies the trained model to future fixtures
    using expanding-window features.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing predicted expected goals for future matches with:
        - Season
        - Date
        - Team
        - Opponent
        - Location
        - Predictions (predicted xG)
    """
    predictors = ['Time', 'att_goals', 'att_shots', 'att_sot', 'att_corners', 
              'opp_def_goals_conceded', 'opp_def_shots_conceded', 
              'opp_def_sot_conceded', 'opp_def_corners_conceded', 
              'attack_strength', 'defence_strength', 
              'agg_for', 'opp_agg_for', 
              'TeamPromoted', 'TeamInEurope', 
              'OppPromoted', 'OppInEurope', 
              'location_code', 'opp_code', 
              'team_code', 'season_code', 
              'weekday_code']

    data_current = prepare_data(6, 14)
    train = data_current

    test = prepare_data_future(stats_start=6, stats_max=6, stats_growth=0.1,
        bet_start=14, bet_max=16, bet_growth=0.04,).copy()

    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.7,
        random_state=1
    )

    xgb.fit(train[predictors], train["PredictedXG"])
    preds = xgb.predict(test[predictors])

    prediction_df = test.copy()[['Season', 'Date', 'Team', 'Opponent', 'Location']]
    prediction_df['Predictions'] = preds

    return prediction_df


def get_full_fixtures(future):
    """
    Combine home and away xG predictions into a single fixtures dataset.

    The function merges the future home and away predictions so each row
    represents a complete fixture with both teams’ expected goals (lambdas)
    for simulation.

    Parameters
    ----------
    future : pandas.DataFrame
        Output from future_xg_model() containing predicted xG for each team
        and match location.

    Returns
    -------
    pandas.DataFrame
        Fixture-level dataset containing:
        - Season
        - Date
        - HomeTeam
        - AwayTeam
        - lambda_home : expected goals for the home team
        - lambda_away : expected goals for the away team
    """
    future_home = future[future["Location"] == "Home"].copy()
    future_away = future[future["Location"] == "Away"].copy()

    fixtures = future_home.merge(
        future_away,
        left_on=["Season", "Date", "Team", "Opponent"],
        right_on=["Season", "Date", "Opponent", "Team"],
        suffixes=("_home", "_away")
    )

    fixtures["HomeTeam"] = fixtures["Team_home"]
    fixtures["AwayTeam"] = fixtures["Opponent_home"]
    fixtures["lambda_home"] = fixtures["Predictions_home"]
    fixtures["lambda_away"] = fixtures["Predictions_away"]

    fixtures = fixtures[["Season", "Date", "HomeTeam", "AwayTeam", "lambda_home", "lambda_away"]]
    return fixtures


def simulate_one_season(fixtures):
    """
    Simulate the remainder of a Premier League season using Poisson goal sampling.

    Starting from the current league table, the function simulates each remaining
    fixture by sampling home and away goals from Poisson distributions using the
    predicted expected goals (lambdas). The league table is updated after every
    simulated match and then sorted to produce the final standings.

    Parameters
    ----------
    fixtures : pandas.DataFrame
        Fixture-level dataset containing predicted goal rates from
        get_full_fixtures(), including:
        - HomeTeam
        - AwayTeam
        - lambda_home
        - lambda_away

    Returns
    -------
    pandas.DataFrame
        Final simulated league table including:
        - GF : Goals For
        - GA : Goals Against
        - GD : Goal Difference
        - Pts : Points
        - Position : Final league position
    """
    table = get_current_table()

    for r in fixtures.itertuples(index=False):
        home = r.HomeTeam
        away = r.AwayTeam

        gh = np.random.poisson(r.lambda_home)
        ga = np.random.poisson(r.lambda_away)

        table.loc[home, "GF"] += gh
        table.loc[home, "GA"] += ga
        table.loc[away, "GF"] += ga
        table.loc[away, "GA"] += gh

        table.loc[home, "GD"] = table.loc[home, "GF"] - table.loc[home, "GA"]
        table.loc[away, "GD"] = table.loc[away, "GF"] - table.loc[away, "GA"]

        if gh > ga:
            table.loc[home, "Pts"] += 3
        elif gh < ga:
            table.loc[away, "Pts"] += 3
        else:
            table.loc[home, "Pts"] += 1
            table.loc[away, "Pts"] += 1

    final = table.sort_values(["Pts", "GD", "GF"], ascending=False).copy()
    final["Position"] = np.arange(1, len(final) + 1)
    return final


def simulate_seasons(num_simulations):
    """
    Run multiple season simulations to estimate finishing position probabilities.

    The function repeatedly simulates the remainder of the season using the
    Poisson-based match simulator and tracks how often each team finishes in
    each league position. The counts are converted into percentage probabilities.

    Parameters
    ----------
    num_simulations : int
        Number of season simulations to run.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by Team with columns representing league positions
        (1-20). Each cell contains the probability (%) of finishing in that
        position.
    """
    table_start = get_current_table()
    future = future_xg_model()
    fixtures = get_full_fixtures(future)

    teams = table_start.index.tolist()
    pos_counts = pd.DataFrame(0, index=teams, columns=np.arange(1, len(teams) + 1))

    for _ in range(num_simulations):
        final = simulate_one_season(fixtures)
        for team, pos in final["Position"].items():
            pos_counts.loc[team, pos] += 1
        
    pos_probs = np.round((pos_counts / num_simulations) * 100, 2)

    return pos_probs


def plot_final_matrix(gtable):
    """
    Clean plotting for the final matrix.
    """
    plt.figure(figsize=(16, 10))

    ax = sns.heatmap(
        gtable,
        cmap="Reds",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar=False
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.title("Premier League Finish Probabilities (%)", pad=40)
    plt.xlabel("Position")
    plt.ylabel("Team")

    plt.tight_layout()
    plt.show()


def _market_df_from_series(series, title, collapse_zeros=True, zero_label="Any other team"):
    """
    Takes a Series indexed by Team with values as probabilities (%),
    returns a tidy DataFrame for presentation/export.

    Output columns: Team, <title>
    """
    s = series.copy()

    if collapse_zeros:
        zeros = s[s == 0]
        nonzeros = s[s != 0].copy()
        if len(zeros) > 0:
            nonzeros.loc[zero_label] = 0.0
        s = nonzeros

    out = (
        s.sort_values(ascending=False)
         .reset_index()
         .rename(columns={"index": "Team"})
    )
    out.columns = ["Team", title]
    return out


def df_finish_1st(gtable):
    # Position 1
    return _market_df_from_series(gtable[1], "Winning The League (%)")


def df_finish_top4(gtable):
    # Positions 1-4
    series = gtable.loc[:, 1:4].sum(axis=1)
    return _market_df_from_series(series, "Finish Top 4 (%)")


def df_finish_top5(gtable):
    # Positions 1-5
    series = gtable.loc[:, 1:5].sum(axis=1)
    return _market_df_from_series(series, "Finish Top 5 (%)")


def df_finish_top10(gtable):
    # Positions 1-10
    series = gtable.loc[:, 1:10].sum(axis=1)
    return _market_df_from_series(series, "Finish Top 10 (%)")


def df_finish_bottom10(gtable):
    # Positions 11-20
    series = gtable.loc[:, 11:20].sum(axis=1)
    return _market_df_from_series(series, "Finish Bottom 10 (%)")


def df_finish_bottom3(gtable):
    # Positions 18-20
    series = gtable.loc[:, 18:20].sum(axis=1)
    return _market_df_from_series(series, "Relegation (%)")