from scipy.stats import skellam
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

import numpy as np
import pandas as pd


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
    Load the schedule.csv file and return only fixtures after the current season data.

    The output is formatted so each fixture becomes two rows, one for the home team and
    one for the away team.

    Arguments:
        current_season (pd.DataFrame): dataframe containing at least a 'Date' column.

    Returns:
        pd.DataFrame: upcoming fixtures with Team, Opponent, Location and Date.
    """
    max_date = current_season['Date'].max()

    schedule = pd.read_csv('Data/schedule.csv')
    schedule['Date'] = pd.to_datetime(schedule['Date']).dt.normalize()

    # Filter out fixtures that have already happened
    schedule = schedule[schedule['Date'] > max_date]

    home_df = schedule.copy()
    away_df = schedule.copy()

    # Create home team rows
    home_df['Location'] = 'Home'
    home_df = home_df.rename(columns={'Home Team': 'Team', 'Away Team': 'Opponent'})
    home_df = home_df.drop(columns=['Match Number', 'Round Number', 'Result'])

    # Create away team rows
    away_df['Location'] = 'Away'
    away_df = away_df.rename(columns={'Home Team': 'Opponent', 'Away Team': 'Team'})
    away_df = away_df.drop(columns=['Match Number', 'Round Number', 'Result'])

    output = pd.concat([home_df, away_df]).sort_values(by='Date', ascending=True).reset_index().drop(columns=['index'])

    # Align team names to match the training dataset
    output.loc[output['Team'] == 'Man Utd', 'Team'] = 'Man United'
    output.loc[output['Team'] == 'Spurs', 'Team'] = 'Tottenham'
    output.loc[output['Opponent'] == 'Man Utd', 'Opponent'] = 'Man United'
    output.loc[output['Opponent'] == 'Spurs', 'Opponent'] = 'Tottenham'

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
