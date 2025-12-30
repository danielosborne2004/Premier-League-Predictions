from scipy.stats import skellam
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

import numpy as np
import pandas as pd


def poisson_u25(mu, U25_prob):
    return np.exp(-mu) * (0.5*mu**2 + mu + 1) - U25_prob

def skellam_loss(phi, mu, draw_prob, win_prob, loss_prob):
    lambda_h = mu * phi
    lambda_a = mu * (1-phi)
    
    draw_loss = (skellam.pmf(0, lambda_h, lambda_a) - draw_prob) ** 2
    win_loss = ((1 - skellam.cdf(0, lambda_h, lambda_a)) - win_prob) ** 2
    loss_loss = (skellam.cdf(-1, lambda_h, lambda_a) - loss_prob) ** 2
    return draw_loss + win_loss + loss_loss

def target_vector_construction(games_original):
    """
    Takes a single dataframe and appends expected home goals and expected away goals: XGHome, XGAway
    """

    games = games_original.copy()
    xg_home = []
    xg_away = []
    for index, game in games.iterrows():
        if pd.isna(game['P<2.5']) or pd.isna(game['P>2.5']):
            U25 = 1/game['Avg<2.5'] / (1/game['Avg>2.5'] + 1/game['Avg<2.5'])
        else:
            U25 = 1/game['P<2.5'] / (1/game['P<2.5'] + 1/game['P>2.5'])
        
        draw_prob = 1/game['PSD'] / (1/game['PSD'] + 1/game['PSH'] + 1/game['PSA'])
        win_prob = 1/game['PSH'] / (1/game['PSD'] + 1/game['PSH'] + 1/game['PSA'])
        loss_prob = 1/game['PSA'] / (1/game['PSD'] + 1/game['PSH'] + 1/game['PSA'])
    
        u25_solution = root_scalar(poisson_u25, args=(U25, ), bracket=[0, 20], method='brentq')
        mu = u25_solution.root

        skellam_solution = minimize_scalar(skellam_loss, args=(mu, draw_prob, win_prob, loss_prob), bounds=[0.001, 0.999], method='bounded')
        phi = skellam_solution.x

        xg_home.append(phi * mu)
        xg_away.append((1-phi) * mu)

    games['XGHome'] = xg_home
    games['XGAway'] = xg_away

    return games

def version1_dataset(games_uncomplete):
    games =  target_vector_construction(games_uncomplete)
    
    home_games = games.copy()
    away_games = games.copy()

    home_games.rename(columns = {"HomeTeam": "Team", "AwayTeam": "Opponent", "XGHome":"PredictedXG", 'FTHG': 'Team Goals',
                                'FTAG': 'Opponent Goals', 'HS': 'Team Shots', 'AS': 'Opponent Shots', 'HST': 'Team SOT', 
                                'AST': 'Opponent SOT', 'HC': 'Team Corners', 'AC': 'Opponent Corners'}, inplace=True)
    home_games.drop(columns={"XGAway"}, inplace=True)
    home_games['Location'] = "Home"
    away_games.rename(columns = {"AwayTeam": "Team", "HomeTeam": "Opponent", "XGAway":"PredictedXG", 'FTHG': 'Opponent Goals',
                                'FTAG': 'Team Goals', 'HS': 'Opponent Shots', 'AS': 'Team Shots', 'HST': 'Opponent SOT', 
                                'AST': 'Team SOT', 'HC': 'Opponent Corners', 'AC': 'Team Corners'}, inplace=True)
    away_games.drop(columns={"XGHome"}, inplace=True)
    away_games['Location'] = "Away"

    cleaned_df = pd.concat([home_games, away_games])
    cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"], format="%d/%m/%Y")
    cleaned_df["Time"] = pd.to_datetime(cleaned_df["Time"])
    cleaned_df = cleaned_df.sort_values(by=["Date", "Time"], ascending=[True, True]).reset_index()
    
    return cleaned_df.loc[:, ["Date", "Season", "Team", "Opponent", "Location", "Team Goals", "Opponent Goals", "Team Shots", "Opponent Shots", 
                              "Team SOT", "Opponent SOT", "Team Corners", "Opponent Corners", "PredictedXG"]]



def get_schedule(current_season):    
    max_date = current_season['Date'].max()

    schedule = pd.read_csv('Data/schedule.csv')
    schedule['Date'] = pd.to_datetime(schedule['Date']).dt.normalize()

    schedule = schedule[schedule['Date'] > max_date]

    home_df = schedule.copy()
    away_df = schedule.copy()

    home_df['Location'] = 'Home'
    home_df = home_df.rename(columns={'Home Team': 'Team', 'Away Team': 'Opponent'})
    home_df = home_df.drop(columns=['Match Number', 'Round Number', 'Result'])

    away_df['Location'] = 'Away'
    away_df = away_df.rename(columns={'Home Team': 'Opponent', 'Away Team': 'Team'})
    away_df = away_df.drop(columns=['Match Number', 'Round Number', 'Result'])

    output = pd.concat([home_df, away_df]).sort_values(by='Date', ascending=True).reset_index().drop(columns=['index'])
    output.loc[output['Team'] == 'Man Utd', 'Team'] = 'Man United'
    output.loc[output['Team'] == 'Spurs', 'Team'] = 'Tottenham'
    output.loc[output['Opponent'] == 'Man Utd', 'Opponent'] = 'Man United'
    output.loc[output['Opponent'] == 'Spurs', 'Opponent'] = 'Tottenham'
    return output


def rolling_averages(group, cols, new_cols, games):
    group = group.sort_values('Date')
    rolling_stats = group[cols].rolling(games, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


def add_rolling_features(df, stats_window, betting_window):
    # ---------------------------
    # 1. TEAM ATTACKING FEATURES
    # ---------------------------
    attacking_cols = {
        'Team Goals': 'att_goals',
        'Team Shots': 'att_shots',
        'Team SOT': 'att_sot',
        'Team Corners': 'att_corners'
    }

    for col, new_name in attacking_cols.items():
        df[new_name] = (
            df
            .groupby(['Season', 'Team'])[col]
            .transform(lambda x: x.shift(1).rolling(stats_window).mean())
        )

    # ---------------------------
    # 2. TEAM DEFENSIVE FEATURES
    # (computed as opponent history)
    # ---------------------------
    defensive_cols = {
        'Opponent Goals': 'def_goals_conceded',
        'Opponent Shots': 'def_shots_conceded',
        'Opponent SOT': 'def_sot_conceded',
        'Opponent Corners': 'def_corners_conceded'
    }

    for col, new_name in defensive_cols.items():
        df[new_name] = (
            df
            .groupby(['Season', 'Team'])[col]
            .transform(lambda x: x.shift(1).rolling(stats_window).mean())
        )

    # ---------------------------
    # 3. RENAME DEF FEATURES TO OPPONENT
    # ---------------------------
    def_features = list(defensive_cols.values())

    defensive_df = df[
        ['Season', 'Date', 'Team'] + def_features
    ].rename(columns={'Team': 'Opponent'})

    defensive_df = defensive_df.rename(
        columns={f: f.replace('def_', 'opp_def_') for f in def_features}
    )

    # ---------------------------
    # 4. MERGE OPPONENT DEFENCE BACK
    # ---------------------------
    df = df.merge(
        defensive_df,
        on=['Season', 'Date', 'Opponent'],
        how='left'
    )

    # ---------------------------
    # 5. DROP TEAM DEFENSIVE COLUMNS
    # ---------------------------
    df = df.drop(columns=def_features)

    # ---------------------------
    # 6. ATTACK & DEFENCE STRENGTH
    # (based on historical PredictedXG)
    # ---------------------------
    # League-wide rolling average (for normalization)
    league_avg = df.groupby('Season')['PredictedXG'].transform('mean')

    # Team rolling attack strength
    df['attack_strength'] = (
        df.groupby(['Team'])['PredictedXG']
        .transform(lambda x: x.shift(1).rolling(betting_window).mean()) / league_avg
    )

    # Opponent rolling defence strength
    df['defence_strength'] = (
        df.groupby(['Opponent'])['PredictedXG']
        .transform(lambda x: x.shift(1).rolling(betting_window).mean()) / league_avg
    )

    # ---------------------------
    # 7. REMOVE ALL ROWS WITH NA
    # ---------------------------
    df = df.dropna().reset_index(drop=True)

    return df