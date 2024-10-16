# backtest_module.py

import pandas as pd
import numpy as np
from scipy.stats import poisson

# =======================
# Data Loading and Preparation
# =======================

import pandas as pd

def load_data():
    # Define the file paths for each season using meaningful naming conventions
    season_2024_25_url = 'data/E0.csv'
    season_2023_24_url = 'data/E1.csv'
    season_2022_23_url = 'data/E2.csv'
    season_2021_22_url = 'data/E3.csv'
    season_2020_21_url = 'data/E4.csv'  # Added for E4.csv equivalent

    # Load data for each season
    season_2024_25_data = pd.read_csv(season_2024_25_url)
    season_2023_24_data = pd.read_csv(season_2023_24_url)
    season_2022_23_data = pd.read_csv(season_2022_23_url)
    season_2021_22_data = pd.read_csv(season_2021_22_url)
    season_2020_21_data = pd.read_csv(season_2020_21_url)

    # Combine all seasons' data into a single DataFrame
    # Ensure that all loaded seasons are included in the concatenation
    full_data = pd.concat([
        season_2020_21_data, 
        season_2021_22_data, 
        season_2022_23_data, 
        season_2023_24_data, 
        season_2024_25_data
    ], ignore_index=True)

    # Convert 'Date' to datetime format
    full_data['Date'] = pd.to_datetime(full_data['Date'], format='%d/%m/%Y')

    # Select relevant columns, including betting odds for over/under
    columns_needed = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
        'B365>2.5', 'B365<2.5', 'B365H', 'B365D', 'B365A'
    ]
    data = full_data[columns_needed].copy()

    return data


# =======================
# Team Strength Calculation with "Matches Ago" Scaling
# =======================

def calculate_team_strengths(data, up_to_date, decay_factor=0.99):
    """
    Calculate team strengths up to a specific date with decay applied based on "matches ago".

    :param data: pandas DataFrame containing match data.
    :param up_to_date: datetime object representing the cutoff date.
    :param decay_factor: float, weighting factor for historical data.
    :return: attack_strength_home, defense_strength_home, attack_strength_away, defense_strength_away, league_home_stats, league_away_stats
    """

    # Filter data up to the cutoff date
    historical_data = data[data['Date'] < up_to_date].copy()

    # Sort historical data chronologically
    historical_data = historical_data.sort_values('Date')

    # Initialize 'Matches_Ago' for each team
    historical_data['Matches_Ago_Home'] = historical_data.groupby('HomeTeam').cumcount(ascending=False)
    historical_data['Matches_Ago_Away'] = historical_data.groupby('AwayTeam').cumcount(ascending=False)

    # For home matches
    historical_data['Matches_Ago'] = historical_data['Matches_Ago_Home']
    # Apply decay factor
    historical_data['Weight_Home'] = decay_factor ** historical_data['Matches_Ago_Home']

    # For away matches
    historical_data['Matches_Ago'] = historical_data['Matches_Ago_Away']
    # Apply decay factor
    historical_data['Weight_Away'] = decay_factor ** historical_data['Matches_Ago_Away']

    # Prepare home data with weights
    home_data = pd.DataFrame({
        'Team': historical_data['HomeTeam'],
        'Opponent': historical_data['AwayTeam'],
        'Goals_Scored': historical_data['FTHG'],
        'Goals_Conceded': historical_data['FTAG'],
        'Home_Away': 'Home',
        'Weight': historical_data['Weight_Home']
    })

    # Prepare away data with weights
    away_data = pd.DataFrame({
        'Team': historical_data['AwayTeam'],
        'Opponent': historical_data['HomeTeam'],
        'Goals_Scored': historical_data['FTAG'],
        'Goals_Conceded': historical_data['FTHG'],
        'Home_Away': 'Away',
        'Weight': historical_data['Weight_Away']
    })

    # Combine home and away data
    full_data_prepared = pd.concat([home_data, away_data], ignore_index=True)

    # Filter teams that have played at least 5 matches
    matches_played = full_data_prepared.groupby('Team').size()
    teams_with_enough_matches = matches_played[matches_played >= 5].index

    # Filter the data to include only these teams
    full_data_prepared = full_data_prepared[full_data_prepared['Team'].isin(teams_with_enough_matches)]

    # Calculate weighted team statistics for home games
    home_games = full_data_prepared[full_data_prepared['Home_Away'] == 'Home']
    home_team_stats = home_games.groupby('Team').apply(
        lambda x: pd.Series({
            'Goals_Scored': np.average(x['Goals_Scored'], weights=x['Weight']),
            'Goals_Conceded': np.average(x['Goals_Conceded'], weights=x['Weight'])
        })
    )

    # Calculate weighted team statistics for away games
    away_games = full_data_prepared[full_data_prepared['Home_Away'] == 'Away']
    away_team_stats = away_games.groupby('Team').apply(
        lambda x: pd.Series({
            'Goals_Scored': np.average(x['Goals_Scored'], weights=x['Weight']),
            'Goals_Conceded': np.average(x['Goals_Conceded'], weights=x['Weight'])
        })
    )

    # Calculate league averages
    league_home_stats = home_games[['Goals_Scored', 'Goals_Conceded']].mean()
    league_away_stats = away_games[['Goals_Scored', 'Goals_Conceded']].mean()

    # Calculate attack and defense strengths
    attack_strength_home = home_team_stats['Goals_Scored'] / league_home_stats['Goals_Scored']
    defense_strength_home = home_team_stats['Goals_Conceded'] / league_away_stats['Goals_Scored']
    attack_strength_away = away_team_stats['Goals_Scored'] / league_away_stats['Goals_Scored']
    defense_strength_away = away_team_stats['Goals_Conceded'] / league_home_stats['Goals_Scored']

    return attack_strength_home, defense_strength_home, attack_strength_away, defense_strength_away, league_home_stats, league_away_stats

# =======================
# Backtest Functions
# =======================

def run_backtest_match_result(data, decay_factor=0.99, confidence_threshold=0.4, confidence_multiplier=1.1, max_goals=5, backtest_matchdays=70):
    """
    Runs backtest for match result betting strategy.

    :param data: pandas DataFrame containing match data.
    :param decay_factor: float, weighting factor for historical data.
    :param confidence_threshold: float, minimum predicted probability to place a bet.
    :param confidence_multiplier: float, multiplier for implied probability to determine value bets.
    :param max_goals: int, maximum number of goals to consider in Poisson calculations.
    :param backtest_matchdays: int, number of unique matchdays to include in the backtest.
    :return: Dictionary containing backtest performance metrics and bet data.
    """

    # Sort data by date to ensure chronological order
    data_sorted = data.sort_values('Date')

    # Select the last 'backtest_matchdays' matches for backtesting
    backtest_data = data_sorted.iloc[-backtest_matchdays:].copy()

    # Initialize lists to store predictions
    predictions = []

    # Iterate through each match in the backtest period
    for idx, match in backtest_data.iterrows():
        up_to_date = match['Date']

        # Calculate team strengths up to the match date with "matches ago" scaling
        attack_home, defense_home, attack_away, defense_away, league_home_stats, league_away_stats = calculate_team_strengths(
            data_sorted, up_to_date, decay_factor
        )

        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        actual_result = match['FTR']
        odds_home_win = match['B365H']
        odds_draw = match['B365D']
        odds_away_win = match['B365A']

        # Check if teams have enough data
        if home_team not in attack_home.index or away_team not in attack_away.index:
            continue  # Skip this match

        # Predict goals
        home_lambda = attack_home[home_team] * defense_away[away_team] * league_home_stats['Goals_Scored']
        away_lambda = attack_away[away_team] * defense_home[home_team] * league_away_stats['Goals_Scored']

        # Calculate match outcome probabilities using Poisson
        home_goals_probs = [poisson.pmf(i, home_lambda) for i in range(0, max_goals + 1)]
        away_goals_probs = [poisson.pmf(i, away_lambda) for i in range(0, max_goals + 1)]

        # Calculate probabilities
        home_win_prob = sum(
            home_goals_probs[i] * sum(away_goals_probs[:i]) for i in range(1, max_goals + 1)
        )
        draw_prob = sum(
            home_goals_probs[i] * away_goals_probs[i] for i in range(max_goals + 1)
        )
        away_win_prob = sum(
            away_goals_probs[i] * sum(home_goals_probs[:i]) for i in range(1, max_goals + 1)
        )

        # Normalize probabilities
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob == 0:
            # Avoid division by zero
            home_win_prob, draw_prob, away_win_prob = 0, 0, 0
        else:
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob

        # Calculate implied probabilities from odds
        implied_home_win_prob = 1 / odds_home_win if odds_home_win > 0 else 0
        implied_draw_prob = 1 / odds_draw if odds_draw > 0 else 0
        implied_away_win_prob = 1 / odds_away_win if odds_away_win > 0 else 0

        # Betting decision based on value and confidence thresholds
        bet_result = None
        expected_return_result = 0

        if home_win_prob > implied_home_win_prob * confidence_multiplier and home_win_prob > confidence_threshold:
            bet_result = 'H'
            expected_return_result = odds_home_win * home_win_prob - 1

        elif draw_prob > implied_draw_prob * confidence_multiplier and draw_prob > confidence_threshold:
            bet_result = 'D'
            expected_return_result = odds_draw * draw_prob - 1

        elif away_win_prob > implied_away_win_prob * confidence_multiplier and away_win_prob > confidence_threshold:
            bet_result = 'A'
            expected_return_result = odds_away_win * away_win_prob - 1

        # Calculate profit or loss for match result bet
        profit_result = 0
        if bet_result:
            if bet_result == actual_result:
                if bet_result == 'H':
                    profit_result = odds_home_win - 1
                elif bet_result == 'D':
                    profit_result = odds_draw - 1
                elif bet_result == 'A':
                    profit_result = odds_away_win - 1
            else:
                profit_result = -1  # Lost the bet

        # Append prediction and betting results
        predictions.append({
            'Date': match['Date'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'ActualResult': actual_result,
            'PredictedResult': bet_result,
            'HomeWinProb': home_win_prob,
            'DrawProb': draw_prob,
            'AwayWinProb': away_win_prob,
            'BetResult': bet_result,
            'ProfitResult': profit_result,
            'ExpectedReturnResult': expected_return_result,
        })

    # Convert predictions to DataFrame
    backtest_df = pd.DataFrame(predictions)

    # Filter to bets placed
    bets_placed_df = backtest_df[backtest_df['PredictedResult'].notnull()].copy()

    # Calculate accuracy
    correct_predictions = bets_placed_df[bets_placed_df['ActualResult'] == bets_placed_df['PredictedResult']]
    accuracy = len(correct_predictions) / len(bets_placed_df) if len(bets_placed_df) > 0 else 0

    # Calculate total profit and ROI
    total_profit_result = bets_placed_df['ProfitResult'].sum()
    total_bets_result = len(bets_placed_df)
    roi_result = (total_profit_result / total_bets_result) * 100 if total_bets_result > 0 else 0

    return {
        'profit_result': total_profit_result,
        'roi_result': roi_result,
        'accuracy_result': accuracy,
        'bets_placed_result': total_bets_result,
        'bet_data': bets_placed_df  # Include the DataFrame
    }

def run_backtest_over_under(data, decay_factor=0.99, confidence_threshold_o_u=0.55, confidence_multiplier=1.1, max_goals=5, backtest_matchdays=70):
    """
    Runs backtest for over/under 2.5 goals betting strategy.

    :param data: pandas DataFrame containing match data.
    :param decay_factor: float, weighting factor for historical data.
    :param confidence_threshold_o_u: float, minimum predicted probability to place a bet.
    :param confidence_multiplier: float, multiplier for implied probability to determine value bets.
    :param max_goals: int, maximum number of goals to consider in Poisson calculations.
    :param backtest_matchdays: int, number of unique matchdays to include in the backtest.
    :return: Dictionary containing backtest performance metrics and bet data.
    """

    # Sort data by date to ensure chronological order
    data_sorted = data.sort_values('Date')

    # Select the last 'backtest_matchdays' matches for backtesting
    backtest_data = data_sorted.iloc[-backtest_matchdays:].copy()

    # Initialize lists to store predictions
    predictions = []

    # Iterate through each match in the backtest period
    for idx, match in backtest_data.iterrows():
        up_to_date = match['Date']

        # Calculate team strengths up to the match date with "matches ago" scaling
        attack_home, defense_home, attack_away, defense_away, league_home_stats, league_away_stats = calculate_team_strengths(
            data_sorted, up_to_date, decay_factor
        )

        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        actual_home_goals = match['FTHG']
        actual_away_goals = match['FTAG']
        odds_over = match['B365>2.5']
        odds_under = match['B365<2.5']

        # Check if teams have enough data
        if home_team not in attack_home.index or away_team not in attack_away.index:
            continue  # Skip this match

        # Predict goals
        home_lambda = attack_home[home_team] * defense_away[away_team] * league_home_stats['Goals_Scored']
        away_lambda = attack_away[away_team] * defense_home[home_team] * league_away_stats['Goals_Scored']

        # Calculate total goals lambda
        predicted_total_goals = home_lambda + away_lambda

        # Calculate probabilities for over and under 2.5 goals
        prob_over = 1 - poisson.cdf(2, predicted_total_goals)  # Probability of more than 2.5 goals
        prob_under = poisson.cdf(2, predicted_total_goals)     # Probability of 2.5 goals or less

        # Normalize probabilities (optional but recommended)
        total_prob = prob_over + prob_under
        if total_prob == 0:
            prob_over, prob_under = 0, 0
        else:
            prob_over /= total_prob
            prob_under /= total_prob

        # Calculate implied probabilities from the odds
        implied_prob_over = 1 / odds_over if odds_over > 0 else 0
        implied_prob_under = 1 / odds_under if odds_under > 0 else 0

        # Betting decision based on value and confidence thresholds
        bet_o_u = None
        expected_return_o_u = 0

        if prob_over > implied_prob_over * confidence_multiplier and prob_over > confidence_threshold_o_u:
            bet_o_u = 'Over'
            expected_return_o_u = odds_over * prob_over - 1

        elif prob_under > implied_prob_under * confidence_multiplier and prob_under > confidence_threshold_o_u:
            bet_o_u = 'Under'
            expected_return_o_u = odds_under * prob_under - 1

        # Calculate profit or loss for over/under bet
        profit_o_u = 0
        actual_total_goals = actual_home_goals + actual_away_goals
        actual_o_u = 'Over' if actual_total_goals > 2.5 else 'Under'

        if bet_o_u:
            if bet_o_u == actual_o_u:
                if bet_o_u == 'Over':
                    profit_o_u = odds_over - 1
                elif bet_o_u == 'Under':
                    profit_o_u = odds_under - 1
            else:
                profit_o_u = -1  # Lost the bet

        # Append prediction and betting results
        predictions.append({
            'Date': match['Date'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'BetOverUnder': bet_o_u,
            'ProfitOverUnder': profit_o_u,
            'ExpectedReturnOverUnder': expected_return_o_u,
            'ActualTotalGoals': actual_total_goals,
            'ActualOverUnder': actual_o_u
        })

    # Convert predictions to DataFrame
    backtest_df = pd.DataFrame(predictions)

    # Filter to bets placed
    bets_placed_df = backtest_df[backtest_df['BetOverUnder'].notnull()].copy()

    # Calculate accuracy of over/under predictions
    correct_predictions = bets_placed_df[bets_placed_df['BetOverUnder'] == bets_placed_df['ActualOverUnder']]
    accuracy = len(correct_predictions) / len(bets_placed_df) if len(bets_placed_df) > 0 else 0

    # Calculate total profit and ROI
    total_profit_o_u = bets_placed_df['ProfitOverUnder'].sum()
    total_bets_o_u = len(bets_placed_df)
    roi_o_u = (total_profit_o_u / total_bets_o_u) * 100 if total_bets_o_u > 0 else 0

    return {
        'profit_over_under': total_profit_o_u,
        'roi_over_under': roi_o_u,
        'accuracy_over_under': accuracy,
        'bets_placed_over_under': total_bets_o_u,
        'bet_data': bets_placed_df  # Include the DataFrame
    }
    
def run_backtest(data, decay_factor=0.99, confidence_threshold=0.4, confidence_threshold_o_u=0.55,
                confidence_multiplier=1.1, max_goals=5, backtest_matchdays=40):
    """
    Runs a backtest of the betting strategy using the provided parameters.

    Parameters:
    - data: pandas DataFrame containing the match data.
    - decay_factor: float, weighting factor for historical data.
    - confidence_threshold: float, minimum predicted probability to place a bet on match results.
    - confidence_threshold_o_u: float, minimum predicted probability for over/under bets.
    - confidence_multiplier: float, multiplier for the implied probability to determine value bets.
    - max_goals: int, maximum number of goals considered in Poisson distribution calculations.
    - backtest_matchdays: int, number of unique matchdays to include in the backtest.

    Returns:
    - Dictionary containing performance metrics of the backtest.
    """

    # Run match result backtest
    match_result_performance = run_backtest_match_result(
        data=data,
        decay_factor=decay_factor,
        confidence_threshold=confidence_threshold,
        confidence_multiplier=confidence_multiplier,
        max_goals=max_goals,
        backtest_matchdays=backtest_matchdays
    )

    # Run over/under backtest
    over_under_performance = run_backtest_over_under(
        data=data,
        decay_factor=decay_factor,
        confidence_threshold_o_u=confidence_threshold_o_u,
        confidence_multiplier=confidence_multiplier,
        max_goals=max_goals,
        backtest_matchdays=backtest_matchdays
    )

    # Combine performance metrics
    performance = {
        'profit_result': match_result_performance['profit_result'],
        'roi_result': match_result_performance['roi_result'],
        'accuracy_result': match_result_performance['accuracy_result'],
        'bets_placed_result': match_result_performance['bets_placed_result'],
        'profit_over_under': over_under_performance['profit_over_under'],
        'roi_over_under': over_under_performance['roi_over_under'],
        'accuracy_over_under': over_under_performance['accuracy_over_under'],
        'bets_placed_over_under': over_under_performance['bets_placed_over_under'],
        'backtest_df_result': match_result_performance['backtest_df'],
        'backtest_df_over_under': over_under_performance['backtest_df']
    }

    return performance

def run_backtest_match_result_all_bets(data, decay_factor=0.999, max_goals=5, backtest_matchdays=70):
    """
    Runs a backtest for match result betting strategy by placing a bet on every match.
    
    :param data: pandas DataFrame containing match data.
    :param decay_factor: float, weighting factor for historical data.
    :param max_goals: int, maximum number of goals to consider in Poisson calculations.
    :param backtest_matchdays: int, number of unique matchdays to include in the backtest.
    :return: Dictionary containing backtest performance metrics and bet data.
    """
    
    # Sort data by date to ensure chronological order
    data_sorted = data.sort_values('Date')
    
    # Select the last 'backtest_matchdays' matches for backtesting
    backtest_data = data_sorted.iloc[-backtest_matchdays:].copy()
    
    # Initialize lists to store predictions
    predictions = []
    
    # Iterate through each match in the backtest period
    for idx, match in backtest_data.iterrows():
        up_to_date = match['Date']
        
        # Calculate team strengths up to the match date with "matches ago" scaling
        attack_home, defense_home, attack_away, defense_away, league_home_stats, league_away_stats = calculate_team_strengths(
            data_sorted, up_to_date, decay_factor
        )
        
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        actual_result = match['FTR']
        odds_home_win = match['B365H']
        odds_draw = match['B365D']
        odds_away_win = match['B365A']
        
        # Check if teams have enough data
        if home_team not in attack_home.index or away_team not in attack_away.index:
            continue  # Skip this match
        
        # Predict goals
        home_lambda = attack_home[home_team] * defense_away[away_team] * league_home_stats['Goals_Scored']
        away_lambda = attack_away[away_team] * defense_home[home_team] * league_away_stats['Goals_Scored']
        
        # Calculate match outcome probabilities using Poisson
        home_goals_probs = [poisson.pmf(i, home_lambda) for i in range(0, max_goals + 1)]
        away_goals_probs = [poisson.pmf(i, away_lambda) for i in range(0, max_goals + 1)]
        
        # Calculate probabilities
        home_win_prob = sum(
            home_goals_probs[i] * sum(away_goals_probs[:i]) for i in range(1, max_goals + 1)
        )
        draw_prob = sum(
            home_goals_probs[i] * away_goals_probs[i] for i in range(max_goals + 1)
        )
        away_win_prob = sum(
            away_goals_probs[i] * sum(home_goals_probs[:i]) for i in range(1, max_goals + 1)
        )
        
        # Normalize probabilities
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob == 0:
            # Avoid division by zero
            home_win_prob, draw_prob, away_win_prob = 0, 0, 0
        else:
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
        
        # Determine the most probable outcome
        predicted_result = 'H' if home_win_prob >= draw_prob and home_win_prob >= away_win_prob else (
            'A' if away_win_prob >= home_win_prob and away_win_prob >= draw_prob else 'D'
        )
        
        # Calculate profit or loss for the bet
        profit_result = 0
        if predicted_result == actual_result:
            if predicted_result == 'H':
                profit_result = odds_home_win - 1  # Net profit
            elif predicted_result == 'D':
                profit_result = odds_draw - 1
            elif predicted_result == 'A':
                profit_result = odds_away_win - 1
        else:
            profit_result = -1  # Lost the bet
        
        # Append prediction and betting results
        predictions.append({
            'Date': match['Date'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'ActualResult': actual_result,
            'PredictedResult': predicted_result,
            'HomeWinProb': home_win_prob,
            'DrawProb': draw_prob,
            'AwayWinProb': away_win_prob,
            'BetResult': predicted_result,
            'ProfitResult': profit_result,
            'Odds': odds_home_win if predicted_result == 'H' else (odds_draw if predicted_result == 'D' else odds_away_win)
        })
    
    # Convert predictions to DataFrame
    backtest_df = pd.DataFrame(predictions)
    
    # Calculate accuracy
    correct_predictions = backtest_df[backtest_df['ActualResult'] == backtest_df['PredictedResult']]
    accuracy = len(correct_predictions) / len(backtest_df) if len(backtest_df) > 0 else 0
    
    # Calculate total profit and ROI
    total_profit = backtest_df['ProfitResult'].sum()
    total_bets = len(backtest_df)
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
    
    return {
        'profit_result': total_profit,
        'roi_result': roi,
        'accuracy_result': accuracy,
        'bets_placed_result': total_bets,
        'bet_data': backtest_df  # Include the DataFrame
    }

def run_backtest_over_under_all_bets(data, decay_factor=0.999, max_goals=5, backtest_matchdays=70):
    """
    Runs a backtest for over/under 2.5 goals betting strategy by placing a bet on every match.
    
    :param data: pandas DataFrame containing match data.
    :param decay_factor: float, weighting factor for historical data.
    :param max_goals: int, maximum number of goals to consider in Poisson calculations.
    :param backtest_matchdays: int, number of unique matchdays to include in the backtest.
    :return: Dictionary containing backtest performance metrics and bet data.
    """
    
    # Sort data by date to ensure chronological order
    data_sorted = data.sort_values('Date')
    
    # Select the last 'backtest_matchdays' matches for backtesting
    backtest_data = data_sorted.iloc[-backtest_matchdays:].copy()
    
    # Initialize lists to store predictions
    predictions = []
    
    # Iterate through each match in the backtest period
    for idx, match in backtest_data.iterrows():
        up_to_date = match['Date']
        
        # Calculate team strengths up to the match date with "matches ago" scaling
        attack_home, defense_home, attack_away, defense_away, league_home_stats, league_away_stats = calculate_team_strengths(
            data_sorted, up_to_date, decay_factor
        )
        
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        actual_home_goals = match['FTHG']
        actual_away_goals = match['FTAG']
        odds_over = match['B365>2.5']
        odds_under = match['B365<2.5']
        
        # Check if teams have enough data
        if home_team not in attack_home.index or away_team not in attack_away.index:
            continue  # Skip this match
        
        # Predict goals
        home_lambda = attack_home[home_team] * defense_away[away_team] * league_home_stats['Goals_Scored']
        away_lambda = attack_away[away_team] * defense_home[home_team] * league_away_stats['Goals_Scored']
        
        # Calculate total goals lambda
        predicted_total_goals = home_lambda + away_lambda
        
        # Calculate probabilities for over and under 2.5 goals using Poisson
        prob_over = 1 - poisson.cdf(2, predicted_total_goals)  # Probability of more than 2.5 goals
        prob_under = poisson.cdf(2, predicted_total_goals)     # Probability of 2.5 goals or less
        
        # Normalize probabilities
        total_prob = prob_over + prob_under
        if total_prob == 0:
            prob_over, prob_under = 0, 0
        else:
            prob_over /= total_prob
            prob_under /= total_prob
        
        # Determine the most probable outcome
        predicted_o_u = 'Over' if prob_over >= prob_under else 'Under'
        
        # Calculate profit or loss for the bet
        profit_o_u = 0
        actual_total_goals = actual_home_goals + actual_away_goals
        actual_o_u = 'Over' if actual_total_goals > 2.5 else 'Under'
        
        # Assign odds based on predicted outcome
        if predicted_o_u == 'Over':
            odds = odds_over
        else:
            odds = odds_under
        
        # Calculate profit or loss
        if predicted_o_u == actual_o_u:
            profit_o_u = odds - 1  # Net profit (assuming $1 stake)
        else:
            profit_o_u = -1  # Lost the bet
        
        # Append prediction and betting results
        predictions.append({
            'Date': match['Date'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'BetOverUnder': predicted_o_u,
            'ActualTotalGoals': actual_total_goals,
            'ActualOverUnder': actual_o_u,
            'PredictedOverUnder': predicted_o_u,
            'ProfitOverUnder': profit_o_u,
            'Odds': odds
        })
    
    # Convert predictions to DataFrame
    backtest_df = pd.DataFrame(predictions)
    
    # Calculate accuracy of over/under predictions
    correct_predictions = backtest_df[backtest_df['ActualOverUnder'] == backtest_df['PredictedOverUnder']]
    accuracy = len(correct_predictions) / len(backtest_df) if len(backtest_df) > 0 else 0
    
    # Calculate total profit and ROI
    total_profit = backtest_df['ProfitOverUnder'].sum()
    total_bets = len(backtest_df)
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
    
    return {
        'profit_over_under': total_profit,
        'roi_over_under': roi,
        'accuracy_over_under': accuracy,
        'bets_placed_over_under': total_bets,
        'bet_data': backtest_df  # Include the DataFrame for detailed analysis
    }

