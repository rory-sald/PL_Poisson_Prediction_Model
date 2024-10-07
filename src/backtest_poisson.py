import pandas as pd
import numpy as np
from scipy.stats import poisson

def predict_goals(home_team, away_team):
    try:
        home_attack = attack_strength_home[home_team]
        away_defense = defense_strength_away[away_team]
        home_lambda = home_attack * away_defense * league_home_stats['Goals_Scored']
    except KeyError:
        home_lambda = league_home_stats['Goals_Scored']

    try:
        away_attack = attack_strength_away[away_team]
        home_defense = defense_strength_home[home_team]
        away_lambda = away_attack * home_defense * league_away_stats['Goals_Scored']
    except KeyError:
        away_lambda = league_away_stats['Goals_Scored']

    return home_lambda, away_lambda

# Step 1: Load Historical Data
# Load current season data
current_season_url = '/Users/rems/Documents/Python/Scraper/E0.csv'
current_season_data = pd.read_csv(current_season_url)

# Load last season data
last_season_url = '/Users/rems/Documents/Python/Scraper/E1.csv'
last_season_data = pd.read_csv(last_season_url)

# Combine both seasons' data into a single DataFrame
data = pd.concat([current_season_data, last_season_data], ignore_index=True)

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Select relevant columns
columns_needed = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
    'B365H', 'B365D', 'B365A'  # Include betting odds columns
]
data = data[columns_needed]

# Calculate 'Days_Ago' and 'Weight'
data['Days_Ago'] = (pd.to_datetime('today') - data['Date']).dt.days
decay_factor = 0.98
data['Weight'] = decay_factor ** data['Days_Ago']

# Prepare the data in the required format
# ... (Same as your existing code, including weights)

# Create home and away data with weights
home_data = pd.DataFrame()
home_data['Team'] = data['HomeTeam']
home_data['Opponent'] = data['AwayTeam']
home_data['Goals_Scored'] = data['FTHG']
home_data['Goals_Conceded'] = data['FTAG']
home_data['Home_Away'] = 'Home'
home_data['Weight'] = data['Weight']

away_data = pd.DataFrame()
away_data['Team'] = data['AwayTeam']
away_data['Opponent'] = data['HomeTeam']
away_data['Goals_Scored'] = data['FTAG']
away_data['Goals_Conceded'] = data['FTHG']
away_data['Home_Away'] = 'Away'
away_data['Weight'] = data['Weight']

# Combine home and away data
full_data = pd.concat([home_data, away_data], ignore_index=True)

# Step 2: Calculate weighted averages
# Calculate weighted team statistics
home_team_stats = home_data.groupby('Team').apply(
    lambda x: pd.Series({
        'Goals_Scored': np.average(x['Goals_Scored'], weights=x['Weight']),
        'Goals_Conceded': np.average(x['Goals_Conceded'], weights=x['Weight'])
    })
)

away_team_stats = away_data.groupby('Team').apply(
    lambda x: pd.Series({
        'Goals_Scored': np.average(x['Goals_Scored'], weights=x['Weight']),
        'Goals_Conceded': np.average(x['Goals_Conceded'], weights=x['Weight'])
    })
)

# Calculate league averages
league_home_stats = home_data[['Goals_Scored', 'Goals_Conceded']].mean()
league_away_stats = away_data[['Goals_Scored', 'Goals_Conceded']].mean()

# Step 3: Calculate attack and defense strengths
attack_strength_home = home_team_stats['Goals_Scored'] / league_home_stats['Goals_Scored']
defense_strength_home = home_team_stats['Goals_Conceded'] / league_away_stats['Goals_Scored']
attack_strength_away = away_team_stats['Goals_Scored'] / league_away_stats['Goals_Scored']
defense_strength_away = away_team_stats['Goals_Conceded'] / league_home_stats['Goals_Scored']

# Step 4: Prepare Historical Fixtures for Backtesting
# We'll use matches from the last N rounds for backtesting
# For example, backtest on the last 5 matchdays
backtest_matchdays = data['Date'].unique()[-5:]  # Adjust as needed

backtest_data = data[data['Date'].isin(backtest_matchdays)]

# Step 5: Perform Backtesting
predictions = []

for index, row in backtest_data.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    actual_result = row['FTR']  # Actual full-time result ('H', 'D', 'A')
    actual_home_goals = row['FTHG']
    actual_away_goals = row['FTAG']
    odds_home_win = row['B365H']
    odds_draw = row['B365D']
    odds_away_win = row['B365A']

    # Check if teams are in the dataset
    if home_team not in attack_strength_home.index or away_team not in attack_strength_away.index:
        continue

    # Predict goals
    home_lambda, away_lambda = predict_goals(home_team, away_team)

    # Simulate match probabilities
    max_goals = 5  # Adjust as needed
    home_goals_probs = [poisson.pmf(i, home_lambda) for i in range(0, max_goals + 1)]
    away_goals_probs = [poisson.pmf(i, away_lambda) for i in range(0, max_goals + 1)]

    # Calculate probabilities
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0

    for i in range(len(home_goals_probs)):
        for j in range(len(away_goals_probs)):
            prob = home_goals_probs[i] * away_goals_probs[j]
            if i > j:
                home_win_prob += prob
            elif i == j:
                draw_prob += prob
            else:
                away_win_prob += prob

    # Determine predicted outcome
    predicted_outcome = 'H' if home_win_prob > max(draw_prob, away_win_prob) else 'D' if draw_prob > away_win_prob else 'A'

    # Simulate betting strategy
    # Example: Bet on the outcome with the highest predicted probability if it exceeds a threshold
    threshold = 0.4  # Adjust threshold as needed
    bet = None
    if home_win_prob > threshold and home_win_prob > max(draw_prob, away_win_prob):
        bet = 'H'
        expected_return = odds_home_win * home_win_prob - 1
    elif draw_prob > threshold and draw_prob > max(home_win_prob, away_win_prob):
        bet = 'D'
        expected_return = odds_draw * draw_prob - 1
    elif away_win_prob > threshold:
        bet = 'A'
        expected_return = odds_away_win * away_win_prob - 1

    # Calculate profit or loss
    profit = 0
    if bet:
        if bet == actual_result:
            if bet == 'H':
                profit = odds_home_win - 1  # Assuming a bet of 1 unit
            elif bet == 'D':
                profit = odds_draw - 1
            elif bet == 'A':
                profit = odds_away_win - 1
        else:
            profit = -1  # Lost the bet

    predictions.append({
        'Date': row['Date'],
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'ActualResult': actual_result,
        'PredictedResult': predicted_outcome,
        'HomeWinProb': home_win_prob,
        'DrawProb': draw_prob,
        'AwayWinProb': away_win_prob,
        'ActualHomeGoals': actual_home_goals,
        'ActualAwayGoals': actual_away_goals,
        'PredictedHomeGoals': home_lambda,
        'PredictedAwayGoals': away_lambda,
        'Bet': bet,
        'Profit': profit
    })

# Step 6: Analyze Backtesting Results
backtest_df = pd.DataFrame(predictions)

# Calculate accuracy of match outcome predictions
correct_predictions = backtest_df[backtest_df['ActualResult'] == backtest_df['PredictedResult']]
accuracy = len(correct_predictions) / len(backtest_df)

print(f"Prediction Accuracy: {accuracy:.2%}")

# Calculate Mean Absolute Error for goals
mae_home_goals = np.mean(np.abs(backtest_df['ActualHomeGoals'] - backtest_df['PredictedHomeGoals']))
mae_away_goals = np.mean(np.abs(backtest_df['ActualAwayGoals'] - backtest_df['PredictedAwayGoals']))

print(f"Mean Absolute Error for Home Goals: {mae_home_goals:.2f}")
print(f"Mean Absolute Error for Away Goals: {mae_away_goals:.2f}")

# Calculate total profit from betting strategy
total_profit = backtest_df['Profit'].sum()
total_bets = backtest_df['Bet'].notnull().sum()
roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0

print(f"Total Profit: {total_profit:.2f} units")
print(f"Number of Bets Placed: {total_bets}")
print(f"Return on Investment (ROI): {roi:.2f}%")
