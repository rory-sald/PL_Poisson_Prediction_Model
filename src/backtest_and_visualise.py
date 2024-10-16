import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better aesthetics
sns.set(style='darkgrid')

# Step 1: Load Historical Data
# Load current season data
current_season_url = 'data/E0.csv'
current_season_data = pd.read_csv(current_season_url)

# Load last season data
last_season_url = 'data/E1.csv'
last_season_data = pd.read_csv(last_season_url)

last_last_season_url = 'data/E2.csv'
last_last_season_data = pd.read_csv(last_last_season_url)

# Combine last season's data first, then current season's data
data = pd.concat([last_last_season_data, last_season_data, current_season_data], ignore_index=True)

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Select relevant columns, including betting odds for over/under
columns_needed = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
    'B365>2.5', 'B365<2.5', 'B365H', 'B365D', 'B365A'  # Include betting odds columns
]
data = data[columns_needed]

# Calculate 'Days_Ago' and 'Weight' for weighting historical data
data['Days_Ago'] = (pd.to_datetime('today') - data['Date']).dt.days
decay_factor = 0.99
data['Weight'] = decay_factor ** data['Days_Ago']

# Prepare the data in the required format
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
# Use matches from the last N matchdays for backtesting
# For example, backtest on the last 5 matchdays
backtest_matchdays = data['Date'].unique()[-15:]  # Adjust as needed
backtest_data = data[data['Date'].isin(backtest_matchdays)]

# Step 5: Perform Backtesting and Simulate Bets
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
    odds_over = row['B365>2.5']
    odds_under = row['B365<2.5']

    # Check if teams are in the dataset
    if home_team not in attack_strength_home.index or away_team not in attack_strength_away.index:
        continue

    # Predict goals
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

    home_lambda, away_lambda = predict_goals(home_team, away_team)

    # Simulate match probabilities using Poisson distribution
    max_goals = 5  # Adjust as needed
    home_goals_probs = [poisson.pmf(i, home_lambda) for i in range(0, max_goals + 1)]
    away_goals_probs = [poisson.pmf(i, away_lambda) for i in range(0, max_goals + 1)]

    # Calculate match outcome probabilities
    home_win_prob = sum(home_goals_probs[i] * sum(away_goals_probs[:i]) for i in range(1, max_goals + 1))
    draw_prob = sum(home_goals_probs[i] * away_goals_probs[i] for i in range(max_goals + 1))
    away_win_prob = sum(away_goals_probs[i] * sum(home_goals_probs[:i]) for i in range(1, max_goals + 1))

    # Calculate implied probabilities from the odds
    implied_home_win_prob = 1 / odds_home_win
    implied_draw_prob = 1 / odds_draw
    implied_away_win_prob = 1 / odds_away_win

    # Initialize variables for match result betting decision
    bet_result = None
    expected_return_result = 0

    # Confidence threshold for placing a bet
    confidence_threshold = 0.4

    # Decision-making based on value and confidence thresholds for match result
    if home_win_prob > implied_home_win_prob and home_win_prob > confidence_threshold:
        bet_result = 'H'
        expected_return_result = odds_home_win * home_win_prob - 1

    elif draw_prob > implied_draw_prob and draw_prob > confidence_threshold:
        bet_result = 'D'
        expected_return_result = odds_draw * draw_prob - 1

    elif away_win_prob > implied_away_win_prob and away_win_prob > confidence_threshold:
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
            profit_result = -1

    # Over/Under Betting Strategy
    # Calculate the predicted probability of over and under 2.5 goals
    predicted_total_goals = home_lambda + away_lambda
    prob_over = 1 - poisson.cdf(2, predicted_total_goals)  # Probability of more than 2.5 goals
    prob_under = poisson.cdf(2, predicted_total_goals)     # Probability of 2.5 goals or less

    # Calculate implied probabilities from the over/under odds
    implied_prob_over = 1 / odds_over
    implied_prob_under = 1 / odds_under

    # Initialize variables for over/under betting decision
    bet_o_u = None
    expected_return_o_u = 0
    confidence_threshold_o_u = 0.55  # Base confidence threshold for placing a bet
    confidence_multiplier = 1.1  # Multiplier indicating how much greater the predicted prob should be

    # Decision-making based on value and confidence thresholds for over/under market
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
            profit_o_u = -1

    # Append prediction and betting results
    predictions.append({
        'Date': row['Date'],
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'ActualHomeGoals': actual_home_goals,
        'ActualAwayGoals': actual_away_goals,
        'ActualResult': actual_result,
        'PredictedResult': bet_result,
        'HomeWinProb': home_win_prob,
        'DrawProb': draw_prob,
        'AwayWinProb': away_win_prob,
        'BetResult': bet_result,
        'ProfitResult': profit_result,
        'ExpectedReturnResult': expected_return_result,
        'BetOverUnder': bet_o_u,
        'ProfitOverUnder': profit_o_u,
        'ExpectedReturnOverUnder': expected_return_o_u,
        'B365H': odds_home_win,
        'B365D': odds_draw,
        'B365A': odds_away_win,
        'B365>2.5': odds_over,
        'B365<2.5': odds_under
    })

# Step 6: Analyze Backtesting Results
backtest_df = pd.DataFrame(predictions)

# Calculate accuracy of match outcome predictions
correct_predictions = backtest_df[backtest_df['ActualResult'] == backtest_df['PredictedResult']]
accuracy = len(correct_predictions) / len(backtest_df)
print(f"Prediction Accuracy (Match Result): {accuracy:.2%}")

# Calculate accuracy of over/under predictions
correct_o_u_predictions = backtest_df[
    ((backtest_df['BetOverUnder'] == 'Over') & (backtest_df['ActualHomeGoals'] + backtest_df['ActualAwayGoals'] > 2.5)) |
    ((backtest_df['BetOverUnder'] == 'Under') & (backtest_df['ActualHomeGoals'] + backtest_df['ActualAwayGoals'] <= 2.5))
]
o_u_accuracy = len(correct_o_u_predictions) / len(backtest_df[backtest_df['BetOverUnder'].notnull()])
print(f"Prediction Accuracy (Over/Under): {o_u_accuracy:.2%}")

# Calculate total profit from match result betting strategy
total_profit_result = backtest_df['ProfitResult'].sum()
total_bets_result = backtest_df['BetResult'].notnull().sum()
roi_result = (total_profit_result / total_bets_result) * 100 if total_bets_result > 0 else 0

print(f"\nMatch Result Betting Strategy:")
print(f"Total Profit: {total_profit_result:.2f} units")
print(f"Number of Bets Placed: {total_bets_result}")
print(f"Return on Investment (ROI): {roi_result:.2f}%")

# Calculate total profit from over/under betting strategy
total_profit_o_u = backtest_df['ProfitOverUnder'].sum()
total_bets_o_u = backtest_df['BetOverUnder'].notnull().sum()
roi_o_u = (total_profit_o_u / total_bets_o_u) * 100 if total_bets_o_u > 0 else 0

print(f"\nOver/Under 2.5 Goals Betting Strategy:")
print(f"Total Profit: {total_profit_o_u:.2f} units")
print(f"Number of Bets Placed: {total_bets_o_u}")
print(f"Return on Investment (ROI): {roi_o_u:.2f}%")

# Step 7: Visualize Bets and Profits

# Cumulative Profit Over Time for Match Result Bets
backtest_df['CumulativeProfitResult'] = backtest_df['ProfitResult'].cumsum()

plt.figure(figsize=(12, 6))
plt.plot(backtest_df['Date'], backtest_df['CumulativeProfitResult'], marker='o', label='Match Result Bets')
plt.title('Cumulative Profit Over Time (Match Result Bets)')
plt.xlabel('Date')
plt.ylabel('Cumulative Profit (units)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cumulative Profit Over Time for Over/Under Bets
backtest_df['CumulativeProfitOverUnder'] = backtest_df['ProfitOverUnder'].cumsum()

plt.figure(figsize=(12, 6))
plt.plot(backtest_df['Date'], backtest_df['CumulativeProfitOverUnder'], marker='o', color='green', label='Over/Under Bets')
plt.title('Cumulative Profit Over Time (Over/Under Bets)')
plt.xlabel('Date')
plt.ylabel('Cumulative Profit (units)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

