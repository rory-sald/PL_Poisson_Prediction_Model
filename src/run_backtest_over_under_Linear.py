import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from backtest_module import load_data

from tqdm import tqdm

def run_backtest_over_under_ml(data, backtest_matchdays=70, retrain_interval=10):
    """
    Runs a backtest for Over/Under 2.5 goals prediction using a machine learning model,
    ensuring that only data available up to each match date is used for training.
    """
    # Ensure data is sorted by date
    data = data.sort_values('Date').reset_index(drop=True)

    # Initialize lists to store results
    predictions = []
    actuals = []
    dates = []
    bet_data_list = []

    # Define the backtest indices
    start_index = len(data) - backtest_matchdays

    # Initialize model to None
    model = None

    for idx in tqdm(range(start_index, len(data)), desc="Processing matches", unit="match"):
        match = data.iloc[idx]
        match_date = match['Date']
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']

        # Retrain the model at specified intervals or if model is None
        if idx % retrain_interval == 0 or model is None:
            # Use data up to the match date
            train_data = data[data['Date'] < match_date].copy()
            X_train, y_train = prepare_features_and_labels(train_data)

            # Check if there is sufficient data to train
            if len(X_train) < 50:
                print(f"Skipping match on {match_date.date()} between {home_team} and {away_team} due to insufficient training data ({len(X_train)} samples).")
                continue  # Skip if not enough data

            # Train the model
            model = train_model(X_train, y_train)
            print(f"Model trained with {len(X_train)} samples up to {match_date.date()}.")

        # Prepare test data using the corrected function
        X_test, y_test = prepare_match_features(match, data)

        if X_test.empty:
            print(f"Skipping match on {match_date.date()} between {home_team} and {away_team} due to insufficient feature data.")
            continue  # Skip if no test data

        # Make prediction
        prob_over = model.predict_proba(X_test)[:, 1]
        print(f"Predicted probability for Over: {prob_over[0]:.2f}")

        # Determine betting decision for the current match
        bet_over_under = 'Over' if prob_over[0] >= 0.5 else 'Under'

        # Store predictions
        predictions.append(prob_over[0])
        actuals.append(y_test.iloc[0])
        dates.append(match_date)
        bet_data = {
            'Date': match_date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'ActualOverUnder': 'Over' if y_test.iloc[0] == 1 else 'Under',
            'PredictedProbOver': prob_over[0],
            'BetOverUnder': bet_over_under,  # Added here
            'B365>2.5': match['B365>2.5'],
            'B365<2.5': match['B365<2.5'],
            'TotalGoals': match['FTHG'] + match['FTAG']
        }
        bet_data_list.append(bet_data)
        print(f"Placed bet on {bet_over_under} for match on {match_date.date()} with predicted probability {prob_over[0]:.2f}.")

    # After predictions are made, evaluate performance
    bet_data_df = pd.DataFrame(bet_data_list)

    if bet_data_df.empty:
        print("No bets were placed during the backtest period.")
        return {
            'Total Profit': 0.0,
            'ROI (%)': 0.0,
            'Accuracy': 0.0,
            'Total Bets': 0,
            'bet_data': bet_data_df
        }

    # Determine profit or loss
    bet_data_df['Profit'] = bet_data_df.apply(calculate_profit, axis=1)

    # Calculate performance metrics
    total_profit = bet_data_df['Profit'].sum()
    total_bets = len(bet_data_df)
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0.0
    accuracy = (bet_data_df['BetOverUnder'] == bet_data_df['ActualOverUnder']).mean()

    print(f"Total Profit: ${total_profit:.2f}")
    print(f"ROI: {roi:.2f}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total Bets Placed: {total_bets}")

    return {
        'Total Profit': total_profit,
        'ROI (%)': roi,
        'Accuracy': accuracy,
        'Total Bets': total_bets,
        'bet_data': bet_data_df
    }

def prepare_match_features(match, data):
    """
    Prepares features for a single match using historical data up to the match date.

    :param match: Series representing the match.
    :param data: DataFrame containing historical match data.
    :return: X (DataFrame of features), y (Series of label)
    """
    match_date = match['Date']
    home_team = match['HomeTeam']
    away_team = match['AwayTeam']

    # Get data up to the match date
    past_data = data[data['Date'] < match_date]

    # Check if teams have enough data
    home_team_data = past_data[(past_data['HomeTeam'] == home_team) | (past_data['AwayTeam'] == home_team)]
    away_team_data = past_data[(past_data['HomeTeam'] == away_team) | (past_data['AwayTeam'] == away_team)]

    if len(home_team_data) < 5 or len(away_team_data) < 5:
        print(f"Skipping match on {match_date.date()} between {home_team} and {away_team} due to insufficient team data.")
        return pd.DataFrame(), pd.Series()

    # Calculate features for home team
    home_team_goals_scored = home_team_data.apply(
        lambda x: x['FTHG'] if x['HomeTeam'] == home_team else x['FTAG'], axis=1)
    home_team_goals_conceded = home_team_data.apply(
        lambda x: x['FTAG'] if x['HomeTeam'] == home_team else x['FTHG'], axis=1)
    home_team_avg_goals_scored = home_team_goals_scored.mean()
    home_team_avg_goals_conceded = home_team_goals_conceded.mean()

    # Calculate features for away team
    away_team_goals_scored = away_team_data.apply(
        lambda x: x['FTHG'] if x['HomeTeam'] == away_team else x['FTAG'], axis=1)
    away_team_goals_conceded = away_team_data.apply(
        lambda x: x['FTAG'] if x['HomeTeam'] == away_team else x['FTHG'], axis=1)
    away_team_avg_goals_scored = away_team_goals_scored.mean()
    away_team_avg_goals_conceded = away_team_goals_conceded.mean()

    # Recent form - last 5 matches
    home_team_recent_data = home_team_data.tail(5)
    away_team_recent_data = away_team_data.tail(5)

    home_team_recent_goals_scored = home_team_recent_data.apply(
        lambda x: x['FTHG'] if x['HomeTeam'] == home_team else x['FTAG'], axis=1).mean()
    home_team_recent_goals_conceded = home_team_recent_data.apply(
        lambda x: x['FTAG'] if x['HomeTeam'] == home_team else x['FTHG'], axis=1).mean()

    away_team_recent_goals_scored = away_team_recent_data.apply(
        lambda x: x['FTHG'] if x['HomeTeam'] == away_team else x['FTAG'], axis=1).mean()
    away_team_recent_goals_conceded = away_team_recent_data.apply(
        lambda x: x['FTAG'] if x['HomeTeam'] == away_team else x['FTHG'], axis=1).mean()

    # Head-to-head statistics
    h2h_data = past_data[((past_data['HomeTeam'] == home_team) & (past_data['AwayTeam'] == away_team)) |
                         ((past_data['HomeTeam'] == away_team) & (past_data['AwayTeam'] == home_team))]

    h2h_total_matches = len(h2h_data)
    if h2h_total_matches > 0:
        h2h_total_goals = h2h_data['FTHG'] + h2h_data['FTAG']
        h2h_avg_goals = h2h_total_goals.mean()
    else:
        h2h_avg_goals = np.nan  # We'll handle missing values later

    # Create feature dictionary
    feature_dict = {
        'HomeTeamAvgGoalsScored': home_team_avg_goals_scored,
        'HomeTeamAvgGoalsConceded': home_team_avg_goals_conceded,
        'AwayTeamAvgGoalsScored': away_team_avg_goals_scored,
        'AwayTeamAvgGoalsConceded': away_team_avg_goals_conceded,
        'HomeTeamRecentAvgGoalsScored': home_team_recent_goals_scored,
        'HomeTeamRecentAvgGoalsConceded': home_team_recent_goals_conceded,
        'AwayTeamRecentAvgGoalsScored': away_team_recent_goals_scored,
        'AwayTeamRecentAvgGoalsConceded': away_team_recent_goals_conceded,
        'H2HAvgGoals': h2h_avg_goals,
        'Date': match_date,
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'TotalGoals': match['FTHG'] + match['FTAG']
    }

    # Create DataFrame for features
    features_df = pd.DataFrame([feature_dict])

    # Handle missing values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())

    # Define features (X) and label (y)
    X = features_df.drop(columns=['Date', 'HomeTeam', 'AwayTeam', 'TotalGoals'])
    y = features_df['TotalGoals'].apply(lambda x: 1 if x > 2.5 else 0)

    return X, y

def train_model(X_train, y_train):
    """
    Trains a Random Forest Classifier.

    :param X_train: DataFrame of training features.
    :param y_train: Series of training labels.
    :return: Trained model.
    """
    # Instantiate the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model

def calculate_profit(row):
    """
    Calculates the profit or loss for a bet.

    :param row: Row of DataFrame containing bet data.
    :return: Profit or loss amount.
    """
    bet = row['BetOverUnder']
    actual = row['ActualOverUnder']
    stake = 1  # Assume $1 stake per bet

    if bet == 'Over':
        odds = row['B365>2.5']
    else:
        odds = row['B365<2.5']

    # Check for missing or invalid odds
    if pd.isna(odds) or odds <= 1:
        return 0  # Skip this bet or handle as no profit/loss

    if bet == actual:
        profit = (odds - 1) * stake  # Net profit
    else:
        profit = -stake  # Lost the bet

    return profit

# Load your data
data = load_data()

# Run the backtest
backtest_results = run_backtest_over_under_ml(data, backtest_matchdays=70)

# Display results
print("Backtest Performance:")
print(f"Total Profit: ${backtest_results['Total Profit']:.2f}")
print(f"ROI: {backtest_results['ROI (%)']:.2f}%")
print(f"Accuracy: {backtest_results['Accuracy'] * 100:.2f}%")
print(f"Total Bets Placed: {backtest_results['Total Bets']}")

# View detailed bet data
bet_data_df = backtest_results['bet_data']
print(bet_data_df.head())
