import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from backtest_module import load_data

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import compute_class_weight
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import optuna
from sklearn.linear_model import LogisticRegression

def run_backtest_over_under_ml(data, backtest_matchdays=70):
    """
    Runs a backtest for Over/Under 2.5 goals prediction using an XGBoost model,
    ensuring that only data available up to each match date is used for training.
    Retrains the model once per unique match date.
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
    last_trained_date = None  # To track retraining per date

    # Iterate through each match in the backtest period
    for idx in tqdm(range(start_index, len(data)), desc="Processing matches", unit="match"):
        match = data.iloc[idx]
        match_date = match['Date']
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']

        # Retrain the model if the date has changed or if model is None
        if match_date != last_trained_date or model is None:
            # Use data up to the match date
            train_data = data[data['Date'] < match_date].copy()
            X_train, y_train = prepare_features_and_labels(train_data)

            # Check if there is sufficient data to train
            if len(X_train) < 50:
                print(f"Skipping match on {match_date.date()} between {home_team} and {away_team} due to insufficient training data ({len(X_train)} samples).")
                continue  # Skip if not enough data

            # Handle class imbalance using SMOTE
            X_train_res, y_train_res = handle_class_imbalance(X_train, y_train)

            # Hyperparameter tuning with Optuna (optional)
            # Uncomment the following line to perform hyperparameter tuning
            # model = train_model_with_optuna(X_train_res, y_train_res)

            # Train the model
            model = train_model(X_train_res, y_train_res)  # Basic training with class weights

            print(f"Model trained with {len(X_train)} samples up to {match_date.date()}.")

            # Optionally, plot feature importances once per retraining
            # Uncomment the following line if you wish to visualize feature importances during backtest
            # plot_feature_importances(model, X_train_res)

            # Update the last trained date
            last_trained_date = match_date

        # Prepare test data using the corrected function
        X_test, y_test = prepare_match_features(match, data)

        if X_test.empty:
            print(f"Skipping match on {match_date.date()} between {home_team} and {away_team} due to insufficient feature data.")
            continue  # Skip if no test data

        # Make prediction
        prob_over = model.predict_proba(X_test)[:, 1][0]
        print(f"Predicted probability for Over: {prob_over:.2f}")

        # Determine betting decision based on value betting
        implied_prob_over = 1 / match['B365>2.5']
        implied_prob_under = 1 / match['B365<2.5']

        if prob_over > implied_prob_over:
            bet_over_under = 'Over'
        elif (1 - prob_over) > implied_prob_under:
            bet_over_under = 'Under'
        else:
            bet_over_under = None  # No bet placed

        # Store predictions and actuals
        if bet_over_under:
            predictions.append(prob_over if bet_over_under == 'Over' else 1 - prob_over)
            actuals.append(y_test.iloc[0])
            dates.append(match_date)

            # Create bet data dictionary
            bet_data = {
                'Date': match_date,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'ActualOverUnder': 'Over' if y_test.iloc[0] == 1 else 'Under',
                'PredictedProbOver': prob_over,
                'BetOverUnder': bet_over_under,
                'B365>2.5': match['B365>2.5'],
                'B365<2.5': match['B365<2.5'],
                'TotalGoals': match['FTHG'] + match['FTAG']
            }
            bet_data_list.append(bet_data)

            print(f"Placed bet on {bet_over_under} for match on {match_date.date()} with predicted probability {prob_over:.2f}.")
        else:
            print(f"No profitable bet placed for match on {match_date.date()}.")

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

    # Evaluate additional performance metrics
    y_true = np.array(actuals)
    y_pred = np.array([1 if p >= 0.5 else 0 for p in predictions])
    y_prob = np.array(predictions)

    evaluate_performance(y_true, y_pred, y_prob)

    return {
        'Total Profit': total_profit,
        'ROI (%)': roi,
        'Accuracy': accuracy,
        'Total Bets': total_bets,
        'bet_data': bet_data_df
    }

def prepare_features_and_labels(train_data):
    """
    Prepares features and labels for training the model.

    :param train_data: DataFrame containing historical match data up to the training date.
    :return: X (DataFrame of features), y (Series of labels)
    """
    features_list = []
    labels = []

    for _, match in train_data.iterrows():
        X, y = prepare_match_features(match, train_data)
        if not X.empty:
            features_list.append(X.iloc[0])
            labels.append(y.iloc[0])

    if not features_list:
        return pd.DataFrame(), pd.Series()

    X_train = pd.DataFrame(features_list)
    y_train = pd.Series(labels)

    return X_train, y_train

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
        h2h_exists = 1
    else:
        h2h_avg_goals = 0  # Neutral value when no H2H data
        h2h_exists = 0

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
        'H2HExists': h2h_exists,
        'TotalGoals': match['FTHG'] + match['FTAG']
    }

    # Create DataFrame for features
    features_df = pd.DataFrame([feature_dict])

    # Handle missing values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())

    # Define features (X) and label (y)
    X = features_df.drop(columns=['TotalGoals'])
    y = features_df['TotalGoals'].apply(lambda x: 1 if x > 2.5 else 0)

    # One-Hot Encoding for categorical features (if any)
    # X = pd.get_dummies(X, columns=['HomeTeam', 'AwayTeam'], drop_first=True)

    return X, y

def handle_class_imbalance(X_train, y_train):
    """
    Handles class imbalance using SMOTE.

    :param X_train: DataFrame of training features.
    :param y_train: Series of training labels.
    :return: Resampled X_train and y_train.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def train_model(X_train, y_train):
    """
    Trains an XGBoost Classifier with class weights to handle imbalance.

    :param X_train: DataFrame of training features.
    :param y_train: Series of training labels.
    :return: Trained XGBoost model.
    """
    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    scale_pos_weight = class_weights[0] / class_weights[1]  # Assuming class '1' is Over

    # Instantiate the XGBoost model with class weights
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False  # Removed to eliminate warning
    )

    # Train the model
    model.fit(X_train, y_train)

    return model

def train_model_with_optuna(X_train, y_train):
    """
    Trains an XGBoost Classifier with hyperparameter tuning using Optuna.

    :param X_train: DataFrame of training features.
    :param y_train: Series of training labels.
    :return: Best trained XGBoost model.
    """
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
        }

        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=compute_scale_pos_weight(y_train),
            **param
        )

        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best Parameters:", study.best_params)
    print("Best Accuracy:", study.best_value)

    # Train the model with best parameters
    best_params = study.best_params
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=compute_scale_pos_weight(y_train),
        **best_params
    )

    model.fit(X_train, y_train)
    return model

def compute_scale_pos_weight(y):
    """
    Computes scale_pos_weight based on class imbalance.

    :param y: Series of labels.
    :return: scale_pos_weight value.
    """
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return class_weights[0] / class_weights[1]  # Assuming class '1' is Over

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

def plot_feature_importances(model, X_train):
    """
    Plots feature importances from the XGBoost model.

    :param model: Trained XGBoost model.
    :param X_train: DataFrame of training features.
    """
    # Get feature importances
    importance = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))  # Top 10 features
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def evaluate_performance(y_true, y_pred, y_prob):
    """
    Evaluates and prints classification metrics.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param y_prob: Predicted probabilities.
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Under', 'Over']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.2f}\n")

# Load your data
data = load_data()

# Run the backtest
backtest_results = run_backtest_over_under_ml(data, backtest_matchdays=70)

# Display summarized results
print("\nBacktest Performance:")
print(f"Total Profit: ${backtest_results['Total Profit']:.2f}")
print(f"ROI: {backtest_results['ROI (%)']:.2f}%")
print(f"Accuracy: {backtest_results['Accuracy'] * 100:.2f}%")
print(f"Total Bets Placed: {backtest_results['Total Bets']}")

# View detailed bet data
bet_data_df = backtest_results['bet_data']
print("\nDetailed Bet Data:")
print(bet_data_df.head())
