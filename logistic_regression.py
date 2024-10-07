# backtesting_framework.py

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone


# =====================
# 1. Data Loading and Preprocessing
# =====================

def load_data() -> pd.DataFrame:
    """
    Loads football match data from multiple seasons.

    :return: Combined and preprocessed DataFrame containing all seasons' data.
    """
    # Define the file paths for each season using meaningful naming conventions
    season_file_paths = {
        '2020_21': '/Users/rems/Documents/Python/Scraper/E4.csv',
        '2021_22': '/Users/rems/Documents/Python/Scraper/E3.csv',
        '2022_23': '/Users/rems/Documents/Python/Scraper/E2.csv',
        '2023_24': '/Users/rems/Documents/Python/Scraper/E1.csv',
        '2024_25': '/Users/rems/Documents/Python/Scraper/E0.csv'
    }

    # Initialize a list to hold DataFrames for each season
    seasonal_data = []

    # Load data for each season
    for season, file_path in season_file_paths.items():
        try:
            season_data = pd.read_csv(file_path)
            season_data['Season'] = season  # Add a column to indicate the season
            seasonal_data.append(season_data)
            print(f"Loaded data for season {season} from {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}. Skipping this season.")
        except pd.errors.EmptyDataError:
            print(f"No data: {file_path} is empty. Skipping this season.")
        except Exception as e:
            print(f"An error occurred while loading {file_path}: {e}. Skipping this season.")

    # Combine all seasons' data into a single DataFrame
    if not seasonal_data:
        raise ValueError("No seasonal data was loaded. Please check the file paths and data availability.")

    full_data = pd.concat(seasonal_data, ignore_index=True)
    print(f"Combined data shape: {full_data.shape}")

    # Convert 'Date' to datetime format
    try:
        full_data['Date'] = pd.to_datetime(full_data['Date'], format='%d/%m/%Y')
    except Exception as e:
        print(f"Error converting 'Date' column to datetime: {e}")
        raise

    # Select relevant columns, including betting odds for over/under and others as needed
    columns_needed = [
        'Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
        'B365>2.5', 'B365<2.5', 'B365H', 'B365D', 'B365A'
    ]

    # Check if all required columns exist
    missing_columns = set(columns_needed) - set(full_data.columns)
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the data: {missing_columns}")


    data = full_data[columns_needed].copy()
    print(f"Selected columns: {data.columns.tolist()}")

    # Convert 'B365>2.5' to float, coerce errors to NaN
    data['B365>2.5'] = pd.to_numeric(data['B365>2.5'], errors='coerce')

    # Handle missing values
    # Fill numerical columns with their mean and categorical with mode
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            mean_value = data[col].mean()
            data[col].fillna(mean_value, inplace=True)
            print(f"Filled missing values in numerical column '{col}' with mean value {mean_value:.2f}")

    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode_value = data[col].mode()[0]
            data[col].fillna(mode_value, inplace=True)
            print(f"Filled missing values in categorical column '{col}' with mode value '{mode_value}'")

    # Optional: Remove duplicates if any
    before_dedup = data.shape[0]
    data.drop_duplicates(inplace=True)
    after_dedup = data.shape[0]
    print(f"Removed {before_dedup - after_dedup} duplicate rows.")

    # Verify 'B365>2.5' statistics
    print("B365>2.5 Odds Statistics:")
    print(data['B365>2.5'].describe())

    return data

# =====================
# 2. Feature Engineering
# =====================

def create_features(data: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
    """
    Creates predictive features for the model.

    :param data: DataFrame containing match data.
    :param include_targets: Whether to include target variables in feature calculations.
    :return: DataFrame with new features.
    """
    data = data.copy()

    # Only calculate 'TotalGoals' and 'Over2.5' if targets are included
    if include_targets:
        # Define the binary target 'Over2.5' based on actual total goals
        data['TotalGoals'] = data['FTHG'] + data['FTAG']
        data['Over2.5'] = (data['TotalGoals'] > 2.5).astype(int)
        print("Created binary target 'Over2.5' based on total goals.")
    else:
        data['FTHG'] = np.nan
        data['FTAG'] = np.nan
        data['Over2.5'] = np.nan

    # Calculate total points for each match
    data['HomeTeamPoints'] = data.apply(
        lambda row: 3 if row['FTHG'] > row['FTAG'] else (1 if row['FTHG'] == row['FTAG'] else 0) if include_targets else 0, axis=1)
    data['AwayTeamPoints'] = data.apply(
        lambda row: 3 if row['FTAG'] > row['FTHG'] else (1 if row['FTAG'] == row['FTHG'] else 0) if include_targets else 0, axis=1)

    # Function to calculate rolling sum of points for the last 5 matches
    def calculate_recent_points(group):
        return group.shift().rolling(window=5, min_periods=1).sum()

    # Calculate recent points for Home and Away teams using transform
    data['HomeTeamRecentPoints'] = data.groupby('HomeTeam')['HomeTeamPoints'].transform(calculate_recent_points)
    data['AwayTeamRecentPoints'] = data.groupby('AwayTeam')['AwayTeamPoints'].transform(calculate_recent_points)

    # Average goals scored and conceded in last 5 matches
    data['HomeTeamGoalsAvg'] = data.groupby('HomeTeam')['FTHG'].transform(
        lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    data['HomeTeamGoalsConcededAvg'] = data.groupby('HomeTeam')['FTAG'].transform(
        lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    data['AwayTeamGoalsAvg'] = data.groupby('AwayTeam')['FTAG'].transform(
        lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    data['AwayTeamGoalsConcededAvg'] = data.groupby('AwayTeam')['FTHG'].transform(
        lambda x: x.shift().rolling(window=5, min_periods=1).mean())

    # Goal Difference Features
    data['HomeTeamGoalDifference'] = data['HomeTeamGoalsAvg'] - data['HomeTeamGoalsConcededAvg']
    data['AwayTeamGoalDifference'] = data['AwayTeamGoalsAvg'] - data['AwayTeamGoalsConcededAvg']

    # Head-to-head average goals excluding the current match
    # Sort data by date to ensure chronological order
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Initialize H2HAvgGoals with NaN
    data['H2HAvgGoals'] = np.nan

    # Iterate through each match to calculate H2HAvgGoals excluding current match
    for idx, row in data.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        past_matches = data[
            ((data['HomeTeam'] == home) & (data['AwayTeam'] == away)) |
            ((data['HomeTeam'] == away) & (data['AwayTeam'] == home))
        ]
        past_matches = past_matches[past_matches['Date'] < row['Date']]  # Only past matches

        if not past_matches.empty:
            total_past_goals = past_matches['FTHG'] + past_matches['FTAG']
            avg_past_goals = total_past_goals.mean()
            data.at[idx, 'H2HAvgGoals'] = avg_past_goals
        else:
            # If no past head-to-head matches, fill with global mean
            data.at[idx, 'H2HAvgGoals'] = data['H2HAvgGoals'].mean()

    # Handle any remaining missing values by separating numeric and categorical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Fill numeric columns with mean
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # Fill categorical columns with mode
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Drop 'TotalGoals' as it's no longer needed
    if 'TotalGoals' in data.columns:
        data.drop(columns=['TotalGoals'], inplace=True)

    return data

# =====================
# 3. Model Training and Prediction
# =====================

def train_and_predict_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains the model and makes predictions.

    :param model: The machine learning model with fit and predict_proba methods.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_test: Test features.
    :return: Tuple of predictions and predicted probabilities.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]  # Probability of class '1' (Over2.5)
    else:
        # For models that do not have predict_proba (e.g., some versions of XGBClassifier), use decision_function
        probabilities = model.decision_function(X_test)
        probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())  # Normalize to [0,1]
    return predictions, probabilities

# =====================
# 4. Backtesting Function
# =====================

def backtest_model(
    data: pd.DataFrame,
    model: Any,
    feature_columns: List[str],
    label_column: str,
    start_index: int
) -> Dict[str, Any]:
    """
    Performs backtesting using a rolling window approach.

    :param data: DataFrame containing the data.
    :param model: Machine learning model to be tested.
    :param feature_columns: List of feature column names.
    :param label_column: Name of the label column.
    :param start_index: Index to start backtesting from.
    :return: Dictionary containing evaluation metrics and results.
    """
    predictions = []
    actuals = []
    probabilities = []
    dates = []

    for i in range(start_index, len(data)):
        train_data = data.iloc[:i]
        test_data = data.iloc[i:i+1]

        X_train = train_data[feature_columns]
        y_train = train_data[label_column]
        X_test = test_data[feature_columns]

        # Ensure there is enough data to train
        if len(X_train) < 50:
            continue

        try:
            pred, prob = train_and_predict_model(model, X_train, y_train, X_test)
            predictions.extend(pred)
            actuals.extend(test_data[label_column].values)
            probabilities.extend(prob)
            dates.extend(test_data['Date'].values)
        except Exception as e:
            print(f"Error during model training/prediction at index {i}: {e}")
            continue

    results = {
        'dates': dates,
        'actuals': actuals,
        'predictions': predictions,
        'probabilities': probabilities
    }

    return results

# =====================
# 5. Model Evaluation
# =====================

def evaluate_model_performance(actuals: List[int], predictions: List[int], probabilities: List[float]) -> Dict[str, Any]:
    """
    Evaluates the model performance using various metrics.

    :param actuals: List of actual labels.
    :param predictions: List of predicted labels.
    :param probabilities: List of predicted probabilities.
    :return: Dictionary containing evaluation metrics.
    """
    if len(set(actuals)) < 2:
        print("Only one class present in actuals. ROC-AUC cannot be computed.")
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(actuals, probabilities)

    accuracy = accuracy_score(actuals, predictions)
    report = classification_report(actuals, predictions, target_names=['Under', 'Over'], output_dict=True)

    evaluation_metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'classification_report': report
    }

    return evaluation_metrics

# =====================
# 6. Collecting Results
# =====================

def collect_results(
    model_name: str,
    results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Collects results into a DataFrame.

    :param model_name: Name of the model.
    :param results: Dictionary containing dates, actuals, predictions, and probabilities.
    :return: DataFrame containing the results.
    """
    results_df = pd.DataFrame({
        'Date': results['dates'],
        'Model': model_name,
        'Actual': results['actuals'],
        'Predicted': results['predictions'],
        'Probability': results['probabilities']
    })
    return results_df

# =====================
# 7. Identifying Value Bets
# =====================

def identify_value_bets(
    data: pd.DataFrame,
    model: Any,
    feature_columns: List[str],
    odds_column: str,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Identifies value bets where the model's estimated probability exceeds the bookmaker's implied probability.

    :param data: DataFrame containing match data with features and odds.
    :param model: Trained machine learning model.
    :param feature_columns: List of feature column names used for prediction.
    :param odds_column: Column name containing the bookmaker's odds for the event (e.g., 'B365>2.5').
    :param threshold: Minimum difference between model probability and implied probability to consider as value bet.
    :return: DataFrame containing matches identified as value bets.
    """
    # Ensure the model is trained by checking for 'classes_' attribute
    if not hasattr(model, 'classes_'):
        raise ValueError("The model provided is not trained. Please train the model before identifying value bets.")

    # Predict the probability of Over 2.5 goals using the model
    X = data[feature_columns]
    if hasattr(model, 'predict_proba'):
        data['ModelProbabilityOver2.5'] = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        data['ModelProbabilityOver2.5'] = model.decision_function(X)
        # Normalize to [0,1] if needed
        data['ModelProbabilityOver2.5'] = (data['ModelProbabilityOver2.5'] - data['ModelProbabilityOver2.5'].min()) / (data['ModelProbabilityOver2.5'].max() - data['ModelProbabilityOver2.5'].min())
    else:
        raise AttributeError("Model does not have predict_proba or decision_function methods.")

    # Calculate implied probability from bookmaker's odds
    data['ImpliedProbability'] = 1 / data[odds_column]

    # Calculate the difference between model's probability and implied probability
    data['ProbabilityDifference'] = data['ModelProbabilityOver2.5'] - data['ImpliedProbability']

    # Identify value bets where the difference exceeds the threshold
    value_bets = data[data['ProbabilityDifference'] > threshold].copy()

    # Calculate Expected Value (EV) for each value bet
    value_bets['ExpectedValue'] = (data.loc[value_bets.index, 'ModelProbabilityOver2.5'] * data.loc[value_bets.index, odds_column]) - 1

    # Sort by the highest Expected Value
    value_bets.sort_values('ExpectedValue', ascending=False, inplace=True)

    # Sort value_bets by Date in ascending order
    value_bets.sort_values('Date', inplace=True)

    print(f"Identified {len(value_bets)} value bets out of {len(data)} matches.")

    # Select relevant columns for output, including 'Over2.5'
    value_bets = value_bets[['Date', 'Season', 'HomeTeam', 'AwayTeam', 'Over2.5', 'ModelProbabilityOver2.5', 
                             'ImpliedProbability', 'ProbabilityDifference', 'ExpectedValue', odds_column]]

    return value_bets

# =====================
# 8. Running Backtests for Multiple Models
# =====================

def run_backtests(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    models: Dict[str, Any],
    feature_columns: List[str],
    label_column: str,
    odds_column: str,
    staking_strategy: str = 'kelly',
    kelly_fraction: float = 0.05,
    fixed_stake: float = 10.0
) -> Dict[str, Any]:
    """
    Runs backtests for multiple models using pre-split training and test data.
    """
    all_results = {}

    X_train = train_data[feature_columns]
    y_train = train_data[label_column]

    for model_name, model in models.items():
        print(f"\nRunning backtest for model: {model_name}")

        # Train the model
        model.fit(X_train, y_train)

        # Prepare test data
        X_test = test_data[feature_columns]
        y_test = test_data[label_column]

        # Predict probabilities
        if hasattr(model, 'predict_proba'):
            test_data['PredictedProbability'] = model.predict_proba(X_test)[:, 1]
        else:
            test_data['PredictedProbability'] = model.predict(X_test)

        # Calculate implied probabilities
        test_data['ImpliedProbability'] = 1 / test_data[odds_column]

        # Calculate probability differences
        test_data['ProbabilityDifference'] = test_data['PredictedProbability'] - test_data['ImpliedProbability']

        # Identify value bets
        value_bets = test_data[test_data['ProbabilityDifference'] > 0.05].copy()

        # Simulate betting
        betting_results, bet_details = simulate_betting(
            value_bets=value_bets,
            initial_bankroll=1000.0,
            staking_strategy=staking_strategy,
            kelly_fraction=kelly_fraction,
            fixed_stake=fixed_stake,
            odds_column=odds_column,
            label_column=label_column
        )

        # Evaluate model performance
        evaluation_metrics = evaluate_model_performance(
            y_test.dropna(),
            (test_data.loc[y_test.dropna().index, 'PredictedProbability'] >= 0.5).astype(int),
            test_data.loc[y_test.dropna().index, 'PredictedProbability']
        )

        # Collect results
        all_results[model_name] = {
            'evaluation_metrics': evaluation_metrics,
            'results_df': test_data[['Date', 'HomeTeam', 'AwayTeam', label_column, 'PredictedProbability']],
            'betting_results': betting_results,
            'bet_details': bet_details,
            'performance_metrics': calculate_performance_metrics(betting_results, 1000.0)
        }

    return all_results

# =====================
# 9. Main Function
# =====================

def calculate_stake_kelly(prob: float, odds: float, bankroll: float, fraction: float = 0.05) -> float:
    """
    Calculates the stake using the Kelly Criterion.

    :param prob: Model's estimated probability of the outcome.
    :param odds: Bookmaker's decimal odds for the outcome.
    :param bankroll: Current bankroll.
    :param fraction: Fraction of the Kelly stake to use (for fractional Kelly).
    :return: Calculated stake amount.
    """
    # Calculate the Kelly fraction
    kelly_fraction = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
    
    # Ensure Kelly fraction is non-negative
    kelly_fraction = max(kelly_fraction, 0)
    
    # Apply fractional Kelly
    stake = bankroll * kelly_fraction * fraction
    
    return stake

def simulate_betting(
    value_bets: pd.DataFrame,
    initial_bankroll: float,
    staking_strategy: str,
    kelly_fraction: float,
    fixed_stake: float,
    odds_column: str,
    label_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates betting on value bets and collects detailed bet information.
    """
    bankroll = initial_bankroll
    bankroll_history = []
    bet_details = []

    for idx, bet in value_bets.iterrows():
        date = bet['Date']
        home_team = bet['HomeTeam']
        away_team = bet['AwayTeam']
        odds = bet[odds_column]
        prob = bet['PredictedProbability']
        implied_prob = bet['ImpliedProbability']
        probability_difference = bet['ProbabilityDifference']
        actual = bet[label_column]

        # Determine stake
        if staking_strategy == 'fixed':
            stake = fixed_stake
        elif staking_strategy == 'kelly':
            stake = calculate_stake_kelly(prob, odds, bankroll, fraction=kelly_fraction)
        else:
            raise ValueError("Invalid staking strategy.")

        # Ensure stake does not exceed bankroll
        stake = min(stake, bankroll)
        if stake <= 0:
            continue

        # Simulate bet outcome
        if actual == 1:  # Over 2.5 goals
            profit = stake * (odds - 1)
            bankroll += profit
            outcome = 'Win'
        else:
            bankroll -= stake
            outcome = 'Lose'

        # Record bankroll history
        bankroll_history.append({
            'Date': date,
            'Bankroll': bankroll,
            'CumulativeProfit': bankroll - initial_bankroll,
            'Outcome': outcome,
            'Stake': stake
        })

        # Record bet details
        bet_details.append({
            'Date': date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Odds': odds,
            'PredictedProbability': prob,
            'ImpliedProbability': implied_prob,
            'ProbabilityDifference': probability_difference,
            'Stake': stake,
            'Outcome': outcome,
            'ActualResult': actual,
            'Bankroll': bankroll
        })

    # Create DataFrames
    bankroll_df = pd.DataFrame(bankroll_history)
    bet_details_df = pd.DataFrame(bet_details)

    return bankroll_df, bet_details_df

def calculate_performance_metrics(betting_results: pd.DataFrame, initial_bankroll: float) -> Dict[str, Any]:
    """
    Calculates additional performance metrics.

    :param betting_results: DataFrame containing betting simulation results.
    :param initial_bankroll: Starting bankroll.
    :return: Dictionary with performance metrics.
    """
    bankroll = betting_results['Bankroll']
    cumulative_profit = betting_results['CumulativeProfit']

    # Maximum Drawdown
    rolling_max = bankroll.cummax()
    drawdown = rolling_max - bankroll
    max_drawdown = drawdown.max()

    # Sharpe Ratio (assuming risk-free rate is 0)
    returns = bankroll.pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized

    # Total Profit
    total_profit = cumulative_profit.iloc[-1]

    metrics = {
        'Final Bankroll': bankroll.iloc[-1],
        'Total Profit': total_profit,
        'ROI (%)': (total_profit / initial_bankroll) * 100,
        'Maximum Drawdown (€)': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Total Bets': len(betting_results),
        'Wins': betting_results['Outcome'].value_counts().get('Win', 0),
        'Losses': betting_results['Outcome'].value_counts().get('Lose', 0),
        'Win Rate (%)': betting_results['Outcome'].value_counts(normalize=True).get('Win', 0) * 100
    }

    return metrics

def main():
    # Load and preprocess data
    data = load_data()

    # Ensure data is sorted chronologically
    data = data.sort_values('Date').reset_index(drop=True)

    # Split data into training and testing sets
    train_size = 0.8
    split_index = int(len(data) * train_size)
    train_data_raw = data.iloc[:split_index]
    test_data_raw = data.iloc[split_index:]

    # Create features for training data (include targets)
    train_data = create_features(train_data_raw, include_targets=True)

    # Create features for test data (exclude targets)
    test_data = create_features(test_data_raw, include_targets=False)

    # Define feature columns and label
    feature_columns = [
        'HomeTeamGoalsAvg', 'HomeTeamGoalsConcededAvg',
        'AwayTeamGoalsAvg', 'AwayTeamGoalsConcededAvg',
        'HomeTeamRecentPoints', 'AwayTeamRecentPoints',
        'H2HAvgGoals', 'HomeTeamGoalDifference', 'AwayTeamGoalDifference'
    ]
    label_column = 'Over2.5'  # Binary label

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
    }

    # Define bookmaker's odds column
    odds_column = 'B365>2.5'

    # Run backtests
    all_results = run_backtests(
        train_data,
        test_data,
        models,
        feature_columns,
        label_column,
        odds_column,
        staking_strategy='kelly',
        kelly_fraction=0.05,
        fixed_stake=10.0
    )

    # Rest of your code...

    # Analyze results
    for model_name, model_results in all_results.items():
        print(f"\nModel: {model_name}")
        eval_metrics = model_results['evaluation_metrics']
        print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"ROC-AUC: {eval_metrics['roc_auc']:.4f}")
        print("Classification Report:")
        report = classification_report(
            model_results['results_df'][label_column],
            (model_results['results_df']['PredictedProbability'] >= 0.5).astype(int),
            target_names=['Under', 'Over']
        )
        print(report)

        # Betting Simulation Results
        perf_metrics = model_results['performance_metrics']
        print("\nBetting Simulation Performance Metrics:")
        for key, value in perf_metrics.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

        # Display first few bets
        print("\nFirst few bets:")
        print(model_results['bet_details'].head())

        # Plot bankroll growth
        plot_bankroll_growth(model_results['betting_results'])

    # Save bet details
    for model_name, model_results in all_results.items():
        model_results['bet_details'].to_csv(f'bet_details_{model_name.replace(" ", "_").lower()}.csv', index=False)
        print(f"\nSaved {model_name} bet details to CSV.")


# =====================
# 11. Visualization Functions
# =====================

def plot_metrics(all_results: Dict[str, Any]):
    """
    Plots a bar chart comparing Accuracy and ROC-AUC for all models.

    :param all_results: Dictionary containing evaluation metrics and results for each model.
    """
    # Create a DataFrame with Accuracy and ROC-AUC
    metrics_data = {
        'Model': [],
        'Accuracy': [],
        'ROC-AUC': []
    }

    for model_name, model_results in all_results.items():
        metrics_data['Model'].append(model_name)
        metrics_data['Accuracy'].append(model_results['evaluation_metrics']['accuracy'])
        metrics_data['ROC-AUC'].append(model_results['evaluation_metrics']['roc_auc'])

    df_metrics = pd.DataFrame(metrics_data)

    # Set the style for better aesthetics
    sns.set(style="whitegrid")

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(df_metrics))

    # Plot Accuracy
    sns.barplot(x='Model', y='Accuracy', data=df_metrics, color='skyblue', label='Accuracy')

    # Plot ROC-AUC
    sns.barplot(x='Model', y='ROC-AUC', data=df_metrics, color='salmon', label='ROC-AUC', alpha=0.7)

    # Add labels and title
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Accuracy and ROC-AUC Comparison', fontsize=14)
    plt.ylim(0, 1)  # Since scores range between 0 and 1

    # Rotate x-axis labels if necessary
    plt.xticks(rotation=45)

    # Add legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

def plot_classification_reports(all_results: Dict[str, Any]):
    """
    Plots Precision, Recall, and F1-Score for each model and class.

    :param all_results: Dictionary containing evaluation metrics and results for each model.
    """
    # Define the classification metrics for each model and class
    metrics = {
        'Model': [],
        'Class': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    for model_name, model_results in all_results.items():
        report = model_results['evaluation_metrics']['classification_report']
        for cls in ['Under', 'Over']:
            metrics['Model'].append(model_name)
            metrics['Class'].append(cls)
            metrics['Precision'].append(report[cls]['precision'])
            metrics['Recall'].append(report[cls]['recall'])
            metrics['F1-Score'].append(report[cls]['f1-score'])

    # Create DataFrame
    df_class_metrics = pd.DataFrame(metrics)

    # Melt the DataFrame for easier plotting
    df_melted = df_class_metrics.melt(id_vars=['Model', 'Class'], 
                                      value_vars=['Precision', 'Recall', 'F1-Score'],
                                      var_name='Metric', value_name='Score')

    # Set the style
    sns.set(style="whitegrid")

    # Create a FacetGrid for separate plots
    g = sns.catplot(
        data=df_melted,
        kind='bar',
        x='Model',
        y='Score',
        hue='Metric',
        col='Class',
        palette='viridis',
        height=5,
        aspect=1
    )

    # Add titles and adjust layout
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Precision, Recall, and F1-Score by Model and Class', fontsize=16)

    # Rotate x-axis labels if necessary
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.show()

def plot_roc_curves(all_results: Dict[str, Any], data: pd.DataFrame, feature_columns: List[str], label_column: str, start_index: int):
    """
    Plots ROC Curves for all models.

    :param all_results: Dictionary containing evaluation metrics and results for each model.
    :param data: DataFrame containing the data.
    :param feature_columns: List of feature column names.
    :param label_column: Name of the label column.
    :param start_index: Index to start backtesting from.
    """
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    for model_name, model_results in all_results.items():
        # Extract actuals and probabilities
        y_true = model_results['results_df']['Actual']
        y_prob = model_results['results_df']['Probability']

        if np.isnan(model_results['evaluation_metrics']['roc_auc']):
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Configure plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def plot_value_bets_distribution(value_bets: pd.DataFrame):
    """
    Plots the distribution of Probability Differences and Expected Values for value bets.

    :param value_bets: DataFrame containing matches identified as value bets.
    """
    sns.set(style="whitegrid")

    # Distribution of Probability Differences
    plt.figure(figsize=(10,6))
    sns.histplot(value_bets['ProbabilityDifference'], bins=50, kde=True, color='green')
    plt.title('Distribution of Probability Differences for Value Bets')
    plt.xlabel('Probability Difference')
    plt.ylabel('Frequency')
    plt.show()

    # Distribution of Expected Values
    plt.figure(figsize=(10,6))
    sns.histplot(value_bets['ExpectedValue'], bins=50, kde=True, color='purple')
    plt.title('Distribution of Expected Values for Value Bets')
    plt.xlabel('Expected Value')
    plt.ylabel('Frequency')
    plt.show()

def plot_bankroll_growth(betting_results: pd.DataFrame):
    """
    Plots the bankroll growth over time.

    :param betting_results: DataFrame containing betting simulation results.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(betting_results['Date'], betting_results['Bankroll'], marker='o', linestyle='-')
    plt.title('Bankroll Growth Over Time')
    plt.xlabel('Date')
    plt.ylabel('Bankroll (€)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_drawdown(betting_results: pd.DataFrame):
    """
    Plots the drawdown over time.

    :param betting_results: DataFrame containing betting simulation results.
    """
    bankroll = betting_results['Bankroll']
    rolling_max = bankroll.cummax()
    drawdown = rolling_max - bankroll

    plt.figure(figsize=(12, 6))
    plt.plot(betting_results['Date'], drawdown, color='red', linestyle='--')
    plt.title('Drawdown Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (€)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sharpe_ratio(betting_results: pd.DataFrame):
    """
    Plots the Sharpe Ratio over time.

    :param betting_results: DataFrame containing betting simulation results.
    """
    returns = betting_results['Bankroll'].pct_change().fillna(0)
    rolling_sharpe = (returns.rolling(window=30).mean() / returns.rolling(window=30).std()) * np.sqrt(252)

    plt.figure(figsize=(12, 6))
    plt.plot(betting_results['Date'], rolling_sharpe, color='blue')
    plt.title('Rolling Sharpe Ratio Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# =====================
# 10. Running the Backtesting and Value Bets Identification
# =====================

if __name__ == "__main__":
    main()
