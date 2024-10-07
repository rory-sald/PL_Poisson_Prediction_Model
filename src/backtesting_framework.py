# backtesting_framework.py

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

    # Handle missing values
    # Depending on the data, you might choose to fill, drop, or impute missing values
    # Here, we'll fill numerical columns with their mean and categorical with mode
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

    return data


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates predictive features for the model.

    :param data: DataFrame containing match data.
    :return: DataFrame with new features.
    """
    data = data.copy()

    # Define the binary target 'Over2.5' based on actual total goals
    data['TotalGoals'] = data['FTHG'] + data['FTAG']
    data['Over2.5'] = (data['TotalGoals'] > 2.5).astype(int)
    print("Created binary target 'Over2.5' based on total goals.")

    # Calculate total points for each match
    data['HomeTeamPoints'] = data.apply(
        lambda row: 3 if row['FTHG'] > row['FTAG'] else (1 if row['FTHG'] == row['FTAG'] else 0), axis=1)
    data['AwayTeamPoints'] = data.apply(
        lambda row: 3 if row['FTAG'] > row['FTHG'] else (1 if row['FTAG'] == row['FTHG'] else 0), axis=1)

    # Function to calculate rolling sum of points for the last 5 matches
    def calculate_recent_points(group):
        return group.shift().rolling(window=5, min_periods=1).sum()

    # Calculate recent points for Home and Away teams using transform instead of apply
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
    # To avoid complexity, assume head-to-head features are calculated without the current match
    # Here's a simplified approach using cumulative averages

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
        ].iloc[:idx]  # Matches before current match

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
    data.drop(columns=['TotalGoals'], inplace=True)

    return data


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
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of Over 2.5 goals
    return predictions, probabilities


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


def run_backtests(
    data: pd.DataFrame,
    models: Dict[str, Any],
    feature_columns: List[str],
    label_column: str,
    start_index: int
) -> Dict[str, Any]:
    """
    Runs backtests for multiple models.

    :param data: DataFrame containing the data.
    :param models: Dictionary of model names and model instances.
    :param feature_columns: List of feature column names.
    :param label_column: Name of the label column.
    :param start_index: Index to start backtesting from.
    :return: Dictionary containing evaluation metrics and results for each model.
    """
    all_results = {}
    for model_name, model in models.items():
        print(f"\nRunning backtest for model: {model_name}")
        results = backtest_model(data, model, feature_columns, label_column, start_index)
        
        if not results['predictions']:
            print(f"No predictions made for model: {model_name}. Skipping evaluation.")
            continue

        evaluation_metrics = evaluate_model_performance(
            results['actuals'], results['predictions'], results['probabilities']
        )
        results_df = collect_results(model_name, results)
        all_results[model_name] = {
            'evaluation_metrics': evaluation_metrics,
            'results_df': results_df
        }
    return all_results



data = load_data()
data = create_features(data)

# Define feature columns and label
feature_columns = [
    'HomeTeamGoalsAvg', 'HomeTeamGoalsConcededAvg',
    'AwayTeamGoalsAvg', 'AwayTeamGoalsConcededAvg',
    'HomeTeamRecentPoints', 'AwayTeamRecentPoints',
    'H2HAvgGoals', 'HomeTeamGoalDifference', 'AwayTeamGoalDifference'
]
label_column = 'Over2.5'  # Correctly defined binary label

# Ensure all feature columns exist
missing_features = set(feature_columns) - set(data.columns)
if missing_features:
    raise ValueError(f"The following feature columns are missing from the data: {missing_features}")

# Define models to test
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Run backtests
start_index = 100  # Start backtesting after 100 matches to allow sufficient training data
all_results = run_backtests(data, models, feature_columns, label_column, start_index)

# Analyze and compare results
for model_name, model_results in all_results.items():
    print(f"\nModel: {model_name}")
    eval_metrics = model_results['evaluation_metrics']
    print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {eval_metrics['roc_auc']:.4f}")
    print("Classification Report:")
    # Convert the classification report dictionary back to string for printing
    report = classification_report(
        model_results['results_df']['Actual'],
        model_results['results_df']['Predicted'],
        target_names=['Under', 'Over']
    )
    print(report)

# Optionally, save results to CSV
combined_results = pd.concat([model_results['results_df'] for model_results in all_results.values()])
# combined_results.to_csv('backtest_results.csv', index=False)