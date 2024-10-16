# forecast_and_suggest_bets.py

import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import json
from datetime import datetime, timedelta
import os

# =======================
# Step 1: Load Best Parameters
# =======================

# Load the saved best optimization results
try:
    with open('best_optimization_results.json', 'r') as f:
        best_results = json.load(f)
except FileNotFoundError:
    print("Error: 'best_optimization_results.json' file not found.")
    exit()

# Extract best parameters for match result strategy
best_params_match_result = {
    'decay_factor': float(best_results['match_result']['decay_factor']),
    'confidence_threshold': float(best_results['match_result']['confidence_threshold']),
    'confidence_multiplier': float(best_results['match_result']['confidence_multiplier']),
    'max_goals': int(best_results['match_result']['max_goals'])
}

# =======================
# Step 2: Load Historical Data
# =======================

# Load current and past seasons data
current_season_url = '/Users/rems/Documents/Python/Scraper/E0.csv'
last_season_url = '/Users/rems/Documents/Python/Scraper/E1.csv'
last_last_season_url = '/Users/rems/Documents/Python/Scraper/E2.csv'
last_last_last_season_url = '/Users/rems/Documents/Python/Scraper/E3.csv'

try:
    current_season_data = pd.read_csv(current_season_url)
    last_season_data = pd.read_csv(last_season_url)
    last_last_season_data = pd.read_csv(last_last_season_url)
    last_last_last_season_data = pd.read_csv(last_last_last_season_url)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# Combine all seasons' data
full_data = pd.concat(
    [last_last_last_season_data, last_last_season_data, last_season_data, current_season_data],
    ignore_index=True
)

# Convert 'Date' to datetime format
full_data['Date'] = pd.to_datetime(full_data['Date'], format='%d/%m/%Y')

# Select relevant columns
columns_needed = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
    'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR'
]
data = full_data[columns_needed].copy()  # Explicit copy to avoid SettingWithCopyWarning

# =======================
# Step 3: Prepare the Data Using Best Decay Factor
# =======================

# Calculate the number of days since each game occurred
data['Days_Ago'] = (pd.to_datetime('today') - data['Date']).dt.days

# Use the best decay factor
decay_factor = best_params_match_result['decay_factor']

# Calculate weights using exponential decay
data['Weight'] = decay_factor ** data['Days_Ago']

# Prepare home and away data
home_data = pd.DataFrame({
    'Team': data['HomeTeam'],
    'Opponent': data['AwayTeam'],
    'Goals_Scored': data['FTHG'],
    'Goals_Conceded': data['FTAG'],
    'Home_Away': 'Home',
    'Weight': data['Weight']
})

away_data = pd.DataFrame({
    'Team': data['AwayTeam'],
    'Opponent': data['HomeTeam'],
    'Goals_Scored': data['FTAG'],
    'Goals_Conceded': data['FTHG'],
    'Home_Away': 'Away',
    'Weight': data['Weight']
})

# Combine home and away data
full_data_prepared = pd.concat([home_data, away_data], ignore_index=True)

# =======================
# Step 4: Calculate Team Strengths
# =======================

# Filter data for home and away games
home_games = full_data_prepared[full_data_prepared['Home_Away'] == 'Home']
away_games = full_data_prepared[full_data_prepared['Home_Away'] == 'Away']

# Calculate weighted team statistics for home games
home_team_stats = home_games.groupby('Team').apply(
    lambda x: pd.Series({
        'Goals_Scored': np.average(x['Goals_Scored'], weights=x['Weight']),
        'Goals_Conceded': np.average(x['Goals_Conceded'], weights=x['Weight'])
    })
)

# Calculate weighted team statistics for away games
away_team_stats = away_games.groupby('Team').apply(
    lambda x: pd.Series({
        'Goals_Scored': np.average(x['Goals_Scored'], weights=x['Weight']),
        'Goals_Conceded': np.average(x['Goals_Conceded'], weights=x['Weight'])
    })
)

# Ensure team names are set as indices
home_team_stats.set_index(home_team_stats.index, inplace=True)
away_team_stats.set_index(away_team_stats.index, inplace=True)

# Calculate league averages
league_home_stats = home_games[['Goals_Scored', 'Goals_Conceded']].mean()
league_away_stats = away_games[['Goals_Scored', 'Goals_Conceded']].mean()

# Calculate attack and defense strengths
attack_strength_home = home_team_stats['Goals_Scored'] / league_home_stats['Goals_Scored']
defense_strength_home = home_team_stats['Goals_Conceded'] / league_away_stats['Goals_Scored']
attack_strength_away = away_team_stats['Goals_Scored'] / league_away_stats['Goals_Scored']
defense_strength_away = away_team_stats['Goals_Conceded'] / league_home_stats['Goals_Scored']

# Retain team names as indices
attack_strength_home.index = home_team_stats.index
defense_strength_home.index = home_team_stats.index
attack_strength_away.index = away_team_stats.index
defense_strength_away.index = away_team_stats.index

# =======================
# Step 5: Fetch Upcoming Fixtures with Odds (Caching Implementation)
# =======================

API_KEY_ODDS = 'API_KEY'  # Replace with your actual Odds API key
SPORT_KEY = 'soccer_epl'  # Ensure this is the correct sport key as per Odds API documentation
REGIONS = 'uk'  # Adjust as needed
CACHE_FILE = 'odds_data_cache.json'  # File to save the cached odds data

BASE_URL_ODDS = f'https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/'

params_odds = {
    'apiKey': API_KEY_ODDS,
    'regions': REGIONS,  # Specify regions as needed
    'markets': 'h2h,totals',  # Head-to-head and totals markets
    'oddsFormat': 'decimal',
    'dateFormat': 'iso'
}

def load_cached_odds():
    """Load odds data from cache file if it exists and is recent."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            cache = json.load(file)
            cache_time = datetime.fromisoformat(cache['timestamp'])
            if datetime.now() - cache_time < timedelta(hours=50):  # Set cache duration as needed
                print("Using cached odds data.")
                return cache['data']
    return None

def save_cached_odds(data):
    """Save odds data to cache file with the current timestamp."""
    cache = {
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    with open(CACHE_FILE, 'w') as file:
        json.dump(cache, file)

def get_upcoming_fixtures_data():
    # Try loading cached data first
    cached_data = load_cached_odds()
    if cached_data:
        return cached_data

    # If no cache or cache is outdated, fetch new data
    response = requests.get(BASE_URL_ODDS, params=params_odds)
    if response.status_code != 200:
        print(f"Error fetching odds: {response.status_code} - {response.text}")
        return []

    try:
        data_response = response.json()
        save_cached_odds(data_response)  # Save new data to cache
        return data_response
    except json.JSONDecodeError:
        print("Error decoding JSON response from Odds API.")
        return []

def get_upcoming_fixtures_with_odds(data_response):
    fixtures = []
    for match in data_response:
        home_team = match['home_team']
        away_team = match['away_team']
        bookmakers = match.get('bookmakers', [])
        odds_data = {}

        # Extract odds from bookmakers
        for bookmaker in bookmakers:
            markets = bookmaker.get('markets', [])
            for market in markets:
                if market['key'] == 'h2h':
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        odds_data[outcome['name']] = float(outcome['price'])
                elif market['key'] == 'totals':
                    # Ensure we are dealing with over/under 2.5 goals
                    if float(market.get('points', 0)) == 2.5:
                        outcomes = market.get('outcomes', [])
                        for outcome in outcomes:
                            # Some APIs label as 'Over'/'Under', others as 'Over 2.5'/'Under 2.5'
                            label = f"{outcome['name']} 2.5" if '2.5' not in outcome['name'] else outcome['name']
                            odds_data[label] = float(outcome['price'])

        fixtures.append({
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Odds_H': odds_data.get(home_team, None),
            'Odds_D': odds_data.get('Draw', None),
            'Odds_A': odds_data.get(away_team, None),
            'Odds_Over2.5': odds_data.get('Over 2.5', None),
            'Odds_Under2.5': odds_data.get('Under 2.5', None)
        })

    return fixtures

fixtures = get_upcoming_fixtures_with_odds(get_upcoming_fixtures_data())

# Remove fixtures without complete result odds data
fixtures = [
    fixture for fixture in fixtures
    if all([
        fixture['Odds_H'] is not None,
        fixture['Odds_D'] is not None,
        fixture['Odds_A'] is not None,
    ])
]

if not fixtures:
    print("No upcoming fixtures with complete result odds data found.")
    exit()

# =======================
# Step 6: Define Team Name Mapping
# =======================

# Map team names from Odds API to historical data if necessary
team_name_mapping = {
    'Newcastle United': 'Newcastle',
    'Manchester City': 'Man City',
    'Arsenal': 'Arsenal',
    'Leicester City': 'Leicester',
    'Brentford': 'Brentford',
    'West Ham United': 'West Ham',
    'Chelsea': 'Chelsea',
    'Brighton & Hove Albion': 'Brighton',
    'Brighton and Hove Albion': 'Brighton',
    'Everton': 'Everton',
    'Crystal Palace': 'Crystal Palace',
    'Nottingham Forest': "Nott'm Forest",
    'Fulham': 'Fulham',
    'Wolverhampton Wanderers': 'Wolves',
    'Liverpool': 'Liverpool',
    'Aston Villa': 'Aston Villa',
    'Manchester United': 'Man United',
    'Tottenham Hotspur': 'Tottenham',
    'Bournemouth': 'Bournemouth',
    'Southampton': 'Southampton',
    'Leeds United': 'Leeds',
    'West Bromwich Albion': 'West Brom',
    'Everton FC': 'Everton',
    'Ipswich Town': 'Ipswich',
    # Add any additional mappings as required for other teams.
}

# =======================
# Step 7: Predict Match Outcomes
# =======================

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

# Use the best parameters for confidence thresholds and max_goals
confidence_threshold = best_params_match_result['confidence_threshold']
confidence_multiplier = best_params_match_result['confidence_multiplier']
max_goals = best_params_match_result['max_goals']

# =======================
# Step 8: User Input for Risk Tolerance
# =======================

# Ask user for risk tolerance
while True:
    try:
        risk_tolerance = float(input("Enter your risk tolerance (0 for low risk, 1 for high risk): "))
        if 0 <= risk_tolerance <= 1:
            break
        else:
            print("Please enter a value between 0 and 1.")
    except ValueError:
        print("Invalid input. Please enter a numeric value between 0 and 1.")

# Current balance
current_balance = 1000.0  # Example starting balance in your currency

# Initialize variables to track total profit and bets
total_profit = 0
bets_made = []

# Define betting parameters
min_bet_percentage = 0.02  # Minimum 2% of balance
max_bet_percentage = 0.10  # Maximum 10% of balance

# Function to calculate implied probability from odds
def implied_probability(odds):
    return 1 / odds if odds > 0 else 0

# Function to suggest bet
def suggest_bet(pred_prob, implied_prob, odds, confidence_threshold, confidence_multiplier, risk_tolerance):
    """
    Determine if there's value in placing a bet based on risk tolerance.

    Returns:
        Boolean indicating whether to bet.
        Expected return value.
        Risk metric.
    """
    confidence = pred_prob - implied_prob
    expected_return = (odds * pred_prob) - 1
    risk_metric = 1 - pred_prob  # High predicted probability means lower risk

    # Decide whether to suggest the bet based on risk tolerance
    if confidence > 0 and risk_metric <= risk_tolerance:
        return True, expected_return, risk_metric
    else:
        return False, expected_return, risk_metric  # Always calculate expected return, even if not betting

# Function to calculate bet size based on confidence
def calculate_bet_size(balance, confidence, min_pct, max_pct):
    """
    Scale bet size based on confidence score.

    :param balance: Current balance.
    :param confidence: Confidence score (scaled between 0 and 1).
    :param min_pct: Minimum percentage of the balance to bet.
    :param max_pct: Maximum percentage of the balance to bet.
    :return: Bet size.
    """
    scaled_pct = min_pct + (max_pct - min_pct) * confidence
    return scaled_pct * balance

# =======================
# Step 9: Integrate Betting Suggestions
# =======================

# Initialize a list to store all probabilities
all_probabilities = []

for fixture in fixtures:
    home_team_full = fixture['HomeTeam']
    away_team_full = fixture['AwayTeam']
    home_team = team_name_mapping.get(home_team_full, home_team_full)
    away_team = team_name_mapping.get(away_team_full, away_team_full)

    # Check if teams are in the dataset
    if home_team not in attack_strength_home.index or away_team not in attack_strength_away.index:
        print(f"Data for {home_team} or {away_team} not available. Skipping this match.")
        continue

    # Predict goals
    home_lambda, away_lambda = predict_goals(home_team, away_team)

    # Simulate match probabilities using Poisson distribution
    home_goals_probs = [poisson.pmf(i, home_lambda) for i in range(0, max_goals + 1)]
    away_goals_probs = [poisson.pmf(i, away_lambda) for i in range(0, max_goals + 1)]

    # Calculate match outcome probabilities
    home_win_prob = sum(home_goals_probs[i] * sum(away_goals_probs[:i]) for i in range(1, max_goals + 1))
    draw_prob = sum(home_goals_probs[i] * away_goals_probs[i] for i in range(max_goals + 1))
    away_win_prob = sum(away_goals_probs[i] * sum(home_goals_probs[:i]) for i in range(1, max_goals + 1))

    # Normalize probabilities
    total_prob = home_win_prob + draw_prob + away_win_prob
    home_win_prob /= total_prob
    draw_prob /= total_prob
    away_win_prob /= total_prob

    # Calculate total goals lambda
    total_goals_lambda = home_lambda + away_lambda

    # Calculate probabilities for over and under 2.5 goals
    prob_under_2_5 = poisson.cdf(2, total_goals_lambda)
    prob_over_2_5 = 1 - prob_under_2_5

    # Calculate implied probabilities from odds
    implied_prob_H = implied_probability(fixture['Odds_H'])
    implied_prob_D = implied_probability(fixture['Odds_D'])
    implied_prob_A = implied_probability(fixture['Odds_A'])

    # Suggest bets and calculate expected returns for match result
    bet_H, exp_return_H, risk_metric_H = suggest_bet(
        home_win_prob, implied_prob_H, fixture['Odds_H'],
        confidence_threshold, confidence_multiplier, risk_tolerance
    )
    bet_D, exp_return_D, risk_metric_D = suggest_bet(
        draw_prob, implied_prob_D, fixture['Odds_D'],
        confidence_threshold, confidence_multiplier, risk_tolerance
    )
    bet_A, exp_return_A, risk_metric_A = suggest_bet(
        away_win_prob, implied_prob_A, fixture['Odds_A'],
        confidence_threshold, confidence_multiplier, risk_tolerance
    )

    # Initialize over/under variables
    bet_over = bet_under = False
    exp_return_over = exp_return_under = None
    implied_prob_over = implied_prob_under = None
    risk_metric_over = risk_metric_under = None

    # Check if over/under odds are available
    if fixture['Odds_Over2.5'] is not None and fixture['Odds_Under2.5'] is not None:
        implied_prob_over = implied_probability(fixture['Odds_Over2.5'])
        implied_prob_under = implied_probability(fixture['Odds_Under2.5'])

        # Suggest bets and calculate expected returns for over/under
        bet_over, exp_return_over, risk_metric_over = suggest_bet(
            prob_over_2_5, implied_prob_over, fixture['Odds_Over2.5'],
            confidence_threshold, confidence_multiplier, risk_tolerance
        )
        bet_under, exp_return_under, risk_metric_under = suggest_bet(
            prob_under_2_5, implied_prob_under, fixture['Odds_Under2.5'],
            confidence_threshold, confidence_multiplier, risk_tolerance
        )

    # Store all probabilities and expected returns
    all_probabilities.append({
        'HomeTeam': home_team_full,
        'AwayTeam': away_team_full,
        'HomeWinProb': home_win_prob,
        'DrawProb': draw_prob,
        'AwayWinProb': away_win_prob,
        'Odds_H': fixture['Odds_H'],
        'Odds_D': fixture['Odds_D'],
        'Odds_A': fixture['Odds_A'],
        'ImpliedProb_H': implied_prob_H,
        'ImpliedProb_D': implied_prob_D,
        'ImpliedProb_A': implied_prob_A,
        'ExpReturn_H': exp_return_H,
        'ExpReturn_D': exp_return_D,
        'ExpReturn_A': exp_return_A,
        'BetSuggested_H': bet_H,
        'BetSuggested_D': bet_D,
        'BetSuggested_A': bet_A,
        'RiskMetric_H': risk_metric_H,
        'RiskMetric_D': risk_metric_D,
        'RiskMetric_A': risk_metric_A,
        'Prob_Over_2_5': prob_over_2_5,
        'Prob_Under_2_5': prob_under_2_5,
        'Odds_Over2.5': fixture.get('Odds_Over2.5'),
        'Odds_Under2.5': fixture.get('Odds_Under2.5'),
        'ImpliedProb_Over2.5': implied_prob_over,
        'ImpliedProb_Under2.5': implied_prob_under,
        'ExpReturn_Over2.5': exp_return_over,
        'ExpReturn_Under2.5': exp_return_under,
        'BetSuggested_Over2.5': bet_over,
        'BetSuggested_Under2.5': bet_under,
        'RiskMetric_Over2.5': risk_metric_over,
        'RiskMetric_Under2.5': risk_metric_under
    })

# =======================
# Step 10: Display the Suggested Bets
# =======================

# Display all probabilities and suggested bets
probabilities_df = pd.DataFrame(all_probabilities)
print("\n=== Match Probabilities and Bet Suggestions ===\n")
for index, row in probabilities_df.iterrows():
    print(f"{row['HomeTeam']} vs {row['AwayTeam']}:")
    print(f"  Home Win Probability: {row['HomeWinProb']:.2%} | Odds: {row['Odds_H']} | Expected Return: {row['ExpReturn_H']:.2f} | Risk Metric: {row['RiskMetric_H']:.2f} | Bet Suggested: {'Yes' if row['BetSuggested_H'] else 'No'}")
    print(f"  Draw Probability: {row['DrawProb']:.2%} | Odds: {row['Odds_D']} | Expected Return: {row['ExpReturn_D']:.2f} | Risk Metric: {row['RiskMetric_D']:.2f} | Bet Suggested: {'Yes' if row['BetSuggested_D'] else 'No'}")
    print(f"  Away Win Probability: {row['AwayWinProb']:.2%} | Odds: {row['Odds_A']} | Expected Return: {row['ExpReturn_A']:.2f} | Risk Metric: {row['RiskMetric_A']:.2f} | Bet Suggested: {'Yes' if row['BetSuggested_A'] else 'No'}")

    # Include over/under probabilities and bets if odds are available
    if row['Odds_Over2.5'] is not None and row['Odds_Under2.5'] is not None:
        print(f"  Over 2.5 Goals Probability: {row['Prob_Over_2_5']:.2%} | Odds: {row['Odds_Over2.5']} | Expected Return: {row['ExpReturn_Over2.5']:.2f} | Risk Metric: {row['RiskMetric_Over2.5']:.2f} | Bet Suggested: {'Yes' if row['BetSuggested_Over2.5'] else 'No'}")
        print(f"  Under 2.5 Goals Probability: {row['Prob_Under_2_5']:.2%} | Odds: {row['Odds_Under2.5']} | Expected Return: {row['ExpReturn_Under2.5']:.2f} | Risk Metric: {row['RiskMetric_Under2.5']:.2f} | Bet Suggested: {'Yes' if row['BetSuggested_Under2.5'] else 'No'}")
    print("\n")

# Proceed with bet placement and profit calculation as before, only considering bets where odds are available
for index, row in probabilities_df.iterrows():
    # Define the outcomes to consider based on availability of odds
    outcomes_to_consider = [
        ('H', row['BetSuggested_H'], row['HomeWinProb'], row['Odds_H'], 'Home Win', row['RiskMetric_H'], row['ExpReturn_H']),
        ('D', row['BetSuggested_D'], row['DrawProb'], row['Odds_D'], 'Draw', row['RiskMetric_D'], row['ExpReturn_D']),
        ('A', row['BetSuggested_A'], row['AwayWinProb'], row['Odds_A'], 'Away Win', row['RiskMetric_A'], row['ExpReturn_A']),
    ]

    if row['Odds_Over2.5'] is not None and row['Odds_Under2.5'] is not None:
        outcomes_to_consider.extend([
            ('Over2.5', row['BetSuggested_Over2.5'], row['Prob_Over_2_5'], row['Odds_Over2.5'], 'Over 2.5 Goals', row['RiskMetric_Over2.5'], row['ExpReturn_Over2.5']),
            ('Under2.5', row['BetSuggested_Under2.5'], row['Prob_Under_2_5'], row['Odds_Under2.5'], 'Under 2.5 Goals', row['RiskMetric_Under2.5'], row['ExpReturn_Under2.5']),
        ])

    for outcome, suggested, pred_prob, odds, outcome_label, risk_metric, exp_return in outcomes_to_consider:
        if suggested:
            # Calculate confidence and bet size
            implied_prob = implied_probability(odds)
            confidence = pred_prob - implied_prob
            confidence = min(max(confidence, 0), 1)  # Ensure confidence is between 0 and 1

            bet_size = calculate_bet_size(current_balance, confidence, min_bet_percentage, max_bet_percentage)
            potential_profit = (odds - 1) * bet_size

            # Since we don't have actual results, we cannot calculate actual profit. In real use, you would update profit based on results.
            # For demonstration, we display potential profit.

            # Log the bet
            bets_made.append({
                'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                'Outcome': outcome_label,
                'Bet Size': round(bet_size, 2),
                'Odds': odds,
                'Risk Metric': round(risk_metric, 2),
                'Potential Profit': round(potential_profit, 2)
            })

# Display bet results
print("\n=== Suggested Bets ===\n")
for bet in bets_made:
    print(f"Match: {bet['Match']}")
    print(f"  Bet on: {bet['Outcome']} | Bet Size: £{bet['Bet Size']} | Odds: {bet['Odds']:.2f} | Risk Metric: {bet['Risk Metric']:.2f}")
    print(f"  Potential Profit: £{bet['Potential Profit']}\n")



def predict_and_recommend_bet(home_team, away_team, odds_over_under, odds_match_result, risk_tolerance=0.5):
    """
    Predicts the score between two teams and recommends a bet based on provided odds.

    :param home_team: Name of the home team.
    :param away_team: Name of the away team.
    :param odds_over_under: List or tuple with two odds [Odds_Over2.5, Odds_Under2.5].
    :param odds_match_result: List or tuple with three odds [Odds_HomeWin, Odds_Draw, Odds_AwayWin].
    :param risk_tolerance: Float between 0 and 1 indicating the user's risk tolerance (default is 0.5).
    :return: A dictionary with score prediction and recommended bets.
    """

    # Ensure the odds are provided correctly
    if len(odds_over_under) != 2 or len(odds_match_result) != 3:
        raise ValueError("Incorrect number of odds provided.")

    # Map team names if necessary (as per your team_name_mapping)
    team_name_mapping = {
        'Newcastle United': 'Newcastle',
        'Manchester City': 'Man City',
        'Arsenal': 'Arsenal',
        'Leicester City': 'Leicester',
        'Brentford': 'Brentford',
        'West Ham United': 'West Ham',
        'Chelsea': 'Chelsea',
        'Brighton & Hove Albion': 'Brighton',
        'Brighton and Hove Albion': 'Brighton',
        'Everton': 'Everton',
        'Crystal Palace': 'Crystal Palace',
        'Nottingham Forest': "Nott'm Forest",
        'Fulham': 'Fulham',
        'Wolverhampton Wanderers': 'Wolves',
        'Liverpool': 'Liverpool',
        'Aston Villa': 'Aston Villa',
        'Manchester United': 'Man United',
        'Tottenham Hotspur': 'Tottenham',
        'Bournemouth': 'Bournemouth',
        'Southampton': 'Southampton',
        'Leeds United': 'Leeds',
        'West Bromwich Albion': 'West Brom',
        'Everton FC': 'Everton',
        'Ipswich Town': 'Ipswich',
        # Add any additional mappings as required for other teams.
    }

    # Adjust team names if necessary
    home_team_adj = team_name_mapping.get(home_team, home_team)
    away_team_adj = team_name_mapping.get(away_team, away_team)

    # Check if teams are in the dataset
    if home_team_adj not in attack_strength_home.index or away_team_adj not in attack_strength_away.index:
        return {
            'ScorePrediction': None,
            'RecommendedBets': [],
            'Message': f"Data for {home_team} or {away_team} not available."
        }

    # Predict goals using your model
    home_lambda, away_lambda = predict_goals(home_team_adj, away_team_adj)

    # Expected goals (mean of Poisson distribution)
    expected_home_goals = home_lambda
    expected_away_goals = away_lambda

    # Generate score prediction (rounded to nearest integer)
    predicted_home_goals = round(expected_home_goals)
    predicted_away_goals = round(expected_away_goals)
    score_prediction = f"{home_team} {predicted_home_goals} - {predicted_away_goals} {away_team}"

    # Calculate probabilities for match outcome
    max_goals = 10  # Define a reasonable maximum number of goals
    home_goals_probs = [poisson.pmf(i, home_lambda) for i in range(0, max_goals + 1)]
    away_goals_probs = [poisson.pmf(i, away_lambda) for i in range(0, max_goals + 1)]

    # Calculate match outcome probabilities
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
    home_win_prob /= total_prob
    draw_prob /= total_prob
    away_win_prob /= total_prob

    # Calculate probabilities for over/under 2.5 goals
    total_goals_lambda = home_lambda + away_lambda
    prob_under_2_5 = poisson.cdf(2, total_goals_lambda)
    prob_over_2_5 = 1 - prob_under_2_5

    # Odds provided
    Odds_Over2_5, Odds_Under2_5 = odds_over_under
    Odds_H, Odds_D, Odds_A = odds_match_result

    # Function to calculate implied probability from odds
    def implied_probability(odds):
        return 1 / odds if odds > 0 else 0

    # Calculate implied probabilities from odds
    implied_prob_H = implied_probability(Odds_H)
    implied_prob_D = implied_probability(Odds_D)
    implied_prob_A = implied_probability(Odds_A)
    implied_prob_over = implied_probability(Odds_Over2_5)
    implied_prob_under = implied_probability(Odds_Under2_5)

    # Function to suggest bet
    def suggest_bet(pred_prob, implied_prob, odds, risk_tolerance):
        confidence = pred_prob - implied_prob
        expected_return = (odds * pred_prob) - 1
        risk_metric = 1 - pred_prob  # High predicted probability means lower risk

        if confidence > 0 and risk_metric <= risk_tolerance:
            return {
                'Bet': True,
                'Odds': odds,
                'PredictedProbability': pred_prob,
                'ImpliedProbability': implied_prob,
                'ExpectedReturn': expected_return,
                'RiskMetric': risk_metric,
            }
        else:
            return {'Bet': False}

    # Suggest bets
    bets = []

    # Match Result Bets
    bet_H = suggest_bet(home_win_prob, implied_prob_H, Odds_H, risk_tolerance)
    bet_D = suggest_bet(draw_prob, implied_prob_D, Odds_D, risk_tolerance)
    bet_A = suggest_bet(away_win_prob, implied_prob_A, Odds_A, risk_tolerance)

    # Over/Under Bets
    bet_over = suggest_bet(prob_over_2_5, implied_prob_over, Odds_Over2_5, risk_tolerance)
    bet_under = suggest_bet(prob_under_2_5, implied_prob_under, Odds_Under2_5, risk_tolerance)

    # Collect recommended bets
    if bet_H['Bet']:
        bets.append({
            'Market': 'Match Result',
            'Selection': f"{home_team} to Win",
            'Odds': Odds_H,
            'ExpectedReturn': bet_H['ExpectedReturn'],
            'RiskMetric': bet_H['RiskMetric'],
            'PredictedProbability': bet_H['PredictedProbability']
        })
    if bet_D['Bet']:
        bets.append({
            'Market': 'Match Result',
            'Selection': 'Draw',
            'Odds': Odds_D,
            'ExpectedReturn': bet_D['ExpectedReturn'],
            'RiskMetric': bet_D['RiskMetric'],
            'PredictedProbability': bet_D['PredictedProbability']
        })
    if bet_A['Bet']:
        bets.append({
            'Market': 'Match Result',
            'Selection': f"{away_team} to Win",
            'Odds': Odds_A,
            'ExpectedReturn': bet_A['ExpectedReturn'],
            'RiskMetric': bet_A['RiskMetric'],
            'PredictedProbability': bet_A['PredictedProbability']
        })
    if bet_over['Bet']:
        bets.append({
            'Market': 'Over/Under 2.5 Goals',
            'Selection': 'Over 2.5 Goals',
            'Odds': Odds_Over2_5,
            'ExpectedReturn': bet_over['ExpectedReturn'],
            'RiskMetric': bet_over['RiskMetric'],
            'PredictedProbability': bet_over['PredictedProbability']
        })
    if bet_under['Bet']:
        bets.append({
            'Market': 'Over/Under 2.5 Goals',
            'Selection': 'Under 2.5 Goals',
            'Odds': Odds_Under2_5,
            'ExpectedReturn': bet_under['ExpectedReturn'],
            'RiskMetric': bet_under['RiskMetric'],
            'PredictedProbability': bet_under['PredictedProbability']
        })

    return {
        'ScorePrediction': score_prediction,
        'RecommendedBets': bets,
        'Message': 'Prediction and recommendations generated successfully.'
    }
