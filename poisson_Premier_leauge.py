import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

# Step 1: Load Historical Data
# Reading data directly from football-data.co.uk for the Premier League 2023/2024 season


# Load current season data
current_season_url = '/Users/rems/Documents/Python/Poisson_Goals_Model/E0.csv'
current_season_data = pd.read_csv(current_season_url)

# Load last season data
last_season_url = '/Users/rems/Documents/Python/Poisson_Goals_Model/E1.csv'
last_season_data = pd.read_csv(last_season_url)

print("Current Season Columns:", current_season_data.columns)
print("Last Season Columns:", last_season_data.columns)

# Combine both seasons' data into a single DataFrame
data = pd.concat([current_season_data, last_season_data], ignore_index=True)

# Assuming your combined data already has a 'Date' column
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Utilize more match statistics to enhance the model
# Relevant columns based on the provided notes
columns_needed = [
    'Date','HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR'
]

# Keep only necessary columns
data = data[columns_needed]

# Calculate the number of days since each game occurred
data['Days_Ago'] = (pd.to_datetime('today') - data['Date']).dt.days

# Decay factor (adjust this value as needed; lower values give more weight to recent games)
decay_factor = 0.98

# Calculate weights using exponential decay
data['Weight'] = decay_factor ** data['Days_Ago']


# Prepare the data in the required format
home_data = pd.DataFrame()
home_data['Team'] = data['HomeTeam']
home_data['Opponent'] = data['AwayTeam']
home_data['Goals_Scored'] = data['FTHG']
home_data['Goals_Conceded'] = data['FTAG']
home_data['Home_Away'] = 'Home'
home_data['Shots'] = data['HS']
home_data['Shots_On_Target'] = data['HST']
home_data['Corners'] = data['HC']
home_data['Fouls'] = data['HF']
home_data['Yellow_Cards'] = data['HY']
home_data['Red_Cards'] = data['HR']
home_data['Weight'] = data['Weight']  # Include the Weight column


away_data = pd.DataFrame()
away_data['Team'] = data['AwayTeam']
away_data['Opponent'] = data['HomeTeam']
away_data['Goals_Scored'] = data['FTAG']
away_data['Goals_Conceded'] = data['FTHG']
away_data['Home_Away'] = 'Away'
away_data['Shots'] = data['AS']
away_data['Shots_On_Target'] = data['AST']
away_data['Corners'] = data['AC']
away_data['Fouls'] = data['AF']
away_data['Yellow_Cards'] = data['AY']
away_data['Red_Cards'] = data['AR']
away_data['Weight'] = data['Weight']  # Include the Weight column


full_data = pd.concat([home_data, away_data], ignore_index=True)

# Step 2: Calculate average goals and additional statistics
# Filter data for home and away games
home_games = full_data[full_data['Home_Away'] == 'Home']
away_games = full_data[full_data['Home_Away'] == 'Away']

# Calculate weighted team statistics for home games
home_team_stats = home_games.groupby('Team').apply(
    lambda x: pd.Series({
        'Goals_Scored': np.average(x['Goals_Scored'], weights=x['Weight']),
        'Goals_Conceded': np.average(x['Goals_Conceded'], weights=x['Weight']),
        'Shots': np.average(x['Shots'], weights=x['Weight']),
        'Shots_On_Target': np.average(x['Shots_On_Target'], weights=x['Weight']),
        'Corners': np.average(x['Corners'], weights=x['Weight']),
        'Fouls': np.average(x['Fouls'], weights=x['Weight']),
        'Yellow_Cards': np.average(x['Yellow_Cards'], weights=x['Weight']),
        'Red_Cards': np.average(x['Red_Cards'], weights=x['Weight'])
    })
)

# Calculate weighted team statistics for away games
away_team_stats = away_games.groupby('Team').apply(
    lambda x: pd.Series({
        'Goals_Scored': np.average(x['Goals_Scored'], weights=x['Weight']),
        'Goals_Conceded': np.average(x['Goals_Conceded'], weights=x['Weight']),
        'Shots': np.average(x['Shots'], weights=x['Weight']),
        'Shots_On_Target': np.average(x['Shots_On_Target'], weights=x['Weight']),
        'Corners': np.average(x['Corners'], weights=x['Weight']),
        'Fouls': np.average(x['Fouls'], weights=x['Weight']),
        'Yellow_Cards': np.average(x['Yellow_Cards'], weights=x['Weight']),
        'Red_Cards': np.average(x['Red_Cards'], weights=x['Weight'])
    })
)

# Calculate league averages
league_home_stats = home_games[['Goals_Scored', 'Goals_Conceded', 'Shots', 'Shots_On_Target', 'Corners', 'Fouls', 'Yellow_Cards', 'Red_Cards']].mean()
league_away_stats = away_games[['Goals_Scored', 'Goals_Conceded', 'Shots', 'Shots_On_Target', 'Corners', 'Fouls', 'Yellow_Cards', 'Red_Cards']].mean()

# Step 3: Calculate attack and defense strengths using additional statistics
# For simplicity, we'll focus on goals, but you can expand the model to include other statistics
attack_strength_home = home_team_stats['Goals_Scored'] / league_home_stats['Goals_Scored']
defense_strength_home = home_team_stats['Goals_Conceded'] / league_away_stats['Goals_Scored']
attack_strength_away = away_team_stats['Goals_Scored'] / league_away_stats['Goals_Scored']
defense_strength_away = away_team_stats['Goals_Conceded'] / league_home_stats['Goals_Scored']

# Step 4: Fetch Upcoming Fixtures using football-data.org API
API_KEY = 'd2250335bc284783a274e4c611a2e485'  # Replace with your football-data.org API key
BASE_URL = 'https://api.football-data.org/v4'
headers = {'X-Auth-Token': API_KEY}

def get_upcoming_fixtures():
    url = f'{BASE_URL}/competitions/PL/matches'
    params = {'status': 'SCHEDULED'}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    matches = data['matches']
    fixtures = []
    for match in matches:
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        matchday = match['matchday']
        fixtures.append({'HomeTeam': home_team, 'AwayTeam': away_team, 'Matchday': matchday})
    return fixtures

fixtures = get_upcoming_fixtures()

# Get fixtures for the next gameweek
next_matchday = min([fixture['Matchday'] for fixture in fixtures])
next_fixtures = [fixture for fixture in fixtures if fixture['Matchday'] == next_matchday]

# Map team names from fixtures to those in historical data
team_name_mapping = {
    'Newcastle United FC': 'Newcastle',
    'Manchester City FC': 'Man City',
    'Arsenal FC': 'Arsenal',
    'Leicester City FC': 'Leicester',
    'Brentford FC': 'Brentford',
    'West Ham United FC': 'West Ham',
    'Chelsea FC': 'Chelsea',
    'Brighton & Hove Albion FC': 'Brighton',
    'Everton FC': 'Everton',
    'Crystal Palace FC': 'Crystal Palace',
    'Nottingham Forest FC': "Nott'm Forest",
    'Fulham FC': 'Fulham',
    'Wolverhampton Wanderers FC': 'Wolves',
    'Liverpool FC': 'Liverpool',
    'Ipswich Town FC': 'Ipswich',  # Ipswich may need special handling as a newly promoted team.
    'Aston Villa FC': 'Aston Villa',
    'Manchester United FC': 'Man United',
    'Tottenham Hotspur FC': 'Tottenham',
    'AFC Bournemouth': 'Bournemouth',
    'Southampton FC': 'Southampton',
    # Add any additional mappings as required for other teams.
}

# Step 5: Function to predict goals using Poisson distribution
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

# Step 6: Predict match outcomes for the next gameweek
predictions = []

for fixture in next_fixtures:
    home_team_full = fixture['HomeTeam']
    away_team_full = fixture['AwayTeam']
    home_team = team_name_mapping.get(home_team_full, home_team_full)
    away_team = team_name_mapping.get(away_team_full, away_team_full)

    # Check if teams are in the dataset
    if home_team not in attack_strength_home.index or away_team not in attack_strength_away.index:
        print(f"Data for {home_team} or {away_team} not available. Skipping this match.")
        continue

    home_lambda, away_lambda = predict_goals(home_team, away_team)

    # Simulate match probabilities
    max_goals = 5  # Maximum number of goals to consider
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

    predictions.append({
        'HomeTeam': home_team_full,
        'AwayTeam': away_team_full,
        'HomeGoalsExpected': home_lambda,
        'AwayGoalsExpected': away_lambda,
        'HomeWinProb': home_win_prob,
        'DrawProb': draw_prob,
        'AwayWinProb': away_win_prob
    })

# Step 7: Display the predictions
predictions_df = pd.DataFrame(predictions)

for index, row in predictions_df.iterrows():
    print(f"{row['HomeTeam']} vs {row['AwayTeam']}:")
    print(f"  Expected Goals for {row['HomeTeam']}: {row['HomeGoalsExpected']:.2f}")
    print(f"  Expected Goals for {row['AwayTeam']}: {row['AwayGoalsExpected']:.2f}")
    print(f"  Probability of {row['HomeTeam']} winning: {row['HomeWinProb']:.2%}")
    print(f"  Probability of a draw: {row['DrawProb']:.2%}")
    print(f"  Probability of {row['AwayTeam']} winning: {row['AwayWinProb']:.2%}\n")
