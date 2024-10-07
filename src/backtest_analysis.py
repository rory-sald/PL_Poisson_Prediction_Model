import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from backtest_and_visualise import backtest_df


# Filter out incorrect predictions for both match results and over/under predictions
incorrect_results = backtest_df[(backtest_df['PredictedResult'].notnull()) & (backtest_df['ActualResult'] != backtest_df['PredictedResult'])]
incorrect_over_under = backtest_df[(backtest_df['BetOverUnder'].notnull()) &
                                   ((backtest_df['BetOverUnder'] == 'Over') & (backtest_df['ActualHomeGoals'] + backtest_df['ActualAwayGoals'] <= 2.5)) |
                                   ((backtest_df['BetOverUnder'] == 'Under') & (backtest_df['ActualHomeGoals'] + backtest_df['ActualAwayGoals'] > 2.5))]

# Analysis 1: Most common teams involved in incorrect match result predictions
incorrect_result_teams = pd.concat([incorrect_results['HomeTeam'], incorrect_results['AwayTeam']]).value_counts()
print("Teams frequently involved in incorrect match result predictions:\n", incorrect_result_teams.head(10))

# Analysis 2: Incorrect predictions by match location (home vs away)
incorrect_by_location = incorrect_results.groupby('HomeTeam')['PredictedResult'].count().reset_index()
incorrect_by_location.columns = ['Team', 'Incorrect Predictions']
sns.barplot(data=incorrect_by_location, x='Team', y='Incorrect Predictions')
plt.title('Incorrect Predictions by Home Teams')
plt.xticks(rotation=90)
plt.show()

# Analysis 3: Incorrect predictions over time (e.g., by matchday or month)
incorrect_by_date = incorrect_results['Date'].dt.to_period('M').value_counts().sort_index()
incorrect_by_date.plot(kind='bar', title='Incorrect Predictions by Month')
plt.xlabel('Month')
plt.ylabel('Number of Incorrect Predictions')
plt.show()

# Analysis 4: Themes in over/under incorrect predictions
over_under_error_analysis = incorrect_over_under.groupby(['BetOverUnder']).size().reset_index(name='Count')
sns.barplot(data=over_under_error_analysis, x='BetOverUnder', y='Count')
plt.title('Incorrect Over/Under Predictions')
plt.show()

# Additional Detailed Analysis: Examine odds ranges where predictions fail
incorrect_odds_analysis = incorrect_results[['B365H', 'B365D', 'B365A']].melt(var_name='BetType', value_name='Odds')
sns.boxplot(data=incorrect_odds_analysis, x='BetType', y='Odds')
plt.title('Odds Distribution in Incorrect Predictions')
plt.show()

# Summary statistics of incorrect odds
print(incorrect_odds_analysis.groupby('BetType').describe())
