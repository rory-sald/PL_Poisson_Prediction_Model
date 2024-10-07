from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from backtest_module import run_backtest_match_result, run_backtest_over_under, data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

target_bets = 40  # Increased from 20 to 40

# =======================
# Optimize Match Result Betting Strategy
# =======================

def optimize_match_result(params):
    decay_factor = params['decay_factor']
    confidence_threshold = params['confidence_threshold']
    confidence_multiplier = params['confidence_multiplier']
    max_goals = int(params['max_goals'])

    performance = run_backtest_match_result(
        data=data,
        decay_factor=decay_factor,
        confidence_threshold=confidence_threshold,
        confidence_multiplier=confidence_multiplier,
        max_goals=max_goals
    )


    # We aim to maximize accuracy and ROI; adjust loss accordingly
    # Encourage more bets by penalizing loss when bets are below desired_max_bets
    # Combined loss function balancing accuracy and ROI

    # loss = (
    #     -performance['accuracy_result']
    #     + 0.02 * (100 - performance['roi_result'])
    #     + 0.07 * abs(target_bets - performance['bets_placed_result'])  # Penalizes deviation from target number of bets
    # )
    
    # Define the weight for accuracy and a penalty for too few bets
    accuracy_weight = 1.0
    bet_penalty_weight = 0.1

    # Penalty based on the number of bets placed
    bet_count = performance['bets_placed_result']
    bet_penalty = bet_penalty_weight * max(0, 10 - bet_count)  # Encourages at least 10 bets

    # Loss function combining accuracy emphasis with bet count penalty
    loss = -(accuracy_weight * performance['accuracy_result']) + bet_penalty


    return {'loss': loss, 'status': STATUS_OK}

# Define the search space with slightly relaxed thresholds
space_match_result = {
    'decay_factor': hp.uniform('decay_factor', 0.95, 1.0),
    'confidence_threshold': hp.uniform('confidence_threshold', 0.10, 0.9),
    'confidence_multiplier': hp.uniform('confidence_multiplier', 1.00, 1.5),
    'max_goals': hp.quniform('max_goals', 5, 7, 1)
}

# Perform Bayesian Optimization
trials_match_result = Trials()
best_match_result = fmin(
    fn=optimize_match_result,
    space=space_match_result,
    algo=tpe.suggest,
    max_evals=150,
    trials=trials_match_result
)

print("Best parameters for Match Result Betting Strategy:")
print(best_match_result)

# =======================
# Optimize Over/Under Betting Strategy
# =======================

def optimize_over_under(params):
    decay_factor = params['decay_factor']
    confidence_threshold_o_u = params['confidence_threshold_o_u']
    confidence_multiplier = params['confidence_multiplier']
    max_goals = int(params['max_goals'])

    performance = run_backtest_over_under(
        data=data,
        decay_factor=decay_factor,
        confidence_threshold_o_u=confidence_threshold_o_u,
        confidence_multiplier=confidence_multiplier,
        max_goals=max_goals
    )


    # We aim to maximize accuracy and ROI; adjust loss accordingly
    # Encourage more bets by penalizing loss when bets are below desired_max_bets
    # Combined loss function balancing accuracy and ROI
    # loss = (
    #     -performance['accuracy_over_under']
    #     + 0.05 * (100 - performance['roi_over_under'])
    #     + 0.07 * abs(target_bets - performance['bets_placed_over_under'])  # Penalizes deviation from target number of bets
    # )

    # Define the weight for accuracy and a penalty for too few bets
    accuracy_weight = 1.0
    bet_penalty_weight = 0.1

    # Penalty based on the number of bets placed
    bet_count = performance['bets_placed_over_under']
    bet_penalty = bet_penalty_weight * max(0, 10 - bet_count)  # Encourages at least 10 bets

    # Loss function combining accuracy emphasis with bet count penalty
    loss = -(accuracy_weight * performance['accuracy_over_under']) + bet_penalty

    return {'loss': loss, 'status': STATUS_OK}

# Define the search space with slightly relaxed thresholds
space_over_under = {
    'decay_factor': hp.uniform('decay_factor', 0.95, 1.0),
    'confidence_threshold_o_u': hp.uniform('confidence_threshold_o_u', 0.20, 0.9),
    'confidence_multiplier': hp.uniform('confidence_multiplier', 1.05, 1.5),
    'max_goals': hp.quniform('max_goals', 5, 7, 1)
}

# Perform Bayesian Optimization
trials_over_under = Trials()
best_over_under = fmin(
    fn=optimize_over_under,
    space=space_over_under,
    algo=tpe.suggest,
    max_evals=150,
    trials=trials_over_under
)

print("\nBest parameters for Over/Under Betting Strategy:")
print(best_over_under)

# Now run backtests with the best parameters
# (Ensure you convert parameter values to the correct types if needed)
best_params_match_result = {
    'decay_factor': float(best_match_result['decay_factor']),
    'confidence_threshold': float(best_match_result['confidence_threshold']),
    'confidence_multiplier': float(best_match_result['confidence_multiplier']),
    'max_goals': int(best_match_result['max_goals'])
}

best_params_over_under = {
    'decay_factor': float(best_over_under['decay_factor']),
    'confidence_threshold_o_u': float(best_over_under['confidence_threshold_o_u']),
    'confidence_multiplier': float(best_over_under['confidence_multiplier']),
    'max_goals': int(best_over_under['max_goals'])
}

# Run the backtest for Match Result Betting Strategy
match_result_performance = run_backtest_match_result(
    data=data,
    decay_factor=best_params_match_result['decay_factor'],
    confidence_threshold=best_params_match_result['confidence_threshold'],
    confidence_multiplier=best_params_match_result['confidence_multiplier'],
    max_goals=best_params_match_result['max_goals']
)

# Extract the bet data DataFrame
match_result_bets = match_result_performance['bet_data']

# Run the backtest for Over/Under Betting Strategy
over_under_performance = run_backtest_over_under(
    data=data,
    decay_factor=best_params_over_under['decay_factor'],
    confidence_threshold_o_u=best_params_over_under['confidence_threshold_o_u'],
    confidence_multiplier=best_params_over_under['confidence_multiplier'],
    max_goals=best_params_over_under['max_goals']
)

# Extract the bet data DataFrame
over_under_bets = over_under_performance['bet_data']

print("Match Result Bets Placed:")
print(match_result_bets.head())

print("\nOver/Under Bets Placed:")
print(over_under_bets.head())

# Count of each bet type
bet_type_counts = match_result_bets['PredictedResult'].value_counts()
print("Distribution of Match Result Bets:")
print(bet_type_counts)

# Count of over/under bets
bet_type_counts_o_u = over_under_bets['BetOverUnder'].value_counts()
print("Distribution of Over/Under Bets:")
print(bet_type_counts_o_u)

# Group by predicted result and calculate total profit and accuracy
performance_by_bet_type = match_result_bets.groupby('PredictedResult').agg(
    TotalProfit=('ProfitResult', 'sum'),
    MeanExpectedReturn=('ExpectedReturnResult', 'mean'),
    Accuracy=('PredictedResult', lambda x: (match_result_bets.loc[x.index, 'ActualResult'] == x).sum() / len(x))
)

print("Performance by Bet Type (Match Result):")
print(performance_by_bet_type)

# Group by over/under bet and calculate total profit and accuracy
performance_by_bet_type_o_u = over_under_bets.groupby('BetOverUnder').agg(
    TotalProfitOverUnder=('ProfitOverUnder', 'sum'),
    MeanExpectedReturnOverUnder=('ExpectedReturnOverUnder', 'mean'),
    Accuracy=('BetOverUnder', lambda x: (over_under_bets.loc[x.index, 'ActualOverUnder'] == x).sum() / len(x))
)

print("Performance by Bet Type (Over/Under):")
print(performance_by_bet_type_o_u)

# For Match Result Betting: Analyze Performance by Home Team
team_bets = match_result_bets.groupby('HomeTeam').agg(
    TotalProfit=('ProfitResult', 'sum'),
    BetsPlaced=('PredictedResult', 'count'),
    Accuracy=('PredictedResult', lambda x: (match_result_bets.loc[x.index, 'ActualResult'] == x).sum() / len(x))
)

print("Performance by Home Team (Match Result):")
print(team_bets.sort_values('BetsPlaced', ascending=False).head())

# Visualizations

# Histogram of profits for match result bets
plt.figure(figsize=(10,6))
plt.hist(match_result_bets['ProfitResult'], bins=20, edgecolor='k')
plt.title('Distribution of Profits for Match Result Bets')
plt.xlabel('Profit')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

def plot_cumulative_profit(match_result_bets, over_under_bets):
    # Calculate cumulative profit for Match Result Bets
    match_result_bets['CumulativeProfitResult'] = match_result_bets['ProfitResult'].cumsum() * 10  # Multiply by £10 bet size
    over_under_bets['CumulativeProfitOverUnder'] = over_under_bets['ProfitOverUnder'].cumsum() * 10  # Multiply by £10 bet size

    # Plot Cumulative Profit for Match Result Bets
    plt.figure(figsize=(12, 6))
    plt.plot(match_result_bets['Date'], match_result_bets['CumulativeProfitResult'], marker='o', color='b', label='Match Result Bets')
    plt.title('Cumulative Profit Over Time (Match Result Bets)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit (£)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Cumulative Profit for Over/Under Bets
    plt.figure(figsize=(12, 6))
    plt.plot(over_under_bets['Date'], over_under_bets['CumulativeProfitOverUnder'], marker='o', color='g', label='Over/Under Bets')
    plt.title('Cumulative Profit Over Time (Over/Under Bets)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit (£)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call the function to plot cumulative profit for both strategies
plot_cumulative_profit(match_result_bets, over_under_bets)
