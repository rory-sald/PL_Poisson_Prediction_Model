# visualize_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from backtest_module import run_backtest_match_result, run_backtest_over_under, load_data

data = load_data()


# Load the saved best optimization results
with open('best_optimization_results.json', 'r') as f:
    best_results = json.load(f)

# Extract best parameters for both strategies
best_params_match_result = {
    'decay_factor': float(best_results['match_result']['decay_factor']),
    'confidence_threshold': float(best_results['match_result']['confidence_threshold']),
    'confidence_multiplier': float(best_results['match_result']['confidence_multiplier']),
    'max_goals': int(best_results['match_result']['max_goals'])
}

best_params_over_under = {
    'decay_factor': float(best_results['over_under']['decay_factor']),
    'confidence_threshold_o_u': float(best_results['over_under']['confidence_threshold_o_u']),
    'confidence_multiplier': float(best_results['over_under']['confidence_multiplier']),
    'max_goals': int(best_results['over_under']['max_goals'])
}

# Run backtests using the best parameters
match_result_performance = run_backtest_match_result(
    data=data,
    decay_factor=best_params_match_result['decay_factor'],
    confidence_threshold=best_params_match_result['confidence_threshold'],
    confidence_multiplier=best_params_match_result['confidence_multiplier'],
    max_goals=best_params_match_result['max_goals']
)

over_under_performance = run_backtest_over_under(
    data=data,
    decay_factor=best_params_over_under['decay_factor'],
    confidence_threshold_o_u=best_params_over_under['confidence_threshold_o_u'],
    confidence_multiplier=best_params_over_under['confidence_multiplier'],
    max_goals=best_params_over_under['max_goals']
)

# Extract bet data for visualization
match_result_bets = match_result_performance['bet_data']
over_under_bets = over_under_performance['bet_data']

# Initialize starting balance and scaling parameters
starting_balance = 100.0
min_bet_percentage = 0.02  # Minimum bet is 1% of the balance
max_bet_percentage = 0.40  # Maximum bet is 5% of the balance

def scale_bet_size(confidence, balance, min_pct, max_pct):
    """
    Scale bet size based on confidence score.
    
    :param confidence: Confidence score (scaled to a value between 0 and 1).
    :param balance: Current balance.
    :param min_pct: Minimum percentage of the balance to bet.
    :param max_pct: Maximum percentage of the balance to bet.
    :return: Bet size.
    """
    scaled_bet_pct = min_pct + (max_pct - min_pct) * confidence
    return scaled_bet_pct * balance

def calculate_scaled_profits_result(bet_data, balance, min_pct, max_pct):
    """
    Calculate scaled profits for a series of bets.

    :param bet_data: DataFrame with bet data including 'ExpectedReturnResult' and 'ProfitResult'.
    :param balance: Starting balance.
    :param min_pct: Minimum percentage of the balance to bet.
    :param max_pct: Maximum percentage of the balance to bet.
    :return: Updated DataFrame with scaled profits and cumulative balance.
    """
    balance_list = [balance]
    scaled_profits = []
    bet_size_list = []

    for index, row in bet_data.iterrows():
        confidence = abs(row['ExpectedReturnResult'])  # Use expected return as a proxy for confidence
        confidence = min(confidence, 1.0)  # Ensure confidence is between 0 and 1

        bet_size = scale_bet_size(confidence, balance, min_pct, max_pct)
        profit = bet_size * row['ProfitResult']  # Scale profit based on bet size
        balance += profit

        scaled_profits.append(profit)
        balance_list.append(balance)
        bet_size_list.append(bet_size)

    bet_data['BetSize'] = bet_size_list
    bet_data['ScaledProfit'] = scaled_profits
    bet_data['CumulativeBalance'] = balance_list[:-1]

    return bet_data

def calculate_scaled_profits_over_under(bet_data, balance, min_pct, max_pct):
    """
    Calculate scaled profits for a series of bets.

    :param bet_data: DataFrame with bet data including 'ExpectedReturnResult' and 'ProfitResult'.
    :param balance: Starting balance.
    :param min_pct: Minimum percentage of the balance to bet.
    :param max_pct: Maximum percentage of the balance to bet.
    :return: Updated DataFrame with scaled profits and cumulative balance.
    """
    balance_list = [balance]
    scaled_profits = []
    bet_size_list = []

    for index, row in bet_data.iterrows():
        confidence = abs(row['ExpectedReturnOverUnder'])  # Use expected return as a proxy for confidence
        confidence = min(confidence, 1.0)  # Ensure confidence is between 0 and 1

        bet_size = scale_bet_size(confidence, balance, min_pct, max_pct)
        profit = bet_size * row['ProfitOverUnder']  # Scale profit based on bet size
        balance += profit

        scaled_profits.append(profit)
        balance_list.append(balance)
        bet_size_list.append(bet_size)
        
    bet_data['BetSize'] = bet_size_list
    bet_data['ScaledProfit'] = scaled_profits
    bet_data['CumulativeBalance'] = balance_list[:-1]

    return bet_data

# Calculate scaled profits for both strategies
match_result_bets = calculate_scaled_profits_result(match_result_bets, starting_balance, min_bet_percentage, max_bet_percentage)
over_under_bets = calculate_scaled_profits_over_under(over_under_bets, starting_balance, 0.1, 0.4)

# Function to plot cumulative profit and balance
def plot_cumulative_profit(match_result_bets, over_under_bets):
    match_result_bets['CumulativeProfitResult'] = match_result_bets['ScaledProfit'].cumsum()
    over_under_bets['CumulativeProfitOverUnder'] = over_under_bets['ScaledProfit'].cumsum()

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


# Plot the cumulative profit and balance for both strategies
plot_cumulative_profit(match_result_bets, over_under_bets)
