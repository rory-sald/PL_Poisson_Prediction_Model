# parameter_tuning.py

import pandas as pd
from itertools import product
from backtest_module import run_backtest, data, run_backtest_match_result, run_backtest_over_under  # Import the function and data

def execute_backtest(
    data,
    decay_factor=0.99,
    confidence_threshold=0.4,
    confidence_threshold_o_u=0.55,
    confidence_multiplier=1.1,
    max_goals=5,
    backtest_matchdays=20
):
    """
    Runs the backtest with the specified parameters.

    Parameters:
    - data: pandas DataFrame containing the match data.
    - decay_factor: float, weighting factor for historical data.
    - confidence_threshold: float, minimum predicted probability to place a bet on match results.
    - confidence_threshold_o_u: float, minimum predicted probability for over/under bets.
    - confidence_multiplier: float, multiplier for the implied probability to determine value bets.
    - max_goals: int, maximum number of goals considered in Poisson distribution calculations.
    - backtest_matchdays: int, number of unique matchdays to include in the backtest.

    Returns:
    - Dictionary containing performance metrics of the backtest for both strategies.
    """
    # Run backtest for match result betting strategy
    match_result_metrics = run_backtest_match_result(
        data=data,
        decay_factor=decay_factor,
        confidence_threshold=confidence_threshold,
        confidence_multiplier=confidence_multiplier,
        max_goals=max_goals
    )

    # Run backtest for over/under betting strategy
    over_under_metrics = run_backtest_over_under(
        data=data,
        decay_factor=decay_factor,
        confidence_threshold_o_u=confidence_threshold_o_u,
        confidence_multiplier=confidence_multiplier,
        max_goals=max_goals
    )

    # Combine the results into a single dictionary for easy reference
    combined_metrics = {
        'match_result': match_result_metrics,
        'over_under': over_under_metrics
    }

    return combined_metrics

# Example usage:
results = execute_backtest(data, decay_factor=0.98, confidence_threshold=0.3, confidence_threshold_o_u=0.55, confidence_multiplier=1.1, max_goals=7)
print(results)
