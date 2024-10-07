# optimize_and_save_results.py
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from backtest_module import run_backtest_match_result, run_backtest_over_under, load_data
import json

data = load_data()

# Define target number of bets
target_bets = 40
def calculate_loss_result(performance, alpha=0.7, beta=0.05, gamma=0.05, 
                  bet_penalty_weight=0.02, min_bet_count=40):
    """
    Calculate the loss to optimize both accuracy and profitability.

    Parameters:
    - performance: dict containing performance metrics.
    - alpha: weight for accuracy.
    - beta: weight for ROI.
    - gamma: weight for total profit (optional, can be set to 0 if focusing on ROI).
    - bet_penalty_weight: weight for the bet count penalty.
    - min_bet_count: minimum number of bets to avoid penalty.

    Returns:
    - loss: float, the calculated loss value.
    """

    # Extract performance metrics
    accuracy = performance.get('accuracy_result', 0)
    roi = performance.get('roi_result', 0)
    total_profit = performance.get('profit_result', 0)
    bet_count = performance.get('bets_placed_result', 0)

    # Calculate bet penalty
    bet_penalty = bet_penalty_weight * max(0, min_bet_count - bet_count)

    # Calculate loss
    loss = - (alpha * accuracy + beta * roi + gamma * total_profit) + bet_penalty

    return loss

def calculate_loss_over_under(performance, alpha=0.7, beta=0.05, gamma=0.05, 
                  bet_penalty_weight=0.02, min_bet_count=40):
    """
    Calculate the loss to optimize both accuracy and profitability.

    Parameters:
    - performance: dict containing performance metrics.
    - alpha: weight for accuracy.
    - beta: weight for ROI.
    - gamma: weight for total profit (optional, can be set to 0 if focusing on ROI).
    - bet_penalty_weight: weight for the bet count penalty.
    - min_bet_count: minimum number of bets to avoid penalty.

    Returns:
    - loss: float, the calculated loss value.
    """

    # Extract performance metrics
    accuracy = performance.get('accuracy_over_under', 0)
    roi = performance.get('roi_over_under', 0)
    total_profit = performance.get('profit_over_under', 0)
    bet_count = performance.get('bets_placed_over_under', 0)

    # Calculate bet penalty
    bet_penalty = bet_penalty_weight * max(0, min_bet_count - bet_count)

    # Calculate loss
    loss = - (alpha * accuracy + beta * roi + gamma * total_profit) + bet_penalty

    return loss


# Define optimization functions
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

    accuracy_weight = 1.0
    bet_penalty_weight = 0.2
    bet_count = performance['bets_placed_result']
    bet_penalty = bet_penalty_weight * max(0, 50 - bet_count)

    loss = calculate_loss_result(performance)

    return {'loss': loss, 'status': STATUS_OK}


# Define the search space
space_match_result = {
    'decay_factor': hp.uniform('decay_factor', 0.95, 1.0),
    'confidence_threshold': hp.uniform('confidence_threshold', 0.10, 0.9),
    'confidence_multiplier': hp.uniform('confidence_multiplier', 1.00, 1.5),
    'max_goals': hp.quniform('max_goals', 5, 7, 1)
}

# Perform Bayesian Optimization for Match Result Strategy
trials_match_result = Trials()
best_match_result = fmin(
    fn=optimize_match_result,
    space=space_match_result,
    algo=tpe.suggest,
    max_evals=40,
    trials=trials_match_result
)

# Define optimization function for Over/Under strategy
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

    accuracy_weight = 1.0
    bet_penalty_weight = 0.2
    bet_count = performance['bets_placed_over_under']
    bet_penalty = bet_penalty_weight * max(0, 50 - bet_count)

    loss = calculate_loss_over_under(performance)

    return {'loss': loss, 'status': STATUS_OK}

# Define the search space for Over/Under strategy
space_over_under = {
    'decay_factor': hp.uniform('decay_factor', 0.95, 1.0),
    'confidence_threshold_o_u': hp.uniform('confidence_threshold_o_u', 0.20, 0.9),
    'confidence_multiplier': hp.uniform('confidence_multiplier', 1.05, 1.5),
    'max_goals': hp.quniform('max_goals', 5, 7, 1)
}

# Perform Bayesian Optimization for Over/Under Strategy
trials_over_under = Trials()
best_over_under = fmin(
    fn=optimize_over_under,
    space=space_over_under,
    algo=tpe.suggest,
    max_evals=40,
    trials=trials_over_under
)

# Save the best parameters to a JSON file
best_results = {
    'match_result': best_match_result,
    'over_under': best_over_under
}

with open('best_optimization_results.json', 'w') as f:
    json.dump(best_results, f)

print("Best parameters saved to best_optimization_results.json")
# Run backtests using the best parameters

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