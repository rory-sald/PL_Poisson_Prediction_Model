Project Overview
This project is designed to predict the number of goals in football matches based on historical data, leveraging the Poisson distribution to model match outcomes. The project includes a robust backtesting framework to ensure predictions are evaluated without look-forward bias. Additionally, it uses Bayesian optimization for parameter tuning to enhance model accuracy.

Key Features
Poisson Distribution Model: Estimates the number of goals scored by home and away teams using historical data.
Backtesting Framework: Implements a fair evaluation method by preventing look-forward bias, testing the model's performance on past matches.
Bayesian Optimization: Fine-tunes the model’s hyperparameters using Bayesian optimization to improve accuracy.
Value Betting Identification: Identifies value bets where the model's estimated probabilities differ significantly from bookmaker odds.
Performance Metrics: Includes metrics like accuracy, ROC-AUC, and betting performance, including Sharpe ratio and drawdown.
