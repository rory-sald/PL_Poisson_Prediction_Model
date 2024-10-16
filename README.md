# Poisson Goals Prediction Model

This repository contains a football match prediction model that estimates the likelihood of Over/Under 2.5 goals using a Poisson distribution. The model is built to predict match outcomes based on historical match data and includes a backtesting framework for evaluating prediction performance without look-forward bias.

## Project Overview

- **Poisson Distribution Model**: Uses historical match data to predict goals scored by home and away teams.
- **Backtesting Framework**: Fairly evaluates the model by splitting data chronologically and avoiding look-forward bias.
- **Bayesian Optimization**: Optimizes hyperparameters to improve prediction accuracy.
- **Value Betting Identification**: Identifies betting opportunities based on discrepancies between predicted probabilities and bookmaker odds.
