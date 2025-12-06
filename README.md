# Wind Power Forecasting Using Transformers, RNNs, MLPs, and Autoencoders

This repository contains a complete deep learning workflow for offshore wind power time-series forecasting. The project includes data preprocessing, outlier detection, feature denoising using an autoencoder, single-step and multi-step forecasting using MLP, RNN, and Transformer architectures, and performance comparison across multiple loss functions and prediction horizons.

## Overview

The objective of this project is to analyze and forecast normalized offshore wind power data. The workflow includes reading and cleaning the dataset, identifying outliers, reconstructing denoised features using an autoencoder, generating sliding window datasets, and evaluating several deep learning models for short-term forecasting.

## Dataset
<img width="852" height="462" alt="image" src="https://github.com/user-attachments/assets/79cc1e6e-5530-4207-a47b-666ced964f1b" />

The dataset contains normalized offshore wind power measurements and additional meteorological features, including wind velocity, rotor diameter, air density, humidity, turbulence intensity, and structural parameters. These variables are used for both denoising and forecasting experiments.

## Autoencoder Denoising
<img width="879" height="483" alt="image" src="https://github.com/user-attachments/assets/3e8e3153-bd1a-4331-93ed-b66ab9537084" />

A fully connected autoencoder is trained to reconstruct the input features and reduce noise. The model includes:

- A bottleneck structure for dimensionality reduction  
- ReLU activations  
- Mean squared error loss  

The reconstructed output is compared with the original data to evaluate denoising quality.

## Outlier Detection

Outliers in the target variable are detected using rolling mean and rolling standard deviation over a fixed window. Points lying outside the mean ± 2 standard deviations are labeled as outliers.

## Sliding Window Preparation

Two forecasting modes are prepared:

### Single-step forecasting
The previous 144 time steps are used to predict the next value.

### Multi-step forecasting
Windows of 144 past observations are used to predict future horizons of  
1, 4, 8, 12, 16, and 24 steps ahead.

All features and target values are scaled using MinMax normalization with training-only fitting.

## Models

Several deep learning architectures are implemented and evaluated:

### MLP
A fully connected model trained using both MSE and Huber loss functions.

### RNN (LSTM)
An LSTM-based model designed to capture temporal dependencies. Evaluated using both MSE and Huber loss.

### Transformer
A custom Transformer encoder architecture implemented with:
- Multi-head attention  
- Position-wise feed-forward network  
- Layer normalization and dropout  
- Global average pooling for sequence representation  

Evaluated using MSE and Huber loss functions for all prediction horizons.
### Multi-step Prediction Comparison (Transformer Only)

| Time Index | Metric | Transformer (MSE) | Transformer (Huber) |
|------------|--------|-------------------|----------------------|
| t+1        | MAE    | 24.1854           | 30.1404              |
| t+1        | MAPE   | 0.0375            | 0.0536               |
| t+1        | RMSE   | 29.6409           | 39.7956              |
| t+4        | MAE    | 28.4684           | 28.9386              |
| t+4        | MAPE   | 0.0095            | 0.0081               |
| t+4        | RMSE   | 33.2122           | 35.6618              |
| t+8        | MAE    | 34.6347           | 25.2773              |
| t+8        | MAPE   | 0.0054            | 0.0046               |
| t+8        | RMSE   | 41.7274           | 30.8781              |
| t+12       | MAE    | 32.7241           | 26.6980              |
| t+12       | MAPE   | 0.0039            | 0.0029               |
| t+12       | RMSE   | 38.8916           | 31.6310              |
| t+16       | MAE    | 31.7033           | 29.4380              |
| t+16       | MAPE   | 0.0027            | 0.0022               |
| t+16       | RMSE   | 36.4002           | 34.3028              |
| t+24       | MAE    | 29.9775           | 30.1381              |
| t+24       | MAPE   | 0.0018            | 0.0016               |
| t+24       | RMSE   | 34.8412           | 35.5519              |

## Loss Function Comparison

Two loss functions are compared across models:
- **Mean Squared Error (MSE)**  
- **Huber Loss**  

The evaluation includes MAE, MAPE, RMSE, and R² in the original unscaled domain.

## Model Evaluation

For single-step prediction, metrics are computed for each architecture under both MSE and Huber losses.  

For multi-step prediction, Transformer models are evaluated for horizons t+1, t+4, t+8, t+12, t+16, and t+24, with metrics computed for each horizon.

## Optimization Experiment

A Slime Mould Algorithm (SMA) is implemented to demonstrate metaheuristic optimization. The algorithm is tested on a sphere function to illustrate convergence behavior.

## Summary

This project provides a comprehensive exploration of deep learning methods for wind power time-series forecasting. It covers noise reduction, outlier detection, supervision under multiple loss functions, and advanced architectures for multi-step forecasting. The results highlight the strengths of Transformer-based models for handling long-range temporal dependencies in energy datasets.
