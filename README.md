# The HTBR Framework for Dynamic Multi-Asset Portfolio Optimization

A novel hybrid deep learning framework combining **Hierarchical Transformers**, **XGBoost**, **Ridge Meta-Learning**, and **Deep Reinforcement Learning (PPO)** for optimized multi-asset portfolio management with sentiment integration.

## Overview

The HTBR Framework addresses dynamic portfolio optimization across diverse asset classes (equities, forex, commodities) by integrating:

- **H**ierarchical Transformer with positional encodings for temporal pattern learning
- **T**abular features with XGBoost gradient boosting
- **B**oosted ensemble via Ridge meta-learning
- **R**einforcement Learning (PPO) for adaptive allocation strategies

This framework achieves superior risk-adjusted returns by combining price forecasting, sentiment analysis, and dynamic rebalancing under realistic transaction costs.

## Key Features

- **Multi-Model Ensemble**: Combines Transformer temporal modeling with XGBoost feature engineering
- **Sentiment Integration**: FinBERT-based news sentiment analysis with EMA smoothing
- **Hierarchical Architecture**: Asset-specific temporal encoders with cross-asset attention
- **Advanced RL Agent**: PPO-based portfolio manager with risk-aware reward shaping
- **Real-World Constraints**: Transaction costs, position limits, and turnover penalties
- **Comprehensive Evaluation**: Sharpe ratio, maximum drawdown, volatility targeting

## Architecture

### 1. Price Forecasting Pipeline

#### Hierarchical Transformer
- **Per-Asset Temporal Encoding**: 3 transformer layers with sinusoidal positional embeddings
- **Cross-Asset Attention**: 2 layers for inter-asset dependency modeling
- **Training**: Huber loss, cosine learning rate decay, gradient clipping
- **Regularization**: Dropout (0.1), early stopping with validation holdout

#### XGBoost Boosters
- **Features**: Flattened 60-step price windows + engineered momentum/volatility
- **Optimization**: RandomizedSearchCV hyperparameter tuning per asset
- **Configuration**: Histogram-based tree method, L2 regularization

#### Stacked Meta-Learner
- **Architecture**: RidgeCV with cross-validated alpha selection
- **Inputs**: Transformer predictions + XGBoost predictions + sentiment scores
- **Output**: Final hybrid price forecasts

### 2. Sentiment Analysis Module

- **Model**: ProsusAI/FinBERT for financial news classification
- **Processing**: Daily aggregated sentiment scores per asset
- **Enhancement**: 3-day EMA smoothing + clipping to [-0.5, 0.5] range
- **Integration**: Normalized sentiment features in both forecasting and RL

### 3. Reinforcement Learning Portfolio Manager

#### Environment (EnhancedPortfolioEnv)
- **State Space**: 
  - Normalized hybrid forecasts
  - Current portfolio weights (deviation from equal-weight)
  - Realized volatility indicator
- **Action Space**: Continuous logits → projected to [0.05, 0.40] weight bounds
- **Constraints**: Minimum 5% / maximum 40% per asset

#### Reward Design
```
reward = net_return * 1.0
       - 2.0 * |realized_vol - target_vol|  # Volatility targeting
       - 5.0 * (drawdown penalty if DD < -25%)
```

#### PPO Configuration
- **Policy**: MLP (256-192 hidden units, Tanh activation)
- **Training**: 1M timesteps, 4096 steps/batch, 10 epochs/update
- **Hyperparameters**: 
  - Learning rate: 3e-4
  - Gamma: 0.995, GAE λ: 0.98
  - Clip range: 0.15, entropy coef: 0.01

## Installation

```
# Clone repository
git clone https://github.com/karthikkemidi/The-HTBR-Framework-for-Dynamic-Multi-Asset-Portfolio-Optimization.git
cd The-HTBR-Framework-for-Dynamic-Multi-Asset-Portfolio-Optimization

# Install dependencies
pip install -q "stable-baselines3[extra]" "gymnasium==0.29.1" "shimmy<1.4.0"
pip install yfinance xgboost scikit-learn tensorflow torch transformers pandas matplotlib
```

## Usage

### Quick Start

```
# Run the complete pipeline
python MPCode.py
```

### Pipeline Stages

#### 1. Data Collection (2015-2025)
```
assets = ["AAPL", "EURUSD=X", "GC=F"]  # Equity, Forex, Commodity
data = yf.download(assets, start="2015-01-01", end="2025-01-01")["Close"]
```

#### 2. Feature Engineering
- 60-step sequence windows with MinMax scaling (60-20-20 train-val-test split)
- FinBERT sentiment extraction aligned to trading dates
- Technical indicators: rolling volatility, momentum

#### 3. Model Training
```
# Hierarchical Transformer
trans_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=80)

# XGBoost per-asset
for asset in range(num_assets):
    xgb_models[asset].fit(X_train_tabular, y_train[:, asset])

# Ridge Meta-Learner
meta_model.fit(stacked_predictions, y_train)
```

#### 4. RL Portfolio Optimization
```
# Initialize PPO agent
model = PPO("MlpPolicy", env_train, n_steps=4096, batch_size=1024)
model.learn(total_timesteps=1_000_000)

# Evaluate on test set
portfolio_weights, daily_returns = evaluate_policy(model, env_test)
```

## Performance Metrics

### Evaluation on Test Period (2021-2025)

| Metric | Target | Achieved |
|--------|--------|----------|
| **Sharpe Ratio** | > 1.0 | Reported in output |
| **Max Drawdown** | < -15% | Monitored continuously |
| **Annualized Volatility** | ~10-15% | Risk-targeted |
| **Win Rate** | > 55% | Calculated on daily returns |

### Output Visualizations

1. **Portfolio Performance**: Cumulative returns, drawdown, dynamic weights
2. **Rolling Sharpe**: 30-day rolling risk-adjusted performance
3. **Prediction Accuracy**: Actual vs. forecasted prices per asset

## Project Structure

```
├── MPCode.py                  # Main pipeline script
├── MPoct19.ipynb             # Jupyter notebook version
├── MP Final Report.pdf        # Detailed research documentation
├── Paper.pdf                  # Research paper
├── HTBR.docx                 # Framework documentation
├── HTBR.pptx                 # Presentation slides
├── ppo_enhanced_htbr_v2.zip  # Trained PPO model
├── vec_normalize_enhanced_v2.pkl  # Normalization stats
├── eval_diagnostics.pkl      # Evaluation metrics
└── *.png                     # Output visualizations
```

## Key Innovations

1. **Hybrid Forecasting**: Combines deep learning (Transformer) with traditional ML (XGBoost) through stacked generalization
2. **Sentiment-Aware**: Integrates NLP-based market sentiment into both prediction and allocation
3. **Risk-Conscious RL**: Multi-objective reward balancing returns, volatility, and drawdown
4. **Production-Ready**: Handles transaction costs, weight constraints, and numerical stability

## Technical Requirements

- **Python**: 3.8+
- **GPU**: CUDA-compatible (optional, significantly speeds training)
- **Memory**: 16GB+ RAM recommended for full pipeline
- **Data**: Internet connection for yfinance downloads




## Future Enhancements

- [ ] Multi-objective Pareto optimization for risk-return trade-offs
- [ ] Alternative RL algorithms (SAC, TD3, Rainbow DQN)
- [ ] Expanded asset universe (crypto, bonds, ETFs)
- [ ] Real-time deployment with live data feeds
- [ ] Explainability module (SHAP, attention visualization)

## License

This project is available for academic and research purposes. For commercial use, please contact the author.

## Contact

**Author**: Karthik Kemidi  
**Repository**: [GitHub](https://github.com/karthikkemidi/The-HTBR-Framework-for-Dynamic-Multi-Asset-Portfolio-Optimization)

---

**Disclaimer**: This framework is for research and educational purposes only. Past performance does not guarantee future results. Always conduct thorough due diligence before making investment decisions.
