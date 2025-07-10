# Predicting Used Car Price Differentials with Regression

## Executive Summary

The goal of this project is to identify key factors influencing used car price differentials — the gap between the market value (MMR) and actual sale price and to determine whether a predictive model can be developed to forecast these differences.
I hypothesized that variables such as make, model, year, transmission, body style, condition, and odometer significantly impact pricing differences. A multiple linear regression model was built and evaluated using statistical diagnostics and regression metrics.

## Dataset

Source: Kaggle – [Vehicle Sales Data](https://www.kaggle.com/)
Rows used: 25,000 (after reduction for computational efficiency)
Key variables:  
  `make`, `model`, `year`, `transmission`, `body`, `condition`, `odometer`, `mmr`, `sellingprice`

### Data Preprocessing

Missing values handled by imputing mean for continuous variables
Categorical variables encoded using one-hot encoding
Outliers removed via Interquartile Range (IQR) method
Multicollinearity addressed using Variance Inflation Factor (VIF) (threshold > 10)

### Feature Selection

P-values calculated via linear regression to retain only statistically significant predictors (p < 0.05)
Lasso Regression used for penalized feature selection and model refinement
Top 10 features identified based on non-zero Lasso coefficients

### Model Construction

Model: Multiple Linear Regression
Evaluation Metrics:
  - R-squared
  - Mean Squared Error (MSE)
  - Residual Standard Error (RSE)
  - Shapiro-Wilk test for residual normality

## Visualizations & Outputs

Key charts and visuals generated using Matplotlib and **Seaborn:
  - Distributions of `price_diff` before and after outlier removal
  - Top 10 most influential features from Lasso Regression
  - Residuals histogram and normality check
  - Scatter plot: actual vs. predicted price differentials

All outputs are included in the Colab notebook.

## Findings
- The model identifies statistically significant predictors of pricing differentials.
- However, performance is limited:
  - R² = 0.031
  - MSE = 1,622,434
  - RSE = 1,288.15
- These indicate high unexplained variance and low predictive accuracy.

---

## Limitations

- Linear regression **assumes linearity and normality**, which were not fully satisfied.
- **Interactions and nonlinear effects** were not accounted for.
- Lack of features such as **location** or **market demand trends** reduced model effectiveness.

## Proposed Next Steps
- Explore nonlinear models (e.g., Decision Trees, Random Forests)
- Expand dataset to include regional, economic, and consumer behavior data

## Expected Benefits
- While accuracy is limited, this model provides a baseline understanding of factors that influence used car pricing.
- Future improvements can lead to:
  - More accurate price forecasting
  - Faster sale cycles with better pricing strategies
  - Increased buyer trust through data-backed transparency


