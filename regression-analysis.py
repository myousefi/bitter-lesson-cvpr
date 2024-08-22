# %%
import bitter_lesson_cvpr

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
OUTPUT_DIR = Path(os.getenv('NLP4SCIENCE_OUTPUT_DIR'))

# Connect to the database
conn = sqlite3.connect('./dbs/cvpr_papers.db')

# Load data
query = """
SELECT p.id, p.year, bls.*, ss.citationCount
FROM papers p
JOIN bitter_lesson_scores_v2 bls ON p.id = bls.paper_id
JOIN semantic_scholar_data ss ON p.id = ss.paper_id
WHERE bls.model = 'gpt-4o'
"""
df = pd.read_sql_query(query, conn)

df = df.dropna()
# %%
# Data Preparation and Exploration
def citation_distribution_analysis(df):
    # Histogram of citation counts
    fig = go.Figure(data=[go.Histogram(x=df['citationCount'])])
    fig.update_layout(title='Distribution of Citation Counts',
                      xaxis_title='Citation Count',
                      yaxis_title='Frequency')
    fig.write_image(OUTPUT_DIR / 'figs' / 'citation_distribution.svg')

    skewness = stats.skew(df['citationCount'])
    kurtosis = stats.kurtosis(df['citationCount'])
    _, p_value = stats.shapiro(df['citationCount'])

    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")
    print(f"Shapiro-Wilk test p-value: {p_value}")

    df['log_citations'] = np.log1p(df['citationCount'])
    return df

df = citation_distribution_analysis(df)

# Correlation Analysis
def correlation_analysis(df):
    corr_matrix = df[['log_citations', 'learning_over_engineering_score', 'search_over_heuristics_score',
                      'scalability_with_computation_score', 'generality_over_specificity_score',
                      'favoring_fundamental_principles_score']].corr()

    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
    fig.update_layout(title='Correlation Heatmap')
    fig.write_image(OUTPUT_DIR / 'figs' / 'correlation_heatmap.svg')

    return corr_matrix

corr_matrix = correlation_analysis(df)

# Multicollinearity Analysis
def vif_analysis(df):
    X = df[['learning_over_engineering_score', 'search_over_heuristics_score',
            'scalability_with_computation_score', 'generality_over_specificity_score',
            'favoring_fundamental_principles_score']]
    X = add_constant(X)
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("Variance Inflation Factors:")
    print(vif)

    condition_number = np.linalg.cond(X)
    print(f"Condition Number: {condition_number}")

    return vif

vif = vif_analysis(df)

# Regression Model Fitting and Comparison
def fit_and_compare_models(df):
    X = df[['learning_over_engineering_score', 'search_over_heuristics_score',
            'scalability_with_computation_score', 'generality_over_specificity_score',
            'favoring_fundamental_principles_score']]
    y = df['log_citations']

    # Standard Linear Regression
    X_with_const = sm.add_constant(X)
    model_ols = sm.OLS(y, X_with_const).fit()
    print(model_ols.summary())

    # Stepwise Regression (forward selection)
    def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
        included = list(initial_list)
        while True:
            changed = False
            excluded = list(set(X.columns) - set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                changed = True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
            if not changed:
                break

        return included

    selected_features = stepwise_selection(X, y)
    model_stepwise = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
    print("Stepwise Regression Results:")
    print(model_stepwise.summary())

    # Regularized Regression Models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        mse = np.mean(cross_val_score(model, X_scaled, y, scoring='neg_mean_squared_error', cv=5)) * -1
        r2 = np.mean(cross_val_score(model, X_scaled, y, scoring='r2', cv=5))
        results[name] = {'MSE': mse, 'R2': r2, 'Coefficients': dict(zip(X.columns, model.coef_))}
        print(f"{name} Regression Results:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")
        print("Coefficients:")
        for feature, coef in zip(X.columns, model.coef_):
            print(f"{feature}: {coef}")
        print("\n")

    return model_ols, model_stepwise, results

model_ols, model_stepwise, regularized_results = fit_and_compare_models(df)

# Model Diagnostics
def model_diagnostics(model, X, y):
    residuals = model.resid
    fitted_values = model.fittedvalues

    # Residuals vs Fitted
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fitted_values, y=residuals, mode='markers'))
    fig.update_layout(title='Residuals vs Fitted',
                      xaxis_title='Fitted values',
                      yaxis_title='Residuals')
    fig.write_image(OUTPUT_DIR / 'figs' / 'residuals_vs_fitted.svg')

    # Q-Q plot
    qq = stats.probplot(residuals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers'))
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1], mode='lines'))
    fig.update_layout(title='Q-Q plot', xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
    fig.write_image(OUTPUT_DIR / 'figs' / 'qq_plot.svg')

    # Cook's distance
    influence = model.get_influence()
    (c, _) = influence.cooks_distance
    fig = go.Figure(data=[go.Scatter(y=c, mode='markers')])
    fig.update_layout(title="Cook's distance", xaxis_title='Observation', yaxis_title="Cook's distance")
    fig.write_image(OUTPUT_DIR / 'figs' / 'cooks_distance.svg')

X = df[['learning_over_engineering_score', 'search_over_heuristics_score',
        'scalability_with_computation_score', 'generality_over_specificity_score',
        'favoring_fundamental_principles_score']]
y = df['log_citations']
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
model_diagnostics(model, X, y)

# Model Comparison and Selection
def model_comparison(model_ols, model_stepwise, regularized_results):
    comparison = pd.DataFrame({
        'OLS': {'AIC': model_ols.aic, 'BIC': model_ols.bic, 'Adj R-squared': model_ols.rsquared_adj},
        'Stepwise': {'AIC': model_stepwise.aic, 'BIC': model_stepwise.bic, 'Adj R-squared': model_stepwise.rsquared_adj}
    })
    for name, results in regularized_results.items():
        comparison[name] = {'AIC': np.nan, 'BIC': np.nan, 'Adj R-squared': results['R2']}
    
    print("Model Comparison:")
    print(comparison)

    return comparison

comparison = model_comparison(model_ols, model_stepwise, regularized_results)

# Interpretation and Reporting
def interpret_results(model):
    coef = model.params
    conf_int = model.conf_int()
    
    results = pd.DataFrame({'Coefficient': coef, 'Lower CI': conf_int[0], 'Upper CI': conf_int[1]})
    results['Percentage Change'] = (np.exp(coef) - 1) * 100
    
    print("Coefficient Interpretation:")
    print(results)
    
    print(f"\nR-squared: {model.rsquared}")
    print(f"Adjusted R-squared: {model.rsquared_adj}")
    print(f"F-statistic: {model.fvalue}")
    print(f"Prob (F-statistic): {model.f_pvalue}")

interpret_results(model_ols)

# Save results to LaTeX
def results_to_latex(model):
    with open(OUTPUT_DIR / 'tables' / 'regression_results.tex', 'w') as f:
        f.write(model.summary().as_latex())

results_to_latex(model_ols)

print("Analysis complete. Check the output directory for figures and tables.")
