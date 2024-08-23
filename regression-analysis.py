# %%
import time
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
def citation_distribution_analysis(df):
    # Histogram of citation counts
    fig = go.Figure(data=[go.Histogram(x=df['citationCount'])])

    skewness = stats.skew(df['citationCount'])
    kurtosis = stats.kurtosis(df['citationCount'])
    _, p_value = stats.shapiro(df['citationCount'])
    mean = np.mean(df['citationCount'])
    std = np.std(df['citationCount'])

    fig.update_layout(title='Distribution of Citation Counts',
                      xaxis_title='Citation Count',
                      yaxis_title='Frequency',
                      annotations=[
                          dict(x=0.95, y=0.95, xref='paper', yref='paper', 
                               text=f"Mean: {mean:.2f}<br>Standard Deviation: {std:.2f}<br>Skewness: {skewness:.2f}<br>Kurtosis: {kurtosis:.2f}<br>Shapiro-Wilk: {p_value:.2f}",
                               showarrow=False, font=dict(size=12), align='left', bordercolor='black', borderwidth=1, borderpad=4, bgcolor='white')
                      ])

    fig.show(renderer="browser")
    fig.write_image(OUTPUT_DIR / 'figs' / 'citation_distribution.svg')
    time.sleep(1)
    fig.write_image(OUTPUT_DIR / 'figs' / 'citation_distribution.pdf', width=600, height=450, scale=4, engine='kaleido')


    # Histogram of log-transformed citation counts
    df['log_citations'] = np.log1p(df['citationCount'])

    log_fig = go.Figure(data=[go.Histogram(x=df['log_citations'])])

    log_skewness = stats.skew(df['log_citations'])
    log_kurtosis = stats.kurtosis(df['log_citations'])
    _, log_p_value = stats.shapiro(df['log_citations'])
    log_mean = np.mean(df['log_citations'])
    log_std = np.std(df['log_citations'])

    log_fig.update_layout(title='Distribution of Log-Transformed Citation Counts',
                          xaxis_title='Log Citation Count',
                          yaxis_title='Frequency',
                          annotations=[
                              dict(x=0.95, y=0.95, xref='paper', yref='paper', 
                                   text=f"Mean: {log_mean:.2f}<br>Standard Deviation: {log_std:.2f}<br>Skewness: {log_skewness:.2f}<br>Kurtosis: {log_kurtosis:.2f}<br>Shapiro-Wilk: {log_p_value:.2f}",
                                   showarrow=False, font=dict(size=12), align='left', bordercolor='black', borderwidth=1, borderpad=4, bgcolor='white')
                          ])

    log_fig.show(renderer="browser")

    time.sleep(1)
    log_fig.write_image(OUTPUT_DIR / 'figs' / 'log_citation_distribution.svg')
    log_fig.write_image(OUTPUT_DIR / 'figs' / 'log_citation_distribution.pdf', width=600, height=450, scale=4, engine='kaleido')


    return df

df = citation_distribution_analysis(df)


# %%
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
    fig.show(renderer="browser")

    # Q-Q plot
    qq = stats.probplot(residuals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers'))
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1], mode='lines'))
    fig.update_layout(title='Q-Q plot', xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
    fig.write_image(OUTPUT_DIR / 'figs' / 'qq_plot.svg')
    fig.show(renderer="browser")

    # Cook's distance
    influence = model.get_influence()
    (c, _) = influence.cooks_distance
    fig = go.Figure(data=[go.Scatter(y=c, mode='markers')])
    fig.update_layout(title="Cook's distance", xaxis_title='Observation', yaxis_title="Cook's distance")
    fig.write_image(OUTPUT_DIR / 'figs' / 'cooks_distance.svg')
    fig.show(renderer="browser")

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
# %%
# Load data
# Load data
query = """
SELECT
    p.id, p.year, 
    AVG(bls.learning_over_engineering_score) AS learning_over_engineering_score,
    AVG(bls.search_over_heuristics_score) AS search_over_heuristics_score,
    AVG(bls.scalability_with_computation_score) AS scalability_with_computation_score,
    AVG(bls.generality_over_specificity_score) AS generality_over_specificity_score,
    AVG(bls.favoring_fundamental_principles_score) AS favoring_fundamental_principles_score,
    ss.citationCount
FROM papers p
JOIN bitter_lesson_scores_v2 bls ON p.id = bls.paper_id
JOIN semantic_scholar_data ss ON p.id = ss.paper_id
WHERE bls.model IN ('claude-3-5-sonnet-20240620', 'gpt-4o', 'gpt-4o-mini-2024-07-18')
GROUP BY p.id, p.year, ss.citationCount
"""
df = pd.read_sql_query(query, conn)

# df = df.dropna()


# Stratified Regression Analysis
years = df['year'].unique()
years = sorted(years)
for year in years:
    print(f"Regression Analysis for Year {year}")
    
    year_df = df[df['year'] == year]
    
    X = year_df[['learning_over_engineering_score', 'search_over_heuristics_score',
                    'scalability_with_computation_score', 'generality_over_specificity_score',
                    'favoring_fundamental_principles_score']]
    y = np.log1p(year_df['citationCount'])
    
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    print(model.summary())
    print("\n")

# %%
def create_regression_table(years, models):
    table_data = []
    for year, model in zip(years, models):
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        f_statistic = model.fvalue
        prob_f = model.f_pvalue
        n_obs = model.nobs
        
        coefficients = model.params
        p_values = model.pvalues
        
        def format_coef(coef, p_value):
            stars = ''
            if p_value <= 0.05:
                stars = '**'
            elif p_value <= 0.1:
                stars = '*'
            return f"{coef:.3f}{stars}"
        
        row = [
            year, r_squared, adj_r_squared, f_statistic, prob_f, n_obs,
            format_coef(coefficients['learning_over_engineering_score'], p_values['learning_over_engineering_score']),
            format_coef(coefficients['search_over_heuristics_score'], p_values['search_over_heuristics_score']),
            format_coef(coefficients['scalability_with_computation_score'], p_values['scalability_with_computation_score']),
            format_coef(coefficients['generality_over_specificity_score'], p_values['generality_over_specificity_score']),
            format_coef(coefficients['favoring_fundamental_principles_score'], p_values['favoring_fundamental_principles_score'])
        ]
        table_data.append(row)
    
    df_table = pd.DataFrame(table_data, columns=[
        'Year', 'R-squared', 'Adj. R-squared', 'F-statistic', 'Prob(F-statistic)', 'N',
        'Learning', 'Search', 'Scalability', 'Generality', 'Principles'
    ])
    
    latex_table = df_table.to_latex(index=False, float_format="%.3f", escape=False)
    return latex_table

# In your main code, after running the regressions:
years = []
models = []
for year in sorted(df['year'].unique()):
    year_df = df[df['year'] == year]
    X = year_df[['learning_over_engineering_score', 'search_over_heuristics_score',
                 'scalability_with_computation_score', 'generality_over_specificity_score',
                 'favoring_fundamental_principles_score']]
    y = np.log1p(year_df['citationCount'])
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    years.append(year)
    models.append(model)

latex_table = create_regression_table(years, models)
print(latex_table)

# %%
# Query to get distinct models
models_query = """
SELECT DISTINCT model
FROM bitter_lesson_scores_v2;
"""

# Execute the query to get distinct models
models = pd.read_sql_query(models_query, conn)['model'].tolist()

# Generate the COUNT expressions for each model
count_expressions = [f"COUNT(CASE WHEN bls.model = '{model}' THEN 1 END) AS {model.replace('-', '_')}" for model in models]

# Query to get the number of scores for each year and model
query = f"""
SELECT 
    p.year,
    {', '.join(count_expressions)}
FROM papers p
JOIN bitter_lesson_scores_v2 bls ON p.id = bls.paper_id
GROUP BY p.year
ORDER BY p.year;
"""

# Execute the query and fetch the results
results = pd.read_sql_query(query, conn)

# Print the results as a table
print(results.to_string(index=False))


# %%
# Query to get distinct models
models_query = """
SELECT DISTINCT model
FROM bitter_lesson_scores_v2;
"""

# Execute the query to get distinct models
models = pd.read_sql_query(models_query, conn)['model'].tolist()

# Generate the COUNT expressions for each filtered model
count_expressions = [f"COUNT(CASE WHEN bls.model = '{model}' THEN 1 END) AS {model.replace('-', '_')}" for model in models]

# Query to get the number of scores for each year and filtered model
query = f"""
SELECT 
    p.year,
    {', '.join(count_expressions)},
    COUNT(DISTINCT p.id) AS total_papers
FROM papers p
JOIN bitter_lesson_scores_v2 bls ON p.id = bls.paper_id
WHERE p.id IN (
    SELECT paper_id
    FROM bitter_lesson_scores_v2
    WHERE model IN ({','.join(["'" + model + "'" for model in models])})
    GROUP BY paper_id
    HAVING COUNT(DISTINCT model) = {len(models)}
)
GROUP BY p.year
ORDER BY p.year;
"""

# Execute the query and fetch the results
results = pd.read_sql_query(query, conn)

# Print the results as a table
print(results.to_string(index=False))

# %%
