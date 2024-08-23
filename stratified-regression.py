# %%
import bitter_lesson_cvpr

import pyperclip
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

# %%
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
WHERE bls.model IN ('gpt-4o', 'gpt-4o-mini-2024-07-18', 'claude-3-5-sonnet-20240620')
GROUP BY p.id, p.year, ss.citationCount
"""
df = pd.read_sql_query(query, conn)

df = df.dropna()


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
            if p_value <= 0.01:
                stars = '***'
            elif p_value <= 0.05:
                stars = '**'
            elif p_value <= 0.1:
                stars = '*'
            return f"{coef:.3f}{stars}"
        
        row = [
            year, r_squared, int(n_obs),
            format_coef(coefficients['learning_over_engineering_score'], p_values['learning_over_engineering_score']),
            format_coef(coefficients['search_over_heuristics_score'], p_values['search_over_heuristics_score']),
            format_coef(coefficients['scalability_with_computation_score'], p_values['scalability_with_computation_score']),
            format_coef(coefficients['generality_over_specificity_score'], p_values['generality_over_specificity_score']),
            format_coef(coefficients['favoring_fundamental_principles_score'], p_values['favoring_fundamental_principles_score'])
        ]
        table_data.append(row)
    
    df_table = pd.DataFrame(table_data, columns=[
        'Year', 'R-squared','N',
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
pyperclip.copy(latex_table)
# %%
# %%
# Stratified Regression Analysis with Overall Alignment Score
years = df['year'].unique()
years = sorted(years)

latex_table_data = []

for year in years:
    print(f"Regression Analysis for Year {year}")
    
    year_df = df[df['year'] == year]
    
    # Calculate the overall alignment score
    year_df['overall_alignment_score'] = year_df[['learning_over_engineering_score', 'search_over_heuristics_score',
                                                   'scalability_with_computation_score', 'generality_over_specificity_score',
                                                   'favoring_fundamental_principles_score']].sum(axis=1)
    
    X = year_df[['overall_alignment_score']]
    y = np.log1p(year_df['citationCount'])
    
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    print(model.summary())
    print("\n")
    
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_statistic = model.fvalue
    prob_f = model.f_pvalue
    n_obs = model.nobs
    
    coefficients = model.params
    p_values = model.pvalues
    conf_int = model.conf_int()
    
    def format_coef(coef, p_value, conf_int):
        stars = ''
        if p_value <= 0.01:
            stars = '***'
        elif p_value <= 0.05:
            stars = '**'
        elif p_value <= 0.1:
            stars = '*'
        return f"{coef:.3f}{stars} [{conf_int[0]:.3f}, {conf_int[1]:.3f}]"
    
    row = [
        year, r_squared, int(n_obs), f_statistic, prob_f,
        format_coef(coefficients['overall_alignment_score'], p_values['overall_alignment_score'], conf_int.loc['overall_alignment_score'])
    ]
    latex_table_data.append(row)

df_table = pd.DataFrame(latex_table_data, columns=[
    'Year', 'R-squared', 'N', 'F-statistic', 'Prob (F-statistic)', 'Overall Alignment Score'
])

latex_table = df_table.to_latex(index=False, float_format="%.3f", escape=False)
print(latex_table)

pyperclip.copy(latex_table)

# %%
