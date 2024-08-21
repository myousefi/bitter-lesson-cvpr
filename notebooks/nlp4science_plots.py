# %%
import bitter_lesson_cvpr

import sqlite3
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
import statsmodels.formula.api as sm

import os
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = "/Users/moji/Library/CloudStorage/Dropbox-Personal/Apps/Overleaf/cvpr-bitter-lesson/cvpr-bitter-lesson-nlp4science/figs/"

# Connect to the database
conn = sqlite3.connect('../dbs/cvpr_papers.db')

# %%
# %%
# SQL query to count papers with bitter_lesson_scores_v2 by year
query = """
SELECT p.year, COUNT(DISTINCT bls.paper_id) as paper_count
FROM papers AS p
JOIN bitter_lesson_scores_v2 AS bls ON p.id = bls.paper_id
GROUP BY p.year
ORDER BY p.year;
"""

# Load data into a Pandas DataFrame
df_paper_count = pd.read_sql_query(query, conn)

# Create the bar plot
fig = px.bar(df_paper_count, x='year', y='paper_count',
             labels={'year': 'Year', 'paper_count': 'Number of Papers'},
             title='Number of Papers with Bitter Lesson Scores by Year')

# Update layout for better readability
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Number of Papers',
    bargap=0.2,
    plot_bgcolor='white',
    xaxis=dict(tickmode='linear', dtick=1)
)

# Add gridlines
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Show the plot
fig.show(renderer="browser")


# save the plot as svg for use in latex
fig.write_image(f"{OUTPUT_DIR}/papers_with_bitter_lesson_scores_by_year.svg")


# %%
# SQL query to calculate average scores for each category per year for gpt-4o
query = """
SELECT
    p.year,
    AVG(bls.learning_over_engineering_score) AS avg_learning_over_engineering_score,
    AVG(bls.search_over_heuristics_score) AS avg_search_over_heuristics_score,
    AVG(bls.scalability_with_computation_score) AS avg_scalability_with_computation_score,
    AVG(bls.generality_over_specificity_score) AS avg_generality_over_specificity_score,
    AVG(bls.favoring_fundamental_principles_score) AS avg_favoring_fundamental_principles_score
FROM papers AS p
JOIN bitter_lesson_scores_v2 AS bls ON p.id = bls.paper_id
WHERE bls.model = 'gpt-4o' OR bls.model = 'gpt-4o-mini-2024-07-18'
GROUP BY p.year
ORDER BY p.year;
"""

# Load data into a Pandas DataFrame
df = pd.read_sql_query(query, conn)
# Create the line plot
fig = go.Figure()

# Add a line trace for each score category
for column in df.columns[1:]:  # Skip the 'year' column
    trace_name = column.replace("avg_", "").replace("_", " ").title().replace(" Score", "") 
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df[column],
            mode='lines+markers',  # Use both lines and markers
            name=trace_name,
            marker=dict(size=8),  # Adjust marker size as needed
        )
    )

# Add vertical bars with annotations for significant papers
significant_papers = {
    2005: "Histograms of Oriented Gradients",
    2006: "Photo Tourism",
    2009: "ImageNet Database",
    2012: "AlexNet",
    2014: "VGGNet",
    2014: "GANs",
    2015: "YOLO",
    2015: "ResNet",
    2017: "Transformer",
    2018: "BERT",
    2020: "NeRF",
    2021: "ViT",
    2022: "Latent Diffusion Models",
    2023: "Segment Anything"
}

for year, annotation in significant_papers.items():
    fig.add_vline(x=year, line_width=2, line_dash="dash", line_color="green")
    fig.add_annotation(
        x=year,
        y=8,  # Position the annotation in the middle of the y-axis
        yanchor="top",  # Align the text to the bottom of the plot
        text=annotation,
        showarrow=False,
        # No arrow needed
        font=dict(size=10),
        bgcolor="white",
        textangle=-90,
        xanchor="right"  # Align the text to the left of the line
    )

# Update layout 
fig.update_layout(
    title=dict(
        text="Average Bitter Lesson Scores Over Time (gpt-4o)",
        font=dict(size=24)
    ),
    xaxis_title=dict(text="Year", font=dict(size=18)),
    yaxis_title=dict(text="Average Score", font=dict(size=18)),
    xaxis=dict(
        tickfont=dict(size=14),
        dtick=1,
        tickvals=list(range(2005, 2025)),  # Show years from 2005 to 2024
        ticktext=[str(year) for year in range(2005, 2025)]
    ),
    yaxis=dict(tickfont=dict(size=14), gridwidth=1, gridcolor="LightGray"),
    legend=dict(
        font=dict(size=14),
        orientation="h",  # Horizontal legend
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        xref="paper",
        x=0.5
    ),
)

# pio.write_json(fig, OUTPUT_DIR+'line_plot_gpt4o.json')

fig.write_image(OUTPUT_DIR+'line_plot_gpt4o.svg')


fig.show(renderer="browser")

# %%

query = """
SELECT 
    p.id,
    gpt.learning_over_engineering_score as gpt_learning,
    gpt.search_over_heuristics_score as gpt_search,
    gpt.scalability_with_computation_score as gpt_scalability,
    gpt.generality_over_specificity_score as gpt_generality,
    gpt.favoring_fundamental_principles_score as gpt_principles,
    mini.learning_over_engineering_score as mini_learning,
    mini.search_over_heuristics_score as mini_search,
    mini.scalability_with_computation_score as mini_scalability,
    mini.generality_over_specificity_score as mini_generality,
    mini.favoring_fundamental_principles_score as mini_principles
FROM papers p
JOIN bitter_lesson_scores_v2 gpt ON p.id = gpt.paper_id AND gpt.model = 'gpt-4o'
JOIN bitter_lesson_scores_v2 mini ON p.id = mini.paper_id AND mini.model = 'gpt-4o-mini-2024-07-18'
"""

df = pd.read_sql_query(query, conn)

df['gpt_overall'] = df[['gpt_learning', 'gpt_search', 'gpt_scalability', 'gpt_generality', 'gpt_principles']].sum(axis=1)
df['mini_overall'] = df[['mini_learning', 'mini_search', 'mini_scalability', 'mini_generality', 'mini_principles']].sum(axis=1)

import plotly.graph_objects as go
from scipy import stats

def create_scatter_plot(df, x_col, y_col, title):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        marker=dict(size=8),
        name='Scores'
    ))
    
    # Calculate R-squared
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
    r_squared = r_value**2
    
    # Add regression line
    x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    y_range = slope * x_range + intercept
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        name=f'Regression Line (R² = {r_squared:.3f})'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=f'GPT-4o Scores',
        yaxis_title=f'GPT-4o Mini Scores',
        legend=dict(x=0.02, y=0.98),
        width=800,
        height=600
    )
    
    return fig, r_squared

dimensions = [
    ('learning', 'Learning Over Engineering'),
    ('search', 'Search Over Heuristics'),
    ('scalability', 'Scalability with Computation'),
    ('generality', 'Generality Over Specificity'),
    ('principles', 'Favoring Fundamental Principles'),
    ('overall', 'Overall Alignment')
]

for dim, dim_name in dimensions:
    fig, r_squared = create_scatter_plot(
        df, 
        f'gpt_{dim}', 
        f'mini_{dim}', 
        f'{dim_name} Scores: GPT-4o vs GPT-4o Mini'
    )
    
    fig.write_image(f"{OUTPUT_DIR}/{dim}_comparison_scatter.svg")
    print(f"{dim_name} R-squared: {r_squared:.3f}")
    
    fig.show(renderer="browser")

# %%

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate correlation matrix
correlation_matrix = df[[
    'gpt_learning', 'gpt_search', 'gpt_scalability', 'gpt_generality', 'gpt_principles',
    'mini_learning', 'mini_search', 'mini_scalability', 'mini_generality', 'mini_principles'
]].corr()

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap: GPT-4o vs GPT-4o Mini Dimensions')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.svg")
plt.show()

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import kstest
for year in range(2005, 2025):
        # SQL query to get bitter_lesson_score and citation count for the specific year
    query = f"""
    SELECT 
        bls.learning_over_engineering_score + 
        bls.search_over_heuristics_score + 
        bls.scalability_with_computation_score + 
        bls.generality_over_specificity_score + 
        bls.favoring_fundamental_principles_score AS bitter_lesson_score,
        ss.citationCount,
        p.title,
        p.authors
    FROM papers AS p
    JOIN bitter_lesson_scores_v2 AS bls ON p.id = bls.paper_id
    JOIN semantic_scholar_data AS ss ON p.id = ss.paper_id
    WHERE p.year = {year} AND bls.model = 'gpt-4o'
    ORDER BY bitter_lesson_score;
    """

    # Load data into a Pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Step 1: Plotting the Data
    sns.scatterplot(x='bitter_lesson_score', y='citationCount', data=df)
    plt.title('Bitter Lesson Score vs Citation Count')
    plt.show()

    # Step 2: Fit the OLS model without log transformation
    model = sm.OLS(df['citationCount'], sm.add_constant(df['bitter_lesson_score'])).fit()
    print(model.summary())

    # Step 3: Residual Analysis
    residuals = model.resid
    fitted = model.fittedvalues

    # Residuals vs Fitted Values
    sns.residplot(x=fitted, y=residuals, lowess=True)
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    # Q-Q Plot for Normality of Residuals
    qqplot(residuals, line='45')
    plt.title('Q-Q Plot of Residuals')
    plt.show()

    # Durbin-Watson Test for Autocorrelation
    dw_stat = durbin_watson(residuals)
    print(f'Durbin-Watson statistic: {dw_stat}')

    # Kolmogorov-Smirnov Test for Normality
    ks_stat, p_value = kstest(residuals, 'norm')
    print(f'Kolmogorov-Smirnov test statistic: {ks_stat}, p-value: {p_value}')

    # Step 4: Log Transformation of Citation Count
    df['log_citationCount'] = np.log(df['citationCount'].replace(0, 1))

    # Fit the OLS model with log transformation
    log_model = sm.OLS(df['log_citationCount'], sm.add_constant(df['bitter_lesson_score'])).fit()
    print(log_model.summary())

    # Residual Analysis for Log-Transformed Model
    log_residuals = log_model.resid
    log_fitted = log_model.fittedvalues

    # Residuals vs Fitted Values for Log-Transformed Model
    sns.residplot(x=log_fitted, y=log_residuals, lowess=True)
    plt.title('Residuals vs Fitted Values (Log-Transformed)')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    # Q-Q Plot for Normality of Residuals (Log-Transformed)
    qqplot(log_residuals, line='45')
    plt.title('Q-Q Plot of Residuals (Log-Transformed)')
    plt.show()

    # Durbin-Watson Test for Autocorrelation (Log-Transformed)
    log_dw_stat = durbin_watson(log_residuals)
    print(f'Durbin-Watson statistic (Log-Transformed): {log_dw_stat}')

    # Kolmogorov-Smirnov Test for Normality (Log-Transformed)
    log_ks_stat, log_p_value = kstest(log_residuals, 'norm')
    print(f'Kolmogorov-Smirnov test statistic (Log-Transformed): {log_ks_stat}, p-value: {log_p_value}')
# %%
# Create scatter plots for each year from 2013 to 2020 for gpt-4o
for year in range(2005, 2025):
    # SQL query to get bitter_lesson_score and citation count for the specific year
    query = f"""
    SELECT 
        bls.learning_over_engineering_score + 
        bls.search_over_heuristics_score + 
        bls.scalability_with_computation_score + 
        bls.generality_over_specificity_score + 
        bls.favoring_fundamental_principles_score AS bitter_lesson_score,
        ss.citationCount,
        p.title,
        p.authors
    FROM papers AS p
    JOIN bitter_lesson_scores_v2 AS bls ON p.id = bls.paper_id
    JOIN semantic_scholar_data AS ss ON p.id = ss.paper_id
    WHERE p.year = {year} AND bls.model = 'gpt-4o'
    ORDER BY bitter_lesson_score;
    """

    # Load data into a Pandas DataFrame
    df = pd.read_sql_query(query, conn)

    df = df.dropna()

    df['citationCount'] = df['citationCount'].replace(0, 1)

    # Create a new column with the log of citationCount
    df['log_citationCount'] = np.log(df['citationCount'])

    # Create the scatter plot
    fig = px.scatter(
        df, 
        x="bitter_lesson_score", 
        y="citationCount", 
        log_y=True,  # Set y-axis to logarithmic scale
        title=f"Bitter Lesson Score vs. Citations (CVPR {year}, gpt-4o)",
        hover_data=["title", "authors"],  # Show title and authors on hover
        template="simple_white"
    )

    # Fit OLS regression on the log of citationCount
    results = sm.ols('log_citationCount ~ bitter_lesson_score', data=df).fit()

    # Get p-value and coefficient of the bitter_lesson_score
    p_value = results.pvalues['bitter_lesson_score']
    coefficient = results.params['bitter_lesson_score']

    # Calculate the multiplier
    multiplier = np.exp(coefficient)

    # Create a new DataFrame for the OLS line
    ols_df = pd.DataFrame({'bitter_lesson_score': df['bitter_lesson_score']})
    ols_df['log_citationCount'] = results.predict(ols_df)
    ols_df['citationCount'] = np.exp(ols_df['log_citationCount'])

    # Add OLS line to the plot
    fig.add_trace(
        go.Scatter(
            x=ols_df['bitter_lesson_score'], 
            y=ols_df['citationCount'], 
            mode='lines', 
            name='OLS Fit',
            line=dict(color='red')
        )
    )

    # Add p-value annotation to the plot
    fig.add_annotation(
        text=f"p-value: {p_value:.3f}",
        xref="paper", yref="paper",
        x=0.05, y=0.95, showarrow=False,
        font=dict(size=12)
    )


    # Add a plain English explanation of the coefficient and multiplier
    explanation = (
        f"For every 1 point increase in the Bitter Lesson Score, \n"
        f"the number of citations is expected to change by a factor of {multiplier:.3f}."
    )

    fig.add_annotation(
        text=explanation,
        xref="paper", yref="paper",
        x=0.05, y=0.90, showarrow=False,
        font=dict(size=12), align="left"
    )

    # Customize the plot layout
    fig.update_layout(
        xaxis_title="Bitter Lesson Score",
        yaxis_title="Citation Count (Log Scale)",
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
    )

    # Save the plot as a JSON file
    pio.write_json(fig, OUTPUT_DIR + f'scatter_plot_{year}_gpt4o.json')

    # Display the plot (optional)
    fig.show(renderer="browser")

    break
# %%
