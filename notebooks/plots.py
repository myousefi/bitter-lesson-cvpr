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

OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# Connect to the database
conn = sqlite3.connect('./dbs/cvpr_papers.db')

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
WHERE bls.model = 'gpt-4o'
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

# Update layout 
fig.update_layout(
    title=dict(
        text="Average Bitter Lesson Scores Over Time (gpt-4o)",
        font=dict(size=24)
    ),
    xaxis_title=dict(text="Year", font=dict(size=18)),
    yaxis_title=dict(text="Average Score", font=dict(size=18)),
    xaxis=dict(tickfont=dict(size=14), dtick=1),
    yaxis=dict(tickfont=dict(size=14), gridwidth=1, gridcolor="LightGray"),
    legend=dict(font=dict(size=14)),
)

pio.write_json(fig, OUTPUT_DIR+'line_plot_gpt4o.json')

fig.show(renderer="browser")

# %%
# Create scatter plots for each year from 2013 to 2020 for gpt-4o
for year in range(2013, 2025):
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

    # Get p-value of the bitter_lesson_score coefficient
    p_value = results.pvalues['bitter_lesson_score']

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
# %%
