# %%
import bitter_lesson_cvpr

import sqlite3
import pandas as pd
import plotly.express as px
import plotly.io as pio

import os
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# Connect to the database
conn = sqlite3.connect('/dbs/cvpr_papers.db')

# %%
# SQL query to get all tables and columns
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

for table in tables:
    print(f"Table: {table[0]}")
    cursor.execute(f"PRAGMA table_info({table[0]});")
    columns = cursor.fetchall()
    for column in columns:
        print(f"  Column: {column[1]} (Type: {column[2]})")


# %%
# SQL query to count papers per year
query = """
SELECT year AS publication_year, COUNT(*) AS paper_count
FROM papers
GROUP BY publication_year
ORDER BY publication_year;
"""

# Load data into a Pandas DataFrame
df = pd.read_sql_query(query, conn)

# Create the bar plot
import plotly.express as px
import plotly.graph_objects as go

# Assuming 'df' is your DataFrame with 'publication_year' and 'paper_count' columns

# Create the bar plot using plotly.graph_objects for more customization
fig = go.Figure(
    data=[
        go.Bar(
            x=df["publication_year"],
            y=df["paper_count"],
            text=[f"{int(y)}" for y in df["paper_count"]],  # Add text labels to bars
            textposition="auto",
            textfont=dict(size=10),  # Adjust text size as needed
        )
    ],
    layout=go.Layout(
        title=dict(
            text="Number of CVPR Papers Published Each Year",
            font=dict(size=24),  # Adjust title font size as needed
        ),
        xaxis_title=dict(text="Year", font=dict(size=18)),
        yaxis_title=dict(text="Number of Papers", font=dict(size=18)),
        yaxis=dict(
            tickfont=dict(size=14),
            gridwidth=1,
            gridcolor="LightGray",
            # Adjust dtick for appropriate tick intervals
        ),
        xaxis=dict(
            tickfont=dict(size=14), 
            tickangle=45,  # Rotate x-axis labels by 45 degrees
            dtick=1 # Display every year on x-axis
        ),    
    ),
)

pio.write_json(fig, OUTPUT_DIR+'bar_plot.json')

fig.show(renderer="browser")

# %%
# SQL query to calculate average scores for each category per year
query = """
SELECT
    p.year,
    AVG(bls.generality_of_approach_score) AS avg_generality_of_approach_score,
    AVG(bls.reliance_on_human_knowledge_score) AS avg_reliance_on_human_knowledge_score,
    AVG(bls.scalability_with_computation_score) AS avg_scalability_with_computation_score,
    AVG(bls.leveraging_search_and_learning_score) AS avg_leveraging_search_and_learning_score,
    AVG(bls.complexity_handling_score) AS avg_complexity_handling_score,
    AVG(bls.adaptability_and_generalization_score) AS avg_adaptability_and_generalization_score,
    AVG(bls.autonomy_and_discovery_score) AS avg_autonomy_and_discovery_score
FROM papers AS p
JOIN bitter_lesson_scores AS bls ON p.id = bls.paper_id
GROUP BY p.year
ORDER BY p.year;
"""

# Load data into a Pandas DataFrame
df = pd.read_sql_query(query, conn)

# Create the stacked bar plot
fig = go.Figure()

# Add a bar trace for each score category
for column in df.columns[1:]:  # Skip the 'year' column
    trace_name = column.replace("avg_", "").replace("_", " ").title().replace(" Score", "") 
    fig.add_trace(
        go.Bar(
            x=df["year"],
            y=df[column],
            name=trace_name,
        )
    )

# Update layout for a stacked bar chart
fig.update_layout(
    barmode='stack',
    title=dict(
        text="Average Bitter Lesson Scores Over Time (Stacked)",
        font=dict(size=24)
    ),
    xaxis_title=dict(text="Year", font=dict(size=18)),
    yaxis_title=dict(text="Average Score", font=dict(size=18)),
    xaxis=dict(tickfont=dict(size=14), dtick=1),
    yaxis=dict(tickfont=dict(size=14), gridwidth=1, gridcolor="LightGray"),
    legend=dict(font=dict(size=14)),
)

pio.write_json(fig, OUTPUT_DIR+'stacked_bar_plot.json')


fig.show(renderer="browser")

# %%
# %%
# Create scatter plots for each year from 2013 to 2020
import statsmodels.formula.api as sm
import numpy as np 

for year in range(2013, 2025):
    # SQL query to get bitter_lesson_score and citation count for the specific year
    query = f"""
    SELECT 
        bls.generality_of_approach_score + 
        bls.reliance_on_human_knowledge_score + 
        bls.scalability_with_computation_score + 
        bls.leveraging_search_and_learning_score + 
        bls.complexity_handling_score + 
        bls.adaptability_and_generalization_score + 
        bls.autonomy_and_discovery_score AS bitter_lesson_score,
        ss.citationCount,
        p.title,
        p.authors
    FROM papers AS p
    JOIN bitter_lesson_scores AS bls ON p.id = bls.paper_id
    JOIN semantic_scholar_data AS ss ON p.id = ss.paper_id
    WHERE p.year = {year}
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
        title=f"Bitter Lesson Score vs. Citations (CVPR {year})",
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
    pio.write_json(fig, OUTPUT_DIR + f'scatter_plot_{year}.json')

    # Display the plot (optional)
    fig.show(renderer="browser")

# %%