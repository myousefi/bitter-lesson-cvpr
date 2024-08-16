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
fig.show()

# Save the plot
pio.write_json(fig, f"{OUTPUT_DIR}/papers_with_bitter_lesson_scores_by_year.json")


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

# Add vertical bars with annotations for significant papers
significant_papers = {
    2005: "Histograms of Oriented Gradients for Human Detection by Navneet Dalal et al.",
    2006: "Photo Tourism: Exploring Photo Collections in 3D by Noah Snavely et al.",
    2009: "ImageNet: A Large-Scale Hierarchical Image Database by Jia Deng et al.",
    2012: "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet) by Alex Krizhevsky et al.",
    2014: "Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet) by Karen Simonyan et al.",
    2014: "Generative Adversarial Networks (GANs) by Ian Goodfellow et al.",
    2015: "You Only Look Once (YOLO): Unified, Real-Time Object Detection by Joseph Redmon et al.",
    2015: "Deep Residual Learning for Image Recognition (ResNet) by Kaiming He et al.",
    2017: "Attention Is All You Need (Transformer) by Ashish Vaswani et al.",
    2018: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin et al.",
    2020: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis by Ben Mildenhall et al.",
    2021: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT) by Alexey Dosovitskiy et al.",
    2022: "High-Resolution Image Synthesis with Latent Diffusion Models by Robin Rombach et al.",
    2023: "Segment Anything by Alexander Kirillov et al."
}

for year, annotation in significant_papers.items():
    fig.add_vline(x=year, line_width=2, line_dash="dash", line_color="green")
    fig.add_annotation(
        x=year,
        y=1.05,  # Position the annotation in the middle of the y-axis
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

pio.write_json(fig, OUTPUT_DIR+'line_plot_gpt4o.json')

fig.show(renderer="browser")

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
