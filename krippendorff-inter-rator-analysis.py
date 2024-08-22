import bitter_lesson_cvpr

import sqlite3
import pandas as pd
import krippendorff
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
OUTPUT_DIR = os.getenv('NLP4SCIENCE_OUTPUT_DIR')
OUTPUT_DIR = Path(OUTPUT_DIR)
# Connect to the database
conn = sqlite3.connect('./dbs/cvpr_papers.db')

# Query data
query = """
SELECT paper_id, model,
       learning_over_engineering_score,
       search_over_heuristics_score,
       scalability_with_computation_score,
       generality_over_specificity_score,
       favoring_fundamental_principles_score
FROM bitter_lesson_scores_v2
"""

df = pd.read_sql_query(query, conn)

# Reshape data
df_wide = df.pivot(index='paper_id', 
                   columns='model', 
                   values=['learning_over_engineering_score',
                           'search_over_heuristics_score',
                           'scalability_with_computation_score',
                           'generality_over_specificity_score',
                           'favoring_fundamental_principles_score'])

df_wide.columns = [f'{col[1]}_{col[0]}' for col in df_wide.columns]
df_wide = df_wide.reset_index()

def calculate_krippendorff_alpha(dataframe, dimension):
    # Extract ratings for the given dimension
    rater_cols = [col for col in dataframe.columns if dimension in col]
    ratings = dataframe[rater_cols]
    
    # Convert to numpy array and transpose
    reliability_data = ratings.values.T
    
    # Calculate Krippendorff's alpha
    alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='ordinal')
    
    return alpha

dimensions = ['learning_over_engineering_score',
              'search_over_heuristics_score',
              'scalability_with_computation_score',
              'generality_over_specificity_score',
              'favoring_fundamental_principles_score']

alpha_results = {}

for dim in dimensions:
    alpha = calculate_krippendorff_alpha(df_wide, dim)
    alpha_results[dim] = alpha

# Print results
for dim, alpha in alpha_results.items():
    print(f"\nKrippendorff's Alpha for {dim}: {alpha:.3f}")
    
    if alpha < 0.4:
        print("Interpretation: Poor reliability")
    elif 0.4 <= alpha < 0.6:
        print("Interpretation: Moderate reliability")
    elif 0.6 <= alpha < 0.8:
        print("Interpretation: Substantial reliability")
    else:
        print("Interpretation: Strong reliability")

# Visualize results
dimensions_display = [dim.replace('_score', '').replace('_', ' ').title() for dim in dimensions]
alpha_values = list(alpha_results.values())

fig = go.Figure(data=[
    go.Bar(
        x=dimensions_display, 
        y=alpha_values, 
        marker_color=alpha_values, 
        marker_colorscale='RdYlGn',
        text=[f"{value:.2f}" for value in alpha_values],
        textposition='inside',
        textfont=dict(size=10, color='white'),
    )
])

fig.update_layout(
    title=dict(
        text="Krippendorff's Alpha Across Dimensions",
        font=dict(size=16)
    ),
    xaxis_title=dict(text="Dimensions", font=dict(size=12)),
    yaxis_title=dict(text="Krippendorff's Alpha", font=dict(size=12)),
    xaxis=dict(
        tickfont=dict(size=10),
        tickangle=45,
    ),
    yaxis=dict(
        tickfont=dict(size=10),
        range=[0, 1],
        gridwidth=1,
        gridcolor="LightGray"
    ),
    font=dict(family="Cambria")
)

thresholds = [
    (0.4, "Poor", "red"),
    (0.6, "Moderate", "orange"),
    (0.8, "Substantial", "green")
]

for value, label, color in thresholds:
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=value,
        x1=len(dimensions) - 0.5,
        y1=value,
        line=dict(color=color, width=2, dash="dash"),
        layer="below"
    )
    fig.add_annotation(
        x=len(dimensions) - 1,
        y=value,
        xref="x",
        yref="y",
        text=f"{label}",
        showarrow=False,
        font=dict(size=8, color=color),
        xanchor="right",
        yanchor="bottom"
    )

fig.write_image(OUTPUT_DIR / "figs" / "krippendorff_alpha_across_dimensions_one_column.svg", width=600, height=450, scale=2)
fig.write_image(OUTPUT_DIR / "figs" / "krippendorff_alpha_across_dimensions_two_column.svg", width=1200, height=900, scale=2)

fig.show(renderer="browser")

# Export results to LaTeX table
def create_latex_table(alpha_results):
    latex_table = "\\begin{table}[h]\n\\centering\n\\caption{Krippendorff's Alpha Results}\n"
    latex_table += "\\begin{tabular}{lcc}\n\\hline\nDimension & Alpha & Interpretation \\\\ \\hline\n"
    
    for dim, alpha in alpha_results.items():
        if alpha < 0.4:
            interpretation = "Poor"
        elif 0.4 <= alpha < 0.6:
            interpretation = "Moderate"
        elif 0.6 <= alpha < 0.8:
            interpretation = "Substantial"
        else:
            interpretation = "Strong"
        
        latex_table += f"{dim.replace('_', ' ').title()} & {alpha:.3f} & {interpretation} \\\\\n"
    
    latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
    return latex_table

latex_output = create_latex_table(alpha_results)

with open(OUTPUT_DIR / "tables" / "krippendorff_alpha_results.tex", "w") as f:
    f.write(latex_output)

print(f"LaTeX table saved to {OUTPUT_DIR / 'tables' / 'krippendorff_alpha_results.tex'}")
