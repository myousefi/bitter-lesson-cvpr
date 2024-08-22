# %%
import sqlite3
import pandas as pd
import pingouin as pg
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
OUTPUT_DIR = os.getenv('NLP4SCIENCE_OUTPUT_DIR')
OUTPUT_DIR = Path(OUTPUT_DIR)
# Connect to the database
conn = sqlite3.connect('./dbs/cvpr_papers.db')
# %%
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
# %%
df_wide = df.pivot(index='paper_id', 
                   columns='model', 
                   values=['learning_over_engineering_score',
                           'search_over_heuristics_score',
                           'scalability_with_computation_score',
                           'generality_over_specificity_score',
                           'favoring_fundamental_principles_score'])

df_wide.columns = [f'{col[1]}_{col[0]}' for col in df_wide.columns]
df_wide = df_wide.reset_index()
# %%
def calculate_icc(dataframe, target_col, rater_cols):
    long_data = dataframe.melt(id_vars=[target_col],
                               value_vars=rater_cols,
                               var_name='Model', value_name='Score')
    icc_results = pg.intraclass_corr(data=long_data, targets=target_col, raters='Model', ratings='Score')
    return icc_results
# %%
dimensions = ['learning_over_engineering_score',
              'search_over_heuristics_score',
              'scalability_with_computation_score',
              'generality_over_specificity_score',
              'favoring_fundamental_principles_score']

icc_results = {}

for dim in dimensions:
    rater_cols = [col for col in df_wide.columns if dim in col]
    icc_result = calculate_icc(df_wide, 'paper_id', rater_cols)
    icc_results[dim] = icc_result
# %%
for dim, result in icc_results.items():
    print(f"\nICC Results for {dim}:")
    print(result)
    
    icc2 = result.loc[result['Type'] == 'ICC2', 'ICC'].values[0]
    
    print(f"ICC(2,k) for {dim}: {icc2:.3f}")
    
    if icc2 < 0.5:
        print("Interpretation: Poor reliability")
    elif 0.5 <= icc2 < 0.75:
        print("Interpretation: Moderate reliability")
    elif 0.75 <= icc2 < 0.9:
        print("Interpretation: Good reliability")
    else:
        print("Interpretation: Excellent reliability")
# %%
import bitter_lesson_cvpr

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

# Prepare data for plotting
original_dimensions = ['learning_over_engineering_score',
                       'search_over_heuristics_score',
                       'scalability_with_computation_score',
                       'generality_over_specificity_score',
                       'favoring_fundamental_principles_score']

dimensions = [dim.replace('_score', '').replace('_', ' ').title() for dim in original_dimensions]
icc_values = [icc_results[dim].loc[icc_results[dim]['Type'] == 'ICC2', 'ICC'].values[0] for dim in original_dimensions]

# Create color scale
color_scale = px.colors.diverging.RdYlGn
# Create the bar plot
fig = go.Figure(data=[
    go.Bar(
        x=dimensions, 
        y=icc_values, 
        marker_color=icc_values, 
        marker_colorscale=color_scale,
        text=[f"{value:.2f}" for value in icc_values],  # Add text for each bar
        textposition='inside',  # Position the text inside the bar
        textfont=dict(size=10, color='white'),  # Set text font size and color
    )
])

# Update layout
fig.update_layout(
    title=dict(
        text="ICC Values Across Dimensions",
        font=dict(size=16)
    ),
    xaxis_title=dict(text="Dimensions", font=dict(size=12)),
    yaxis_title=dict(text="ICC Value", font=dict(size=12)),
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
# Add horizontal lines for reliability thresholds
thresholds = [
    (0.5, "Poor", "red"),
    (0.75, "Moderate", "orange"),
    (0.9, "Good", "green")
]

for value, label, color in thresholds:
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=value,
        x1=len(dimensions) - 0.5,
        y1=value,
        line=dict(color=color, width=2, dash="dash"),
        layer="below"  # This places the line below the bars
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

# Save the plot as SVG files for one-column and two-column layouts
fig.write_image(OUTPUT_DIR/ "figs" / "icc_values_across_dimensions_one_column.svg", width=600, height=450, scale=1)
fig.write_image(OUTPUT_DIR/ "figs" /"icc_values_across_dimensions_two_column.svg", width=1200, height=900, scale=1)

# Display the plot
fig.show(renderer="browser")


# %%
import pandas as pd

def create_latex_table(icc_results):
    latex_tables = []
    for dim, result in icc_results.items():
        table = result.round(3)  # Round to 3 decimal places
        latex = table.to_latex(index=False, escape=False)
        latex = f"\\begin{{table}}[h]\n\\centering\n\\caption{{ICC Results for {dim}}}\n{latex}\\end{{table}}\n\n"
        latex_tables.append(latex)
    
    return '\n'.join(latex_tables)

# Generate LaTeX tables
latex_output = create_latex_table(icc_results)

# Save to file
with open(OUTPUT_DIR / "tables" / "icc_results_tables.tex", "w") as f:
    f.write(latex_output)

print(f"LaTeX tables saved to {OUTPUT_DIR / 'icc_results_tables.tex'}")
