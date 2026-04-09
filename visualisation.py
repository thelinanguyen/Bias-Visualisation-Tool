# This file contains the visualization functions for the Streamlit app, including confusion matrix comparisons, feature distributions, and group metric comparisons.

import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
import plotly.express as px
import pandas as pd


def plot_confusion_matrix_comparison(y_test, y_pred_base, y_pred_custom):
    """
    Draws two side-by-side confusion matrices using Plotly.
    """
    cm_base = confusion_matrix(y_test, y_pred_base)
    cm_custom = confusion_matrix(y_test, y_pred_custom)

    labels = ['<=50K', '>50K']

    # Create Baseline Figure
    fig_base = ff.create_annotated_heatmap(
        z=cm_base, x=labels, y=labels, 
        colorscale='Blues', showscale=False
    )
    fig_base.update_layout(title="Baseline Model", xaxis_title="Predicted", yaxis_title="Actual", width=300, height=300)

    # Create Custom Figure
    fig_custom = ff.create_annotated_heatmap(
        z=cm_custom, x=labels, y=labels, 
        colorscale='Reds', showscale=False
    )
    fig_custom.update_layout(title="New Model", xaxis_title="Predicted", yaxis_title="Actual", width=300, height=300)

    return fig_base, fig_custom
  
  
  

def plot_feature_distribution(df, column_name):
    """
    Draws a dynamic histogram for any column, split by Income level.
    """
    fig = px.histogram(
        df,
        x=column_name,
        color="income",
        barmode="group",
        title=f"Distribution of {column_name} by Income",
        color_discrete_sequence=["#EF553B", "#636EFA"] 
    )
    fig.update_layout(yaxis_title="Number of People", xaxis_title=column_name.capitalize())
    return fig

def plot_group_metrics(base_group, biased_group, metric="Accuracy"):
    """
    Takes the baseline and biased group metrics, shapes them into a single dataframe,
    and returns a grouped bar chart for a specific metric.
    """
    # 1. Prepare Baseline Data
    df_base = base_group[[metric]].reset_index()
    df_base.columns = ["Demographic Group", "Score"]
    df_base["Model"] = "Baseline"

    # 2. Prepare New Model Data
    df_biased = biased_group[[metric]].reset_index()
    df_biased.columns = ["Demographic Group", "Score"]
    df_biased["Model"] = "New Model"

    # 3. Combine them
    df_combined = pd.concat([df_base, df_biased])

    # 4. Create the Grouped Bar Chart
    fig = px.bar(
        df_combined,
        x="Demographic Group",
        y="Score",
        color="Model",
        barmode="group",
        title=f"{metric} Comparison by Demographic Group",
        color_discrete_sequence=["#636EFA", "#EF553B"] # Blue for Baseline, Red for New Model
    )
    
    # Format the Y-axis as percentages
    if metric == "MCC":
        fig.update_layout(
            yaxis_tickformat=".3f", 
            yaxis_title="MCC Score",
            yaxis_range=[-1, 1]  # MCC spans from -1 to 1
        )
    else:
        fig.update_layout(
            yaxis_tickformat=".1%", 
            yaxis_title=f"{metric} Score",
            yaxis_range=[0, 1]   # Standard metrics span 0 to 100%
        )
    
    return fig