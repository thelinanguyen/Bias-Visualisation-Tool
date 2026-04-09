import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
import plotly.express as px


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
        color_discrete_sequence=["#EF553B", "#636EFA"] # Red and Blue
    )
    fig.update_layout(yaxis_title="Number of People", xaxis_title=column_name.capitalize())
    return fig