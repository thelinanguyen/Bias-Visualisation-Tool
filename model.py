import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(X_train, y_train):
    """
    This takes whatever data app.py hands it, 
    scales it dynamically, and trains the model.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model_bias(model_pipeline, X_test_custom, y_test, X_test_original, protected_attributes):
    """
    Takes a trained model and evaluates it across protected groups.
    Evaluates the model and returns TWO objects: 
    1. A dictionary of global metrics.
    2. A DataFrame of group-specific metrics.
    """
    
    """@parameter model_pipeline: The trained model pipeline (scaling + classifier)
    @parameter X_test_custom: The modified test set
    @parameter X_test_original: The untouched test set used for baseline model
    @parameter protected_attributes: List of columns that indicate protected groups
    @parameter y_test: The true labels for the test set
    @returns: global_metrics (dict), group_bias_df (DataFrame)
    """
    
    # Model trained using the modified data
    y_pred = model_pipeline.predict(X_test_custom)
    
    results = {}
    
    # CALCULATE OVERALL METRICS
    global_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "Samples": len(y_test)
    }
    
    # CALCULATE METRICS FOR EACH PROTECTED GROUP

    group_results = {}
    for attr_name in protected_attributes:
        
        # The Auditor finds the demographics using the ORIGINAL untouched data
        if attr_name not in X_test_original.columns:
            continue

        mask = X_test_original[attr_name] == 1
        group_size = mask.sum()
        
        if group_size == 0:
             group_results[attr_name] = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0, "Samples": 0}
             continue

        # Calculate metrics
        group_results[attr_name] = {
            "Accuracy": accuracy_score(y_test[mask], y_pred[mask]),
            "Precision": precision_score(y_test[mask], y_pred[mask], zero_division=0),
            "Recall": recall_score(y_test[mask], y_pred[mask], zero_division=0),
            "F1-Score": f1_score(y_test[mask], y_pred[mask], zero_division=0),
            "Samples": group_size
        }

    group_bias_df = pd.DataFrame(group_results).T
    return global_metrics, group_bias_df

def get_demographic_stats(X_data):
    """
    Helper function to calculate the percentage makeup of the dataset.
    """
    stats = {}
    if 'sex_Female' in X_data.columns:
        stats['Female %'] = (X_data['sex_Female'] == 1).mean() * 100
    if 'race_White' in X_data.columns:
        stats['White %'] = (X_data['race_White'] == 1).mean() * 100
    if 'race_Black' in X_data.columns:
        stats['Black %'] = (X_data['race_Black'] == 1).mean() * 100
    if 'race_Other' in X_data.columns:
        stats['Other %'] = (X_data['race_Other'] == 1).mean() * 100
    return stats