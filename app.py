import streamlit as st
import pandas as pd
import data_preparation as dp
import model as md
import visualisation as vis

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Bias Dashboard", layout="wide")

# ==========================================
# BACKEND: CACHE DATA & BASELINE MODEL
# ==========================================
@st.cache_data
def load_and_cache_data():
    return dp.get_clean_census_data()

X_train, X_test, y_train, y_test, protected_attributes, raw_df = load_and_cache_data()

@st.cache_resource
def load_and_cache_baseline(_X_train, _y_train):
    return md.train_model(_X_train, _y_train)

baseline_model = load_and_cache_baseline(X_train, y_train)

@st.cache_data
def get_baseline_metrics(_model, _X_test, _y_test, _X_test_original, _protected_attributes):
    return md.evaluate_model_bias(_model, _X_test, _y_test, _X_test_original, _protected_attributes)

# Calculate baseline metrics once
base_global, base_group = get_baseline_metrics(baseline_model, X_test, y_test, X_test, protected_attributes)


st.title("Dataset Bias Diagnostic Tool")

# ==========================================
# NEW: DATA EXPLORER EXPANDER
# ==========================================
with st.expander("📊 Explore the Original Dataset", expanded=False):
    
    # 1. The Interactive Histogram
    # We remove 'income' from the dropdown options so they don't plot income vs income
    feature_options = [col for col in raw_df.columns if col != 'income']
    selected_feature = st.selectbox("Select a feature to visualize:", feature_options, index=0)
    
    fig_eda = vis.plot_feature_distribution(raw_df, selected_feature)
    st.plotly_chart(fig_eda, use_container_width=True)
    
    # 2. The Data Table (Showing only a sample so it doesn't lag)
    st.subheader("Raw Data Sample (First 100 Rows)")
    st.dataframe(raw_df.head(100), use_container_width=True)

st.divider()

# ==========================================
# FRONTEND: UI LAYOUT
# ==========================================


# Create two main columns: Left (Controls - 30% width), Right (Results - 70% width)
col_controls, col_results = st.columns([1, 2.5])

# ------------------------------------------
# LEFT COLUMN: DATASET CONTROLS
# ------------------------------------------
with col_controls:
    st.header("1. Dataset Controls")
    
    # Ommitted Variable Bias Controls (Column Dropping) #
    st.subheader("Omitted Variable Bias (Columns)")
    # Let the user pick any column except the protected attributes (to avoid crashing the app)
    safe_cols = [c for c in X_train.columns if c not in protected_attributes]
    cols_to_drop = st.multiselect("Hide Features from Model:", options=safe_cols, default=["education-num"])

    # Representation Bias #
    st.subheader("Representation Bias (Rows)")
    st.write("Starve the model of specific demographic examples.")
    
    # 1. Choose the group 
    bias_group = st.selectbox("Select Demographic to Reduce:", options=["None"] + protected_attributes)
    
    # 2. Choose the condition
    bias_condition = st.radio(
        "Apply reduction to:", 
        options=["All Income Levels", "Only High-Earners (>50K)", "Only Low-Earners (<=50K)"]
    )
    
    # 3. Choose the severity
    bias_percentage = st.slider("Percentage of group to remove:", min_value=0, max_value=90, value=0, step=10)
    
    run_button = st.button("Train New Model", type="primary", use_container_width=True)

# ------------------------------------------
# RIGHT COLUMN: RESULTS DISPLAY
# ------------------------------------------
with col_results:
    st.header("2. Model Comparison")
    
    # --- 1. THE TRAINING PHASE ---
    # We only run the heavy training process when the button is actually clicked
    if run_button:
        with st.spinner("Applying biases and training custom model..."):
            
            # ... (KEEP ALL YOUR EXISTING FILTERING LOGIC HERE: A. APPLY USER FILTERS) ...
            X_train_custom = X_train.copy()
            y_train_custom = y_train.copy()
            X_test_custom = X_test.copy()
            st.session_state['rows_dropped'] = 0

            if bias_group != "None" and bias_percentage > 0:
                group_mask = (X_train_custom[bias_group] == 1)
                if bias_condition == "Only High-Earners (>50K)":
                    group_mask = group_mask & (y_train_custom == 1)
                elif bias_condition == "Only Low-Earners (<=50K)":
                    group_mask = group_mask & (y_train_custom == 0)
                
                target_indices = X_train_custom[group_mask].index
                num_to_drop = int(len(target_indices) * (bias_percentage / 100.0))
                st.session_state['rows_dropped'] = num_to_drop
                
                if num_to_drop > 0:
                    import numpy as np
                    np.random.seed(42)
                    drop_indices = np.random.choice(target_indices, size=num_to_drop, replace=False)
                    X_train_custom = X_train_custom.drop(drop_indices)
                    y_train_custom = y_train_custom.drop(drop_indices)
                    st.toast(f"🗑️ Dropped {num_to_drop} rows matching: {bias_group} ({bias_condition})")

            if len(cols_to_drop) > 0:
                X_train_custom = X_train_custom.drop(columns=cols_to_drop)
                X_test_custom = X_test_custom.drop(columns=cols_to_drop)

            # Train and evaluate
            biased_model = md.train_model(X_train_custom, y_train_custom)
            biased_global, biased_group = md.evaluate_model_bias(
                biased_model, X_test_custom, y_test, X_test, protected_attributes
            )
            y_pred_biased = biased_model.predict(X_test_custom)

            # SAVE RESULTS TO MEMORY (st.session_state)
            st.session_state['biased_global'] = biased_global
            st.session_state['biased_group'] = biased_group
            st.session_state['y_pred_biased'] = y_pred_biased
            st.session_state['is_trained'] = True


    # --- 2. THE DISPLAY PHASE ---
    # Check if we have trained data saved in memory
    if not st.session_state.get('is_trained', False):
        st.info("👈 Adjust the dataset controls on the left and click 'Train New Model' to see the comparison.")
    
    else:
        # Retrieve the data from memory so the radio buttons can use it without re-training!
        biased_global = st.session_state['biased_global']
        biased_group = st.session_state['biased_group']
        y_pred_biased = st.session_state['y_pred_biased']

        # --- B. DISPLAY TRAINING DATA CHANGES ---
        st.info(
            "🎓 **Training Data Used:**  \n"
            f"The Baseline model was trained on {len(X_train):,} rows.  \n"
            f"The New Model was trained on {len(X_train) - st.session_state.get('rows_dropped', 0):,} rows.  \n"
            f"Both models were evaluated on the same untouched test set ({len(X_test):,} rows) for a fair comparison."
        )

        # --- C. DISPLAY GLOBAL METRICS (KPI Cards) ---
        st.subheader("Global Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        
        # We calculate the Delta (Biased - Baseline) so the UI shows red/green arrows
        m1.metric("Accuracy", f"{biased_global['Accuracy']:.1%}", f"{biased_global['Accuracy'] - base_global['Accuracy']:.1%}")
        m2.metric("Precision", f"{biased_global['Precision']:.1%}", f"{biased_global['Precision'] - base_global['Precision']:.1%}")
        m3.metric("Recall", f"{biased_global['Recall']:.1%}", f"{biased_global['Recall'] - base_global['Recall']:.1%}")
        m4.metric("F1-Score", f"{biased_global['F1-Score']:.1%}", f"{biased_global['F1-Score'] - base_global['F1-Score']:.1%}")

        st.divider()

        # --- D. DISPLAY GROUP METRICS (Tables) ---
        st.subheader("Group Performance Metrics")
        
        st.markdown("**Change from Baseline (New - Baseline)**")
        
        # Calculate the difference (only for the metric columns, keeping samples static)
        metrics_only = ["Accuracy", "Precision", "Recall", "F1-Score"]
        delta_df = biased_group[metrics_only] - base_group[metrics_only]
        
        # Style the dataframe to highlight negative drops in red and positive gains in green
        def color_negative_red(val):
            if isinstance(val, (int, float)):
                color = 'red' if val < 0 else 'green' if val > 0 else 'gray'
                return f'color: {color}'
            return ''

        st.dataframe(
            delta_df.style
            .format({m: "{:+.1%}" for m in metrics_only})
            .map(color_negative_red, subset=metrics_only),
            use_container_width=True
        )
        
        # 1. Combine the DataFrames
        combined_df = pd.concat([base_group, biased_group], axis=1, keys=['Baseline', 'New Model'])
        
        # 2. Swap the column levels so the Metric (Accuracy, etc.) is on top
        combined_df = combined_df.swaplevel(0, 1, axis=1)
        
        # 3. Reorder the top-level columns so they appear in our preferred order
        metric_order = ["Accuracy", "Precision", "Recall", "F1-Score"]
        combined_df = combined_df[metric_order]
        
        # 4. Create specific formatting rules for the MultiIndex columns
        format_rules = {}
        for metric in metric_order:
            format_rules[(metric, 'Baseline')] = "{:.1%}"
            format_rules[(metric, 'New Model')] = "{:.1%}"

        # 5. Display the styled DataFrame
        st.dataframe(combined_df.style.format(format_rules), use_container_width=True)
        
        st.divider()

        # --- D.2 VISUALIZE GROUP METRICS ---
        st.subheader("Visual Metric Comparison")
        
        # Let the user choose which metric to look at on the chart
        chart_metric = st.radio(
            "Select a metric to chart:",
            options=["Accuracy", "Precision", "Recall", "F1-Score"],
            horizontal=True
        )
        
        # Generate and display the chart
        fig_groups = vis.plot_group_metrics(base_group, biased_group, metric=chart_metric)
        st.plotly_chart(fig_groups, use_container_width=True)
        
        st.divider()

        # --- E. DISPLAY CONFUSION MATRICES ---
        st.subheader("Confusion Matrix Comparison")
        
        y_pred_base = baseline_model.predict(X_test)
        
        # Notice we removed the 'biased_model.predict()' line here and are just 
        # passing the 'y_pred_biased' we retrieved from st.session_state above!
        fig_base, fig_biased = vis.plot_confusion_matrix_comparison(y_test, y_pred_base, y_pred_biased)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_base, use_container_width=True)
        with c2:
            st.plotly_chart(fig_biased, use_container_width=True)