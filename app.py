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
    st.write(f"**Original Training Size:** {len(X_train)} rows")
    
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
# RIGHT COLUMN: COMPARISON DASHBOARD
# ------------------------------------------
with col_results:
    st.header("2. Model Comparison")
    
    if not run_button:
        st.info("👈 Adjust the dataset controls on the left and click 'Train New Model' to see the comparison.")
    
    if run_button:
        with st.spinner("Applying biases and training custom model..."):
            
            # --- A. APPLY USER FILTERS ---
            X_train_custom = X_train.copy()
            y_train_custom = y_train.copy()
            X_test_custom = X_test.copy()

            # 1. Apply Row Filters (To TRAIN only)
            if bias_group != "None" and bias_percentage > 0:
                
                # Find everyone in the target demographic
                group_mask = (X_train_custom[bias_group] == 1)
                
                # Narrow it down by income condition
                if bias_condition == "Only High-Earners (>50K)":
                    group_mask = group_mask & (y_train_custom == 1)
                elif bias_condition == "Only Low-Earners (<=50K)":
                    group_mask = group_mask & (y_train_custom == 0)
                
                # Get the exact row indices for the people who match
                target_indices = X_train_custom[group_mask].index
                
                # Calculate exactly how many rows to delete based on the slider
                num_to_drop = int(len(target_indices) * (bias_percentage / 100.0))
                
                # E. Randomly select and drop those rows
                if num_to_drop > 0:
                    import numpy as np
                    np.random.seed(42) # Keeps the random drop consistent if they click twice
                    drop_indices = np.random.choice(target_indices, size=num_to_drop, replace=False)
                    
                    X_train_custom = X_train_custom.drop(drop_indices)
                    y_train_custom = y_train_custom.drop(drop_indices)
                    
                    # Optional: Tell the user exactly what happened!
                    st.toast(f"🗑️ Dropped {num_to_drop} rows matching: {bias_group} ({bias_condition})")

            # 2. Apply Column Filters (To TRAIN and TEST)
            if len(cols_to_drop) > 0:
                X_train_custom = X_train_custom.drop(columns=cols_to_drop)
                X_test_custom = X_test_custom.drop(columns=cols_to_drop)
                

            # --- B. TRAIN AND EVALUATE NEW MODEL ---
            biased_model = md.train_model(X_train_custom, y_train_custom)
            
            # Note: We pass X_test_original (X_test) to allow the auditor to check demographic metrics!
            biased_global, biased_group = md.evaluate_model_bias(
                biased_model, X_test_custom, y_test, X_test, protected_attributes
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
            
            tab1, tab2 = st.tabs(["New Model Results", "Baseline Model Results"])
            with tab1:
                # Format the dataframe to look like percentages
                st.dataframe(biased_group.style.format("{:.1%}", subset=["Accuracy", "Precision", "Recall", "F1-Score"]))
            with tab2:
                st.dataframe(base_group.style.format("{:.1%}", subset=["Accuracy", "Precision", "Recall", "F1-Score"]))

            st.divider()

            # --- E. DISPLAY CONFUSION MATRICES ---
            st.subheader("Confusion Matrix Comparison")
            
            y_pred_base = baseline_model.predict(X_test)
            y_pred_biased = biased_model.predict(X_test_custom)
            
            fig_base, fig_biased = vis.plot_confusion_matrix_comparison(y_test, y_pred_base, y_pred_biased)
            
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_base, use_container_width=True)
            with c2:
                st.plotly_chart(fig_biased, use_container_width=True)