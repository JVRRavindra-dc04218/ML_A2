import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cloudpickle
import traceback
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Advanced ML Analytics", layout="wide")

st.title("🛡️ Multiple Classificaiton modal")
st.write("Specialized Machine Learning - Performance Visualization Interface")

SIDEBAR_CONFIG_HEADER = "System Settings"
STORED_MODELS_PATH = "model"

identified_classifiers = [
    f.replace(".joblib", "").replace("_", " ").title() 
    for f in os.listdir(STORED_MODELS_PATH) 
    if f.endswith(".joblib")
]

st.sidebar.header(SIDEBAR_CONFIG_HEADER)
active_learner = st.sidebar.selectbox("Choose Classification Engine", identified_classifiers)

st.sidebar.subheader("Data Intake")
CLIENT_DATA_INPUT = st.sidebar.file_uploader("Import Evaluation Records (CSV)", type=["csv"])

def convert_to_dense(matrix_data):
    return matrix_data.toarray() if hasattr(matrix_data, "toarray") else matrix_data

def extract_confidence_scores(learner_obj, feature_set):
    if hasattr(learner_obj, "predict_proba"):
        return learner_obj.predict_proba(feature_set)[:, 1]
    if hasattr(learner_obj, "decision_function"):
        raw_vals = learner_obj.decision_function(feature_set)
        v_min, v_max = np.min(raw_vals), np.max(raw_vals)
        if v_max == v_min:
            return np.zeros_like(raw_vals, dtype=float)
        return (raw_vals - v_min) / (v_max - v_min)
    return learner_obj.predict(feature_set).astype(float)

if CLIENT_DATA_INPUT is not None:
    input_data = pd.read_csv(CLIENT_DATA_INPUT)
    st.write("### Input Data Snapshot")
    st.dataframe(input_data.head())

    TARGET_FIELD = "TARGET_COL"
    
    if TARGET_FIELD in input_data.columns:
        eval_features = input_data.drop(columns=[TARGET_FIELD])
        eval_labels = input_data[TARGET_FIELD].astype(str).str.strip().replace({"1": 1, "0": 0}).astype(int)
        
        selected_file_path = os.path.join(STORED_MODELS_PATH, f"{active_learner.lower().replace(' ', '_')}.joblib")
        
        if os.path.exists(selected_file_path):
            trained_model = None
            # Try to load with joblib first; if that fails show traceback and attempt a cloudpickle fallback.
            try:
                trained_model = joblib.load(selected_file_path)
            except Exception:
                st.error("Failed to load model with joblib. Showing traceback and attempting cloudpickle fallback (if available).")
                st.text(traceback.format_exc())
                cp_path = selected_file_path.replace('.joblib', '.cpkl')
                if os.path.exists(cp_path):
                    try:
                        with open(cp_path, 'rb') as _f:
                            trained_model = cloudpickle.load(_f)
                        st.success("Loaded model via cloudpickle fallback.")
                    except Exception:
                        st.error("Cloudpickle fallback also failed. See traceback below.")
                        st.text(traceback.format_exc())
                else:
                    st.warning(f"No cloudpickle fallback file found at {cp_path}. Consider re-saving the model with cloudpickle from the training script.")

            if trained_model is None:
                st.error("Unable to load the selected model. Cannot proceed with evaluation.")
            else:
                hard_preds = trained_model.predict(eval_features)
                soft_preds = extract_confidence_scores(trained_model, eval_features)

                acc_score = accuracy_score(eval_labels, hard_preds)
                auc_val = roc_auc_score(eval_labels, soft_preds) if len(np.unique(eval_labels)) == 2 else np.nan
                prec_val = precision_score(eval_labels, hard_preds, zero_division=0)
                rec_val = recall_score(eval_labels, hard_preds, zero_division=0)
                f1_metric = f1_score(eval_labels, hard_preds, zero_division=0)
                matthews_val = matthews_corrcoef(eval_labels, hard_preds)

                st.write(f"## {active_learner} - Performance Insights")

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy Rate", f"{acc_score:.4f}")
                m2.metric("AUC Metric", f"{auc_val:.4f}")
                m3.metric("Precision", f"{prec_val:.4f}")
                m4.metric("Recall", f"{rec_val:.4f}")
                m5.metric("F1-Score", f"{f1_metric:.4f}")
                m6.metric("MCC", f"{matthews_val:.4f}")

                st.divider()

                viz_left, viz_right = st.columns(2)

                with viz_left:
                    st.subheader("Error Distribution Matrix")
                    error_mat = confusion_matrix(eval_labels, hard_preds)
                    fig_obj, ax_obj = plt.subplots()
                    sns.heatmap(error_mat, annot=True, fmt='d', cmap='Oranges', ax=ax_obj)
                    ax_obj.set_xlabel('Predicted Categorization')
                    ax_obj.set_ylabel('Actual Label')
                    st.pyplot(fig_obj)

                with viz_right:
                    st.subheader("Detailed Classification Summary")
                    summary_dict = classification_report(eval_labels, hard_preds, output_dict=True)
                    summary_frame = pd.DataFrame(summary_dict).transpose()
                    st.dataframe(summary_frame.style.background_gradient(cmap='YlOrRd'))

        else:
            st.error(f"Execution Error: Resource {selected_file_path} is missing. Initialize training.")
    else:
        st.warning(f"Validation Warning: Identification field '{TARGET_FIELD}' is absent in the source.")
else:
    st.info("Awaiting CSV data upload containing the target variable for validation.")
    st.write("#### Requirements for Analysis:")
    st.write("- Minimum of 12 descriptive attributes.")
    st.write("- Inclusion of 'TARGET_COL' with binary values (0/1).")

