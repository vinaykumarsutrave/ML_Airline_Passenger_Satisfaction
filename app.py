import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

st.set_page_config(page_title="Airline Satisfaction Classifier", layout="wide")

st.title("✈️ Airline Passenger Satisfaction Classifier")
st.write("Upload test dataset, select a trained model, and evaluate model performance metrics and visualizations.")

# Model selection mapping
MODEL_FILES = {
    "Logistic Regression": "saved_models/logistic_regression.joblib",
    "Decision Tree Classifier": "saved_models/decision_tree.joblib",
    "K-Nearest Neighbor Classifier": "saved_models/k-nearest_neighbors.joblib",
    "Naive Bayes Classifier": "saved_models/naive_bayes.joblib",
    "Support Vector Machine": "saved_models/support_vector_machine.joblib"
}

# Sidebar inputs
st.sidebar.header("User Controls")
selected_model_name = st.sidebar.selectbox("Select Classification Model", list(MODEL_FILES.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv", "xlsx"])

test_button = st.sidebar.button("Test", disabled=(uploaded_file is None))

def preprocess_data(df):
    # Data Cleaning: Impute missing numerical values with Median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Drop identifier column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # Feature Encoding: Target Column
    if 'satisfaction' in df.columns:
        le_target = LabelEncoder()
        df['satisfaction'] = le_target.fit_transform(df['satisfaction'])
    else:
        st.error("Target column 'satisfaction' not found in dataset!")
        return None, None
        
    # Feature Encoding: Categorical Columns
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    existing_cat_cols = [c for c in categorical_cols if c in df.columns]
    
    df_encoded = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)
    
    X = df_encoded.drop(columns=['satisfaction'])
    y = df_encoded['satisfaction']
    
    # Feature Scaling: StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, df

if test_button and uploaded_file is not None:
    st.header(f"Results for Model: {selected_model_name}")
    
    # Load Data
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
            
        st.subheader("Data Exploration & Preview")
        st.dataframe(df_raw.head())
        
        col_shape1, col_shape2 = st.columns(2)
        col_shape1.metric("Total Rows", df_raw.shape[0])
        col_shape2.metric("Total Columns", df_raw.shape[1])
        
        # Preprocessing
        X_test_scaled, y_test, df_processed = preprocess_data(df_raw)
        
        if X_test_scaled is not None:
            # Load Selected Model
            model_path = MODEL_FILES[selected_model_name]
            if not os.path.exists(model_path):
                st.error(f"Model file not found at path: {model_path}. Please train and save the model first.")
            else:
                model = joblib.load(model_path)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_proba = model.decision_function(X_test_scaled)
                else:
                    y_proba = y_pred

                # Calculate Evaluation Metrics
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)

                # Display Metrics
                st.subheader("📊 Evaluation Metrics")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{acc:.4f}")
                m2.metric("AUC Score", f"{auc:.4f}")
                m3.metric("Precision", f"{prec:.4f}")
                m4.metric("Recall", f"{rec:.4f}")
                m5.metric("F1 Score", f"{f1:.4f}")
                m6.metric("MCC Score", f"{mcc:.4f}")

                # Display Graphs
                st.subheader("📈 Model Performance Visualizations")
                fig_col1, fig_col2 = st.columns(2)

                with fig_col1:
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                xticklabels=['Unsatisfied', 'Satisfied'],
                                yticklabels=['Unsatisfied', 'Satisfied'])
                    ax_cm.set_xlabel('Predicted Label')
                    ax_cm.set_ylabel('True Label')
                    st.pyplot(fig_cm)

                with fig_col2:
                    st.write("**ROC Curve**")
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
                    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.set_title('Receiver Operating Characteristic')
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc)

    except Exception as e:
        st.error(f"Error processing file: {e}")
