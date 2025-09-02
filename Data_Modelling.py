import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data
def load_data(filepath):
    """
    Loads, cleans, and preprocesses the German credit data.
    Caches the data to avoid reloading on every interaction.
    """
    df = pd.read_csv(filepath)
    # Drop the redundant index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # --- Data Cleaning ---
    # Handle 'NA' or missing values by filling with the mode (most frequent value)
    for col in ['Saving accounts', 'Checking account']:
        if df[col].isnull().any() or (df[col] == 'NA').any():
            # Replace 'NA' string with actual NaN to be safe
            df[col] = df[col].replace('NA', pd.NA)
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    return df

@st.cache_resource
def train_and_compare_models(df):
    """
    Preprocesses data, trains multiple models, compares their performance,
    and returns the best model along with comparison results.
    """
    df_encoded = df.copy()
    encoders = {}
    
    # Encode categorical features
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
        
    # Define features (X) and target (y)
    X = df_encoded.drop('Risk', axis=1)
    y = df_encoded['Risk']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'SVM': SVC(random_state=42, class_weight='balanced', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
    }
    
    # Train and evaluate models
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Use scaled data for models that benefit from scaling
        if name in ['Logistic Regression', 'SVM', 'KNN']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            trained_models[name] = (model, scaler)  # Store model with scaler
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            trained_models[name] = (model, None)  # Store model without scaler
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC (handle multiclass)
        try:
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                roc_auc = 0
        except:
            roc_auc = 0
        
        # Cross-validation score
        if name in ['Logistic Regression', 'SVM', 'KNN']:
            cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
        else:
            cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        
        model_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_score': cv_score,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    # Find the best model based on F1 score (balanced metric)
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['f1_score'])
    best_model, best_scaler = trained_models[best_model_name]
    
    # Retrain the best model on the full dataset for final use
    if best_model_name in ['Logistic Regression', 'SVM', 'KNN']:
        X_full_scaled = scaler.fit_transform(X)
        best_model.fit(X_full_scaled, y)
    else:
        best_model.fit(X, y)
        best_scaler = None
    
    return best_model, best_scaler, encoders, X.columns, model_results, best_model_name, X_test, y_test

def create_comparison_plots(model_results):
    """Create visualization plots for model comparison"""
    
    # Prepare data for plotting
    models = list(model_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cv_score']
    
    # Create comparison dataframe
    comparison_data = []
    for model in models:
        for metric in metrics:
            comparison_data.append({
                'Model': model,
                'Metric': metric.replace('_', ' ').title(),
                'Score': model_results[model][metric]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    fig = px.bar(comparison_df, x='Model', y='Score', color='Metric', 
                 title='Model Performance Comparison',
                 barmode='group', height=500)
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def create_confusion_matrix_plot(model_results, best_model_name):
    """Create confusion matrix heatmap for the best model"""
    
    cm = model_results[best_model_name]['confusion_matrix']
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Good', 'Bad'],
                    y=['Good', 'Bad'],
                    title=f'Confusion Matrix - {best_model_name}',
                    text_auto=True,
                    aspect="auto")
    
    return fig