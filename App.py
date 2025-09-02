import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from Data_Modelling import load_data, train_and_compare_models, create_comparison_plots, create_confusion_matrix_plot

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data and Train Models ---
with st.spinner('Loading data and training models...'):
    df_credit = load_data('german_credit_data.csv')
    (best_model, best_scaler, encoders, feature_names, 
     model_results, best_model_name, X_test, y_test) = train_and_compare_models(df_credit)

# --- Main Panel Display ---
st.title("📊 German Credit Risk Analysis Dashboard")
st.markdown(f"""
Welcome to the Credit Risk Prediction dashboard. This app analyzes customer data to predict credit risk using **{best_model_name}** 
as the best performing model. You can explore the dataset, visualize key metrics, and use the sidebar to get a risk prediction for a new customer profile.
""")
st.markdown("---")

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Credit Risk", 
    "📊 Data Visualizations & Explorer", 
    "🤖 Model Comparison & Performance"
])

# --- Tab 1: User Input & Prediction ---
with tab1:
    st.header("🔮 Predict Credit Risk")
    st.write("Adjust the sliders and dropdowns to match a customer's profile.")
    st.info(f"🏆 Using **{best_model_name}** (Best Performing Model)")

    def user_input_features():
        """Collects user input from the tab."""
        inputs = {}
        for col in df_credit.columns:
            if col == 'Risk':
                continue
            if pd.api.types.is_numeric_dtype(df_credit[col]):
                min_val = int(df_credit[col].min())
                max_val = int(df_credit[col].max())
                default_val = int(df_credit[col].mean())
                # Use a smaller width and center the input
                with st.container():
                    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                    inputs[col] = st.slider(
                        f'Enter {col}', min_val, max_val, default_val, key=col, label_visibility="visible"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                options = df_credit[col].unique().tolist()
                with st.container():
                    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                    inputs[col] = st.selectbox(
                        f'Select {col}', options, key=col, label_visibility="visible"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
        return pd.DataFrame([inputs])

    # Center the input form
    input_col1, input_col2, input_col3 = st.columns([1,3,1])
    with input_col2:
        input_df = user_input_features()
        predict_btn = st.button("Predict Credit Risk", use_container_width=True)

    # --- Prediction Logic ---
    if predict_btn:
        input_df_encoded = input_df.copy()
        for col in input_df_encoded.columns:
            if col in encoders:
                le = encoders[col]
                input_df_encoded[col] = input_df_encoded[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                input_df_encoded[col] = le.transform(input_df_encoded[col])
        input_df_encoded = input_df_encoded[feature_names]

        if best_scaler is not None:
            input_df_scaled = best_scaler.transform(input_df_encoded)
            prediction_encoded = best_model.predict(input_df_scaled)
            prediction_proba = best_model.predict_proba(input_df_scaled)
        else:
            prediction_encoded = best_model.predict(input_df_encoded)
            prediction_proba = best_model.predict_proba(input_df_encoded)

        risk_le = encoders['Risk']
        prediction = risk_le.inverse_transform(prediction_encoded)
    else:
        prediction = ['good']
        prediction_proba = [[0.0, 0.0]]
        risk_le = encoders['Risk']

    st.header("👤 Customer Profile Prediction")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Prediction Result:")
        if prediction[0] == 'good':
            st.success(f"✅ Risk: **Good**")
        else:
            st.error(f"⚠️ Risk: **Bad**")

        st.subheader("Prediction Confidence:")
        good_prob = prediction_proba[0][list(risk_le.classes_).index('good')]
        bad_prob = prediction_proba[0][list(risk_le.classes_).index('bad')]
        st.info(f"**Good Risk:** {good_prob:.2%}")
        st.info(f"**Bad Risk:** {bad_prob:.2%}")

    with col2:
        # Center the subheader using markdown and HTML
        st.markdown("<h3 style='text-align: center;'>Good Credit Risk Probability</h3>", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = good_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probability (%)"},
            delta = {'reference': 70},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

# --- Tab 2: Data Visualizations & Explorer ---
with tab2:
    st.header("Key Data Visualizations")
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.subheader("Risk Distribution")
        risk_counts = df_credit['Risk'].value_counts()
        fig_pie = px.pie(risk_counts, values=risk_counts.values, names=risk_counts.index, 
                         title='Overall Credit Risk Distribution', hole=0.3,
                         color_discrete_map={'good':'#4CAF50', 'bad':'#F44336'})
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Credit Amount by Risk")
        fig_box = px.box(df_credit, x='Risk', y='Credit amount', color='Risk',
                         title='Credit Amount vs. Risk Status',
                         color_discrete_map={'good':'#4CAF50', 'bad':'#F44336'})
        st.plotly_chart(fig_box, use_container_width=True)

    with viz_col2:
        st.subheader("Age Distribution")
        fig_hist = px.histogram(df_credit, x='Age', nbins=20, color='Risk',
                                title='Age Distribution by Risk',
                                color_discrete_map={'good':'#4CAF50', 'bad':'#F44336'})
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Purpose of Credit")
        fig_bar = px.bar(df_credit, y='Purpose', color='Risk',
                         title='Credit Purpose by Risk', barmode='stack',
                         color_discrete_map={'good':'#4CAF50', 'bad':'#F44336'},
                         height=400)
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.header("Explore the Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", df_credit.shape[0])
    with col2:
        st.metric("Total Features", df_credit.shape[1])
    with col3:
        st.metric("Good Risk %", f"{(df_credit['Risk'] == 'good').mean():.1%}")

    st.dataframe(df_credit, use_container_width=True)
    st.subheader("Dataset Summary Statistics")
    st.dataframe(df_credit.describe(), use_container_width=True)

# --- Tab 3: Model Comparison & Performance ---
with tab3:
    st.header("🤖 Model Comparison Results")
    st.subheader(f"🏆 Best Model: {best_model_name}")

    comparison_df = pd.DataFrame(model_results).T
    comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cv_score']]
    comparison_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'CV Score']

    def highlight_best(s):
        is_best = s.name == best_model_name
        return ['background-color: lightgreen' if is_best else '' for _ in s]

    st.dataframe(comparison_df.style.apply(highlight_best, axis=1).format("{:.4f}"), use_container_width=True)

    st.subheader("Model Performance Comparison")
    comparison_fig = create_comparison_plots(model_results)
    st.plotly_chart(comparison_fig, use_container_width=True)

    st.subheader(f"Confusion Matrix - {best_model_name}")
    cm_fig = create_confusion_matrix_plot(model_results, best_model_name)
    st.plotly_chart(cm_fig, use_container_width=True)

    st.header("📈 Detailed Model Performance")
    selected_model = st.selectbox("Select a model to view detailed performance:", list(model_results.keys()))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{selected_model} - Performance Metrics")
        metrics = model_results[selected_model]
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            st.metric("Precision", f"{metrics['precision']:.4f}")
            st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
        with metric_col2:
            st.metric("Recall", f"{metrics['recall']:.4f}")
            st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
            st.metric("CV Score", f"{metrics['cv_score']:.4f}")

    with col2:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    if selected_model in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
        st.subheader(f"Feature Importance - {selected_model}")
        if selected_model == best_model_name:
            model_for_importance = best_model
        else:
            st.info("Feature importance is only available for the best model in this implementation.")
            model_for_importance = best_model if selected_model == best_model_name else None

        if model_for_importance and hasattr(model_for_importance, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model_for_importance.feature_importances_
            }).sort_values('importance', ascending=False)

            fig_importance = px.bar(feature_importance.head(10), x='importance', y='feature', 
                                    orientation='h', title=f'Top 10 Feature Importance - {selected_model}',
                                    height=500)
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **About the Models:**
    - **Random Forest**: Ensemble of decision trees with voting
    - **Gradient Boosting**: Sequential ensemble with error correction
    - **Logistic Regression**: Linear model with logistic function
    - **SVM**: Support Vector Machine with RBF kernel
    - **KNN**: K-Nearest Neighbors classifier
    - **Decision Tree**: Single decision tree model

    The best model is automatically selected based on F1-score for balanced performance.
    """)
