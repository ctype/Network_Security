import os
import sys
import pandas as pd
import streamlit as st
import certifi
import pymongo
import plotly.express as px
from datetime import datetime

from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import load_object

# Load Environment Variables
load_dotenv()
mongo_db_url = os.getenv('MONGO_DB_URL')
ca = certifi.where()

# --- Page Configuration ---
st.set_page_config(
    page_title="SafeNet ML | Enterprise Phishing Detection",
    page_icon="🛡️",
    layout="wide"
)

# --- Session State Initialization ---
# This prevents the app from losing data when the user interacts with widgets
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .stAlert { border-radius: 10px; }
    div.stButton > button:first-child { background-color: #238636; color: white; width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🛡️ SafeNet AI v2.0")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", ["Dashboard", "System Training", "Feature Glossary"])

# --- FEATURE GLOSSARY ---
if menu == "Feature Glossary":
    st.header("🔍 Network Feature Encyclopedia")
    st.write("The model classifies traffic based on 30 distinct attributes defined in `schema.yaml`.")
    
    # Organized by category
    tabs = st.tabs(["URL Structure", "Identity & Trust", "Traffic Metrics"])
    
    with tabs[0]:
        st.info("**URL_Length**: Long URLs often hide malicious redirects.\n\n**having_At_Symbol**: Using '@' redirects browsers to ignore the preceding string.\n\n**Prefix_Suffix**: Legitimate brands rarely use dashes (e.g., `amazon-login.com`).")
    with tabs[1]:
        st.info("**SSLfinal_State**: Evaluates HTTPS certificate age and trust.\n\n**Domain_registration_length**: Phishing domains are usually registered for < 1 year.\n\n**SFH**: Checks if form handlers lead to blank pages or different domains.")
    with tabs[2]:
        st.info("**web_traffic**: High Alexa/PageRank scores suggest a site is established.\n\n**Google_Index**: Phishing sites often haven't been indexed by search engines yet.")

# --- SYSTEM TRAINING ---
elif menu == "System Training":
    st.header("⚙️ Machine Learning Pipeline Control")
    st.markdown("Re-train the model artifacts using the latest data from MongoDB.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Current Pipeline Status")
        st.code("Ingestion: OK\nValidation: OK\nTransformation: OK\nTrainer: READY")
        
    if st.button("🚀 Trigger Full Re-training"):
        try:
            with st.status("Pipeline in progress...", expanded=True) as status:
                st.write("Step 1: Data Ingestion from MongoDB...")
                training_pipeline = TrainingPipeline()
                training_pipeline.run_pipeline()
                status.update(label="Training Complete! Artifacts Saved.", state="complete")
            st.success("The preprocessor and model have been updated.")
        except Exception as e:
            st.error(f"Critical Failure: {e}")

# --- DASHBOARD (INFERENCE) ---
else:
    st.header("📊 Enterprise Threat Dashboard")
    
    # 1. File Upload
    uploaded_file = st.file_uploader("Upload Network Traffic Logs (CSV)", type="csv")
    
    if uploaded_file:
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🛡️ Run Scan"):
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    with st.spinner("Analyzing traffic patterns..."):
                        # Load Model from your Pipeline output folder
                        preprocessor = load_object("final_models/preprocessor.pkl")
                        model = load_object("final_models/model.pkl")
                        network_model = NetworkModel(processor=preprocessor, model=model)
                        
                        # Generate Predictions
                        df['prediction'] = network_model.predict(df)
                        df['Status'] = df['prediction'].map({1.0: "✅ Legitimate", 0.0: "🚨 Phishing", -1.0: "🚨 Phishing"})
                        
                        # Calculate Risk Score (Logic: % of features that are -1)
                        feat_cols = [c for c in df.columns if c not in ['prediction', 'Status']]
                        df['Risk_Score'] = (df[feat_cols] == -1).sum(axis=1) / len(feat_cols) * 100
                        
                        # Store in session
                        st.session_state.prediction_data = df
                        
                except Exception as e:
                    st.error(f"Scan Error: {e}")

    # 2. Results Display
    if st.session_state.prediction_data is not None:
        df = st.session_state.prediction_data
        
        # Metrics
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        total = len(df)
        phish = (df['prediction'] != 1.0).sum()
        
        m1.metric("Batch Size", total)
        m2.metric("Threats Detected", phish, delta=f"{phish/total:.1%}", delta_color="inverse")
        m3.metric("Avg Risk Intensity", f"{df['Risk_Score'].mean():.1f}%")
        m4.metric("Security Level", "HIGH" if phish < total*0.1 else "CRITICAL")

        # Visualizations
        st.write("---")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            fig_pie = px.pie(df, names='Status', hole=0.5,
                            color='Status', color_discrete_map={'✅ Legitimate':'#238636','🚨 Phishing':'#da3633'},
                            title="Threat Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            # Logic to find "Red Flag" features in this specific dataset
            if phish > 0:
                phish_only = df[df['prediction'] != 1.0]
                red_flags = (phish_only[feat_cols] == -1).sum().sort_values(ascending=False).head(10)
                fig_bar = px.bar(x=red_flags.values, y=red_flags.index, orientation='h',
                                title="Top Vulnerability Indicators Found",
                                labels={'x':'Incidents','y':'Feature'},
                                color_discrete_sequence=['#ff4b4b'])
                st.plotly_chart(fig_bar, use_container_width=True)

        # 3. Table and Export
        st.subheader("📋 Forensic Audit Log")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forensic Report", csv, "security_scan.csv", "text/csv")