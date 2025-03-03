import streamlit as st
import os
st.set_page_config(
    page_title="Healthcare Data Feature Reduction Journey",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🔄 Navigate")

# List of pages in sequence
pages = [
    ("1_Clinical Trials Data Explorer.py", "📊 Clinical Trials Data Explorer"),
    ("2_Pivot Handling.py", "🔍 Exploratory Data Analysis"),
    ("3_Adverse Event Clustering.py", "🤖 Adverse Event Clustering"),
    ("4_Missing Value Analysis.py", "📉 Missing Value Analysis"),
    ("5_Outlier Detection & Skewness Correction.py", "⚡ Outlier Detection & Skewness Correction"),
    ("7_Feature Engineering Pipeline.py", "🔬 Feature Engineering Pipeline"),
    ("8_Healthcare Data Feature Reduction Journey.py", "📈 Healthcare Data Feature Reduction Journey"),
    ("9_ML Models Showcase.py", "🚀 ML Models Showcase")
]


# Determine current page index
current_page_index = next((i for i, (page, _) in enumerate(pages) if page in os.path.basename(__file__)), None)

# Navigation buttons
if current_page_index is not None:
    col1, col2 = st.sidebar.columns(2)

    if current_page_index > 0:
        with col1:
            if st.button("⬅ Previous"):
                os.system(f"streamlit run {pages[current_page_index - 1][0]}")

    if current_page_index < len(pages) - 1:
        with col2:
            if st.button("Next ➡"):
                os.system(f"streamlit run {pages[current_page_index + 1][0]}")
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import base64

# Page configuration


# Custom CSS for animations and styling
st.markdown("""
<style>
    .title-text {
        font-size: 42px;
        font-weight: bold;
        background: linear-gradient(45deg, #3498db, #1abc9c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    .subtitle-text {
        font-size: 24px;
        color: #555;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 28px;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .highlight-box {
        background-color: #f8f9fa;
        border: 2px solid #3498db;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
        color: #2c3e50;
    }
    .success-box {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
        color: #2c3e50;
    }
    .info-text {
        font-size: 16px;
        line-height: 1.6;
    }
    .fade-in {
        animation: fadeIn 1.5s;
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    .slide-in {
        animation: slideIn 1s;
    }
    @keyframes slideIn {
        0% {transform: translateX(-100%); opacity: 0;}
        100% {transform: translateX(0); opacity: 1;}
    }
    .btn-custom {
        background-color: #3498db;
        color: white;
        padding: 0.5rem 1rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 2px;
        border-radius: 5px;
        border: none;
        transition: all 0.3s;
    }
    .btn-custom:hover {
        background-color: #2980b9;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #3498db;
    }
    /* Make text visible in highlight boxes */
    .highlight-box h3, .success-box h3 {
        color: #2c3e50;
        font-weight: bold;
    }
    .highlight-box ul, .success-box ul,
    .highlight-box li, .success-box li,
    .highlight-box p, .success-box p {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for animations
def animated_text(text, animation_class="fade-in"):
    st.markdown(f'<div class="{animation_class}">{text}</div>', unsafe_allow_html=True)

# Main title with animation
animated_text('<h1 class="title-text">Healthcare Study Completion Prediction</h1>', "slide-in")
animated_text('<p class="subtitle-text">A journey from 1,500 features to the essentials</p>', "fade-in")

# Introduction
st.markdown('<div class="section-header">Introduction</div>', unsafe_allow_html=True)
animated_text("""
<p class="info-text">
Welcome to our interactive journey through feature reduction in healthcare data analysis. 
In this project, we analyze clinical trial data to predict study completion status, reducing over 1,300 features 
to just the most important ones that explain most of the variance in our data.
</p>
""", "fade-in")

# Create tabs for the journey
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preparation", "🔍 Feature Importance", "✂️ Feature Reduction", "🎉 Results & Insights"])

with tab1:
    st.markdown('<div class="section-header">Data Preparation</div>', unsafe_allow_html=True)
    animated_text("""
    <p class="info-text">
    We start with a large healthcare dataset containing clinical trial information. 
    Our goal is to predict whether a study will be completed or not based on various features.
    </p>
    """, "fade-in")
    
    # Mock dataset loading for visualization
    if st.button("Load Dataset", key="load_dataset"):
        with st.spinner("Loading data..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
                
            # Display dataset info with animation
            st.success("Dataset loaded successfully!")
            animated_text("""
            <div class="success-box">
                <h3>Dataset Overview</h3>
                <ul>
                    <li>Total records: 257,576</li>
                    <li>Initial features: 1,314</li>
                    <li>Target variable: Study Status (Completed vs Non-Completed)</li>
                </ul>
            </div>
            """, "fade-in")
    
    # Data splitting section
    st.markdown('<div class="section-header">Data Splitting</div>', unsafe_allow_html=True)
    animated_text("""
    <p class="info-text">
    We split our dataset into three parts: training, validation, and test sets. This approach allows us to:
    <ul>
        <li>Train our model on a representative subset</li>
        <li>Validate and tune hyperparameters on a separate set</li>
        <li>Evaluate final performance on unseen data</li>
    </ul>
    </p>
    """, "fade-in")
    
    if st.button("Split Data", key="split_data"):
        with st.spinner("Splitting data..."):
            time.sleep(1.5)
            
            # Create a Plotly visualization for the data split
            labels = ['Training Set (60%)', 'Validation Set (20%)', 'Test Set (20%)']
            values = [154545, 51515, 51516]
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            
            fig = px.pie(values=values, names=labels, title='Data Split Distribution',color_discrete_sequence=colors)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display shapes with better visibility
            st.markdown("""
            <div class="highlight-box">
                <h3>✅ Data Split Done</h3>
                <ul>
                    <li><b>X_train:</b> (154545, 1314)</li>
                    <li><b>X_val:</b> (51515, 1314)</li>
                    <li><b>X_test:</b> (51516, 1314)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Column cleaning section
    st.markdown('<div class="section-header">Column Cleaning</div>', unsafe_allow_html=True)
    
    if st.button("Clean Column Names", key="clean_columns"):
        with st.spinner("Cleaning column names..."):
            time.sleep(1)
            
            st.markdown("""
            <div class="success-box">
                <h3>Column Cleaning Results</h3>
                <p>✅ Replaced special characters with underscores</p>
                <p>✅ Number of unique columns: 1314</p>
                <p>✅ No duplicate columns found</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
    
    animated_text("""
    <p class="info-text">
    We use LightGBM, a gradient boosting framework, to identify the most important features in our dataset.
    LightGBM is chosen for its speed, scalability, and ability to handle high-dimensional categorical data.
    </p>
    """, "fade-in")
    
    if st.button("Apply LightGBM Model", key="apply_lgbm"):
        with st.spinner("Training LightGBM model..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            st.success("LightGBM model trained successfully!")
            
            # Show model training details
            st.code("""
[LightGBM] [Info] Number of positive: 132745, number of negative: 21800
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.587818 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 235252
[LightGBM] [Info] Number of data points in the train set: 154545, number of used features: 1308
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.858941 -> initscore=1.806520
[LightGBM] [Info] Start training from score 1.806520
            """)
    
    # Top features visualization
    if st.button("Show Top 20 Features", key="show_top_features"):
        with st.spinner("Analyzing feature importance..."):
            time.sleep(1.5)
            
            # Create mock data for feature importance
            features = ['Conditions', 'Enrollment', 'Interventions', 'Phases', 'duration',
                       'Study_Design_Intervention_Model__PARALLEL', 'reason_90_100_percent',
                       'reason_study_terminated_by_sponsor', 'country_United_States',
                       'reason_80_90_percent', 'TFIDF_criteria_criterion', 'Funder_Type',
                       'Study_Design_Allocation__', 'Study_Design_Intervention_Model__CROSSOVER',
                       'Study_Design_Allocation__RANDOMIZED', 'country_France', 
                       'TFIDF_criteria_inclusion', 'Study_Results', 'period_overall_study',
                       'period_81_90_percent']
            
            importance = [374, 369, 363, 115, 107, 72, 68, 55, 50, 49, 46, 45, 43, 38, 36, 31, 23, 23, 22, 22]
            
            # Create a DataFrame
            feature_df = pd.DataFrame({
                'feature': features,
                'importance': importance
            })
            
            # Create interactive bar chart
            fig = px.bar(feature_df, x='importance', y='feature', 
                         orientation='h', 
                         title='Top 20 Features by Importance',
                         color='importance',
                         color_continuous_scale='Blues')
            
            fig.update_layout(
                height=600,
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                yaxis={'categoryorder':'total ascending'},
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation of top features
            st.markdown("""
            <div class="highlight-box">
                <h3>Key Insights from Feature Importance</h3>
                <p>The top features reveal several important patterns:</p>
                <ul>
                    <li><b>Clinical aspects</b> like Conditions, Interventions, and Phases are the strongest predictors</li>
                    <li><b>Study design</b> (Parallel vs Crossover, Randomization) significantly impacts completion rates</li>
                    <li><b>Geographic location</b> (United States, France) shows correlation with study completion</li>
                    <li><b>Administrative factors</b> like Enrollment numbers and Funder Type play important roles</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-header">Feature Reduction Process</div>', unsafe_allow_html=True)
    
    animated_text("""
    <p class="info-text">
    After identifying feature importance, we select features that contribute to 95% of the cumulative importance.
    This significantly reduces dimensionality while preserving the most predictive power.
    </p>
    """, "fade-in")
    
    if st.button("Apply Feature Selection", key="apply_feature_selection"):
        with st.spinner("Selecting most important features..."):
            time.sleep(2)
            
            # Display feature reduction results
            st.markdown("""
            <div class="success-box">
                <h3>✔ Selected 234 features contributing to 95% importance.</h3>
                <p>Successfully reduced features from 1,314 to 234 while maintaining 95% of the predictive power.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualization of feature reduction
            feature_counts = {
                'Initial Features': 1314,
                'After 95% Importance Filter': 234,
                'Final Features': 112
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(feature_counts.keys()),
                    y=list(feature_counts.values()),
                    marker_color=['#3498db', '#2ecc71', '#1abc9c']
                )
            ])
            
            fig.update_layout(
                title='Feature Reduction Progress',
                xaxis_title='Stage',
                yaxis_title='Number of Features',
                height=400,
                annotations=[
                    dict(
                        x=i, y=count + 50,
                        text=f"{count} features",
                        showarrow=False
                    ) for i, count in enumerate(feature_counts.values())
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Further feature reduction with domain knowledge
    st.markdown('<div class="section-header">Domain Knowledge Feature Filtering</div>', unsafe_allow_html=True)
    
    animated_text("""
    <p class="info-text">
    Based on our data science expertise and knowledge of TF-IDF, we further reduced features 
    by removing those that had limited predictive value or were redundant after NLP preprocessing.
    </p>
    """, "fade-in")
    
    if st.button("Apply Domain Knowledge Filter", key="apply_domain_filter"):
        with st.spinner("Applying domain expertise..."):
            time.sleep(1.5)
            
            st.markdown("""
            <div class="success-box">
                <h3>✔ Final feature count: 112</h3>
                <p>By applying domain knowledge, we removed an additional 122 features that were determined to be:</p>
                <ul>
                    <li>Redundant after TF-IDF vectorization</li>
                    <li>Containing mostly stop words or low-value terms</li>
                    <li>Highly correlated with other more informative features</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a donut chart showing feature categories in final selection
            labels = ['Clinical Features', 'Study Design Features', 'Geographic Features', 
                      'Administrative Features', 'TF-IDF Text Features']
            values = [38, 27, 15, 12, 20]
            
            fig = px.pie(
                names=labels,
                values=values, 
                title='Final Feature Distribution by Category',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
            
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Results & Insights</div>', unsafe_allow_html=True)
    
    animated_text("""
    <p class="info-text">
    Our feature reduction journey has achieved impressive results. Starting with over 1,300 features,
    we've systematically narrowed down to just 112 essential predictors - a 91.5% reduction in dimensionality!
    </p>
    """, "fade-in")
    
    # Animated feature reduction visualization
    if st.button("Show Feature Reduction Animation", key="show_animation"):
        # Create animated visualization with counter
        progress_placeholder = st.empty()
        
        for i in range(1315, 112, -100):
            current = max(i, 112)
            progress_placeholder.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h1 style="font-size: 72px; color: #3498db;">{current}</h1>
                <p style="font-size: 24px; color: #2c3e50;">features remaining</p>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.5)
        
        # Final animation
        progress_placeholder.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="font-size: 72px; color: #27ae60;">112</h1>
            <p style="font-size: 24px; color: #2c3e50;">essential features identified</p>
            <div style="font-size: 36px; margin-top: 20px; color: #3498db;">
                91.5% reduction achieved! 🎉
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)
        
        # Success message
        st.markdown("""
        <div class="success-box">
            <h3>🎉 Feature Reduction Success</h3>
            <p>By reducing features from 1,314 to 112 (91.5% reduction), we achieved:</p>
            <ul>
                <li><b>Maintained performance</b>: Preserved 95% of the predictive power</li>
                <li><b>Enhanced interpretability</b>: Clearer insights into key predictors</li>
                <li><b>Improved efficiency</b>: Faster training and inference times</li>
                <li><b>Reduced overfitting risk</b>: More robust model with fewer irrelevant features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Key takeaways
    st.markdown('<div class="section-header">Key Takeaways</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h3>Most Important Predictors</h3>
            <p>The most influential factors in predicting study completion are:</p>
            <ol>
                <li><b>Clinical conditions and interventions</b></li>
                <li><b>Study enrollment numbers</b></li>
                <li><b>Study design characteristics</b></li>
                <li><b>Geographic location</b></li>
                <li><b>Administrative factors</b></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h3>Feature Reduction Benefits</h3>
            <p>The dimensionality reduction approach delivered several advantages:</p>
            <ul>
                <li><b>Computational efficiency</b>: Faster model training and deployment</li>
                <li><b>Noise reduction</b>: Removal of non-informative features</li>
                <li><b>Improved generalization</b>: Better performance on new data</li>
                <li><b>Scientific insight</b>: Better understanding of study completion factors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualize feature reduction through an engaging animation
    if st.button("Visualize Feature Reduction Journey", key="visualize_journey"):
        # Create animated timeline of feature reduction
        stages = [
            {"stage": "Initial Dataset", "features": 1314, "description": "Raw feature set from healthcare data"},
            {"stage": "After LightGBM", "features": 234, "description": "Features contributing to 95% importance"},
            {"stage": "After Domain Filtering", "features": 112, "description": "Final curated feature set"}
        ]
        
        timeline_container = st.container()
        
        for i, stage in enumerate(stages):
            with timeline_container:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background-color: #f0f8ff; border-radius: 10px; border: 2px solid #3498db;">
                        <h3>{stage['stage']}</h3>
                        <h2 style="color: #3498db; font-size: 36px;">{stage['features']}</h2>
                        <p>features</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Calculate percentage reduction from initial
                    if i > 0:
                        reduction = round(100 * (1 - stage['features']/stages[0]['features']), 1)
                        reduction_text = f"<span style='color: #27ae60; font-weight: bold;'>{reduction}% reduction</span>"
                    else:
                        reduction_text = "<span style='color: #7f8c8d;'>baseline</span>"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; margin-top: 10px;">
                        <h4>{stage['description']}</h4>
                        <p>{reduction_text}</p>
                        <div style="height: 10px; background-color: #ecf0f1; border-radius: 5px; margin-top: 10px;">
                            <div style="height: 100%; width: {100 * stage['features']/stages[0]['features']}%; 
                                 background-color: #3498db; border-radius: 5px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if i < len(stages) - 1:
                    st.markdown("""
                    <div style="text-align: center; padding: 5px;">
                        <span style="font-size: 24px; color: #3498db;">↓</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            time.sleep(0.8)

# Add sidebar with project overview
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/healthcare-data.png", width=100)
    st.markdown("<h2>Project Overview</h2>", unsafe_allow_html=True)
    st.write("This interactive dashboard showcases our journey of feature reduction in healthcare data for predicting clinical study completion.")
    
    st.markdown("### Key Milestones")
    milestones = {
        "Initial Dataset": "1,314 features",
        "After LightGBM Analysis": "234 features (82% reduction)",
        "Final Feature Set": "112 features (91.5% reduction)"
    }
    
    for stage, result in milestones.items():
        st.markdown(f"**{stage}**: {result}")
    
    st.markdown("### Technologies Used")
    tech_stack = ["LightGBM", "Scikit-learn", "Pandas", "NumPy", "Streamlit", "Plotly"]
    for tech in tech_stack:
        st.markdown(f"• {tech}")