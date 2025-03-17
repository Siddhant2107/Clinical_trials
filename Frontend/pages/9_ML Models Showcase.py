import streamlit as st
import os
st.set_page_config(
    page_title="ML Models Showcase",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.sidebar.title("🔄 Navigate")

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
import pickle
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
import json

# Page Configuration

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .model-header {
        font-size: 2rem;
        color: #1565C0;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: 500;
        background-color: #808080;
        padding: 1rem;
        border-radius: 5px;
    }
    .metrics-container {
        background-color: #2F4F4F;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #EEEEEE;
    }
    .metric-value {
        font-weight: 500;
        color: #1565C0;
    }
    .fact-box {
        background-color: #556B25;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        border-left: 4px solid #4CAF50;
    }
    .report-container {
        background-color: #FAFAFA;
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1rem;
        font-family: monospace;
        white-space: pre-wrap;
        overflow: auto;
        max-height: 300px;
    }
    .btn-primary {
        background-color: #1976D2;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        margin-right: 0.5rem;
        cursor: pointer;
    }
    .btn-secondary {
        background-color: #90CAF9;
        color: #0D47A1;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        margin-right: 0.5rem;
        cursor: pointer;
    }
    .stButton>button {
        background-color: #1976D2;
        color: white;
        font-weight: 500;
    }
    .info-box {
        background-color: #556B25;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .footnote {
        font-size: 0.8rem;
        color: #757575;
        margin-top: 2rem;
        text-align: center;
    }
    .animated-box {
        animation: fadeIn 1s;
        
        
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Functions to load lottie animations
# Function to load Lottie animations with error handling
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            st.warning(f"Failed to load animation from {url}. Status Code: {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"Error fetching animation from {url}: {e}")
        return None

# Load animations
lottie_ml = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_data = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_loading = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_p8bfn5to.json")

# Helper functions
def load_image(image_path):
    try:
        return Image.open(image_path)
    except:
        st.error(f"Could not load image from {image_path}. Using placeholder instead.")
        return None

# Simulated data loading (replace with actual file paths)
def get_image_path(file_name):
    # Replace this with the correct path for your environment
    base_path = r"C:\Users\Siddhant Nijhawan\Downloads\Nest images of model webpage"
    return os.path.join(base_path, file_name)

# Main app structure
def main():
    # Sidebar for navigation
    st.sidebar.title("Model Selection")
    
    # Display lottie animation in sidebar
    with st.sidebar:
        st_lottie(lottie_ml, height=200)
    
    # Model selection
    model_options = [
        "Home", 
        "Logistic Regression", 
        "Random Forest", 
        "LightGBM", 
        "XGBoost"
    ]
    
    selected_model = st.sidebar.radio("Select a model to explore:", model_options)
    
    # Home page
    if selected_model == "Home":
        display_home()
    elif selected_model == "Logistic Regression":
        display_logistic_regression()
    elif selected_model == "Random Forest":
        display_random_forest()
    elif selected_model == "LightGBM":
        display_lightgbm()
    elif selected_model == "XGBoost":
        display_xgboost()
    
    # Footer
    st.markdown('<div class="footnote">© 2025 ML Models Showcase. Created with Streamlit.</div>', unsafe_allow_html=True)

def display_home():
    # Animated header with a typing effect (simulated)
    st.markdown('<h1 class="main-header animated-box">Welcome to ML Models Showcase</h1>', unsafe_allow_html=True)
    
    # Main content with animation
    with st.container():
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="animated-box" style="animation-delay: 0.5s;">
                <h2 class="sub-header">Interactive ML Model Exploration</h2>
                <p>This interactive dashboard showcases various machine learning models trained on our dataset. 
                Explore different models, compare their performance metrics, and gain insights into their strengths and weaknesses.</p>
                
            </div> """, unsafe_allow_html=True)
            st.markdown("""
            <div class="animated-box" style="animation-delay: 1.2s;">
                <h3>Key Features:</h3>
                <ul>
                    <li><strong>Interactive visualization of model metrics</strong></li>
                    <li><strong>Pre-computed results for quick analysis</strong></li>
                    <li><strong>Model explainability with LIME</strong></li>
                    <li><strong>Direct comparison of model performance</strong></li>
                </ul>
                
            </div>""", unsafe_allow_html=True)
            
            
        
        with col2:
            st_lottie(lottie_data, height=300)
    
    # Dataset information
    st.markdown('<h2 class="sub-header animated-box" style="animation-delay: 1s;">About the Dataset</h2>', unsafe_allow_html=True)
    
    with st.expander("Dataset Details", expanded=True):
        st.markdown("""
        <div class="animated-box" style="animation-delay: 1.2s;">
            <p>For all models, we've divided our dataset into three parts:</p>
            <ul>
                <li><strong>Training Set:</strong> Used to train the models</li>
                <li><strong>Validation Set:</strong> Used for hyperparameter tuning</li>
                <li><strong>Test Set:</strong> Used to evaluate final model performance</li>
            </ul>
            <p>This approach ensures that our models are robust and generalizable to new data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Showcase preview
    st.markdown('<h2 class="sub-header animated-box" style="animation-delay: 1.5s;">Models Showcase</h2>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="animated-box" style="animation-delay: 1.7s; text-align: center;">
                <h3>Logistic Regression</h3>
                <p>Simple yet powerful linear model for classification tasks.</p>
                <p class="metric-value">AUC-ROC: 0.9404</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="animated-box" style="animation-delay: 1.9s; text-align: center;">
                <h3>Random Forest</h3>
                <p>Ensemble of decision trees with strong performance.</p>
                <p class="metric-value">AUC-ROC: 0.8831</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="animated-box" style="animation-delay: 2.1s; text-align: center;">
                <h3>LightGBM</h3>
                <p>Gradient boosting framework focusing on efficiency.</p>
                <p class="metric-value">AUC-ROC: 0.8917</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="animated-box" style="animation-delay: 2.3s; text-align: center;">
                <h3>XGBoost</h3>
                <p>Our MVP model with Optuna optimization.</p>
                <p class="metric-value">AUC-ROC: 0.9526</p>
            </div>
            """, unsafe_allow_html=True)

def display_logistic_regression():
    # Header with animation
    st.markdown('<h1 class="model-header animated-box">Logistic Regression Model</h1>', unsafe_allow_html=True)
    
    # Model information and facts (all elements sequential)
    st.markdown("""<div class="animated-box">""", unsafe_allow_html=True)

    # Logistic Regression Information
    st.markdown("""
    <div class="animated-box">
        <p>Logistic regression is a fundamental classification algorithm that models the probability of a binary outcome. 
        Despite its simplicity, it often performs well on many real-world problems and serves as an excellent baseline model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fun facts about Logistic Regression
    st.markdown("""
    <div class="fact-box animated-box">
        <h4>Did you know?</h4>
        <ul>
            <li>Logistic regression was developed by statistician David Cox in 1958</li>
            <li>It's highly interpretable, with coefficients directly related to feature importance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


    # 🔹 Directly Call the Results Display Function Instead of Showing Buttons
    display_logistic_regression_results()

def display_logistic_regression_results():
    st.markdown('<h3 class="sub-header animated-box">Model Performance</h3>', unsafe_allow_html=True)

    # Results container (all elements sequential)
    st.markdown("""<div class="animated-box">""", unsafe_allow_html=True)

    # Performance Metrics
    st.markdown("""
    <div class="metrics-container animated-box">
        <h4>Performance Metrics:</h4>
        <p>🔹 <span class="metric-value">Accuracy: 0.8924</span></p>
        <p>🔹 <span class="metric-value">Precision: 0.9705</span></p>
        <p>🔹 <span class="metric-value">Recall: 0.9021</span></p>
        <p>🔹 <span class="metric-value">F1 Score: 0.9351</span></p>
        <p>🔹 <span class="metric-value">AUC-ROC: 0.9404</span></p>
    </div>
    """, unsafe_allow_html=True)

    # Classification Report - Improved Table Format
    st.markdown('<h4>📊 Classification Report</h4>', unsafe_allow_html=True)
    
    # Creating the classification report as a dataframe
    data = {
        "Class": ["0", "1", "Accuracy", "Macro Avg", "Weighted Avg"],
        "Precision": [0.58, 0.97, "", 0.78, 0.92],
        "Recall": [0.83, 0.90, "", 0.87, 0.89],
        "F1-Score": [0.69, 0.94, 0.89, 0.81, 0.90],
        "Support": [7267, 44249, 51516, 51516, 51516]
    }
    df = pd.DataFrame(data)

    table_html = f"""
    <style>
        .styled-table {{
            width: 100%;
            max-width: 800px; /* Keep the width fixed */
            margin: auto;
            border-collapse: collapse;
            font-family: 'Arial', sans-serif;
            font-size: 14px;  /* Adjust font size to avoid excess size */
            text-align: center;
            background-color: #f4f4f4;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
        }}
        .styled-table th, .styled-table td {{
            padding: 8px; /* Adjust padding for a balanced size */
            border: 1px solid #ddd;
        }}
        .styled-table th {{
            background-color: #2c3e50;
            color: white;
            text-transform: uppercase;
            font-weight: bold;
        }}
        .styled-table td {{
            color: black;
        }}
        .styled-table tr:nth-child(even) {{
            background-color: #ebebeb;
        }}
        .styled-table tr:hover {{
            background-color: #d5e6f3;
            transition: 0.3s;
        }}
    </style>
    <table class="styled-table">
        <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
        </tr>
        {''.join(f"<tr><td>{row['Class']}</td><td>{row['Precision']}</td><td>{row['Recall']}</td><td>{row['F1-Score']}</td><td>{row['Support']}</td></tr>" for _, row in df.iterrows())}
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown('<h4>Confusion Matrix:</h4>', unsafe_allow_html=True)
    confusion_matrix = load_image(get_image_path("logistic_confusion.png"))
    if confusion_matrix:
        st.image(confusion_matrix, use_column_width=True)
    else:
        st.info("Confusion matrix visualization not available")

    # ROC Curve
    st.markdown('<h4>ROC Curve:</h4>', unsafe_allow_html=True)
    roc_curve = load_image(get_image_path("logistic_roc.png"))
    if roc_curve:
        st.image(roc_curve, use_column_width=True)
    else:
        st.info("ROC curve visualization not available")

    # LIME Explanation
    st.markdown('<h3 class="sub-header animated-box">Model Explainability (LIME)</h3>', unsafe_allow_html=True)
    
    lime_image = load_image(get_image_path("lime_logistioc.png"))
    if lime_image:
        st.image(lime_image, use_column_width=True)
    else:
        st.info("LIME explanation visualization not available")

    # Interpretation
    st.markdown("""
    <div class="info-box animated-box">
        <h4>Interpretation:</h4>
        <p>The logistic regression model shows strong overall performance with an accuracy of 89.24% and an impressive AUC-ROC score of 0.9404, 
        indicating good discriminative ability between classes. The model achieves high precision (0.9705) for the positive class, 
        meaning it has a low false positive rate. However, it does show some limitations in identifying true negatives as indicated by 
        the lower precision (0.58) for class 0. The LIME explanation helps identify the most influential features for individual predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_random_forest():
    # Header with animation
    st.markdown('<h1 class="model-header animated-box">Random Forest Model</h1>', unsafe_allow_html=True)
    
    # Model information and facts (all elements sequential)
    st.markdown("""<div class="animated-box">""", unsafe_allow_html=True)

    # Random Forest Information
    st.markdown("""
    <div class="animated-box">
        <p>Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training 
        and outputting the class that is the mode of the classes of the individual trees. This approach helps reduce overfitting 
        and improves generalization.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fun facts about Random Forest
    st.markdown("""
    <div class="fact-box animated-box">
        <h4>Did you know?</h4>
        <ul>
            <li>Random Forests were developed by Leo Breiman in 2001</li>
            <li>They perform implicit feature selection and can handle high-dimensional spaces</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 🔹 Directly Call the Results Display Function Instead of Showing Buttons
    display_random_forest_results()



def display_random_forest_results():
    st.markdown('<h3 class="sub-header animated-box">Model Performance</h3>', unsafe_allow_html=True)

    # Results container
    st.markdown("""<div class="animated-box">""", unsafe_allow_html=True)

    # Performance Metrics
    st.markdown("""
    <div class="metrics-container animated-box">
        <h4>Optimized Random Forest Performance Metrics:</h4>
        <p>🔹 <span class="metric-value">Accuracy: 0.9139</span></p>
        <p>🔹 <span class="metric-value">Precision: 0.8324</span></p>
        <p>🔹 <span class="metric-value">Recall: 0.4880</span></p>
        <p>🔹 <span class="metric-value">F1 Score: 0.6153</span></p>
        <p>🔹 <span class="metric-value">AUC-ROC: 0.8831</span></p>
    </div>
    """, unsafe_allow_html=True)

    # Classification Report - Improved Table Format
    st.markdown('<h4>📊 Classification Report (Optimized Random Forest)</h4>', unsafe_allow_html=True)

    # Creating the classification report as a dataframe
    data_rf = {
        "Class": ["0", "1", "Accuracy", "Macro Avg", "Weighted Avg"],
        "Precision": [0.92, 0.83, "", 0.88, 0.91],
        "Recall": [0.98, 0.49, "", 0.74, 0.91],
        "F1-Score": [0.95, 0.62, 0.91, 0.78, 0.90],
        "Support": [44249, 7267, 51516, 51516, 51516]
    }
    df_rf = pd.DataFrame(data_rf)

    table_html_rf = f"""
    <style>
        .styled-table {{
            width: 100%;
            max-width: 800px;
            margin: auto;
            border-collapse: collapse;
            font-family: 'Arial', sans-serif;
            font-size: 14px;
            text-align: center;
            background-color: #f4f4f4;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
        }}
        .styled-table th, .styled-table td {{
            padding: 8px;
            border: 1px solid #ddd;
        }}
        .styled-table th {{
            background-color: #2c3e50;
            color: white;
            text-transform: uppercase;
            font-weight: bold;
        }}
        .styled-table td {{
            color: black;
        }}
        .styled-table tr:nth-child(even) {{
            background-color: #ebebeb;
        }}
        .styled-table tr:hover {{
            background-color: #d5e6f3;
            transition: 0.3s;
        }}
    </style>
    <table class="styled-table">
        <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
        </tr>
        {''.join(f"<tr><td>{row['Class']}</td><td>{row['Precision']}</td><td>{row['Recall']}</td><td>{row['F1-Score']}</td><td>{row['Support']}</td></tr>" for _, row in df_rf.iterrows())}
    </table>
    """
    st.markdown(table_html_rf, unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown('<h4>Confusion Matrix:</h4>', unsafe_allow_html=True)
    confusion_matrix = load_image(get_image_path("random_confusion111.png"))
    if confusion_matrix:
        st.image(confusion_matrix, use_column_width=True)
    else:
        st.info("Confusion matrix visualization not available")

    # ROC Curve
    st.markdown('<h4>ROC Curve:</h4>', unsafe_allow_html=True)
    roc_curve = load_image(get_image_path("random_auc112.png"))
    if roc_curve:
        st.image(roc_curve, use_column_width=True)
    else:
        st.info("ROC curve visualization not available")

    # LIME Explanation
    st.markdown('<h3 class="sub-header animated-box">Model Explainability (LIME)</h3>', unsafe_allow_html=True)

    lime_image = load_image(get_image_path("random_lime.png"))
    if lime_image:
        st.image(lime_image, use_column_width=True)
    else:
        st.info("LIME explanation visualization not available")

    # Interpretation
    st.markdown("""
    <div class="info-box animated-box">
        <h4>Interpretation:</h4>
        <p>The optimized Random Forest model demonstrates strong overall performance with an accuracy of 91.39%. However, it shows a 
        notable imbalance in its predictive capabilities. While it achieves excellent performance for the majority class (class 0) with 
        a precision of 0.92 and recall of 0.98, it struggles somewhat with the minority class (class 1), achieving only 0.49 recall.
        This indicates that while the model rarely misclassifies negative cases, it misses about half of the positive cases. 
        The AUC-ROC score of 0.8831 confirms good but not excellent discriminative ability.</p>
        <p>The LIME explanation provides insight into which features the model relies on most heavily for individual predictions, 
        helping to understand its decision-making process.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def display_lightgbm():
    # Header with animation
    st.markdown('<h1 class="model-header animated-box">LightGBM Model</h1>', unsafe_allow_html=True)
    
    # Model information and facts (all elements sequential)
    st.markdown("""<div class="animated-box">""", unsafe_allow_html=True)

    # LightGBM Information
    st.markdown("""
    <div class="animated-box">
        <p>LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed 
        for distributed and efficient training and has proven to be highly effective for structured/tabular data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fun facts about LightGBM
    st.markdown("""
    <div class="fact-box animated-box">
        <h4>Did you know?</h4>
        <ul>
            <li>LightGBM was developed by Microsoft in 2017</li>
            <li>It uses a novel technique called Gradient-based One-Side Sampling (GOSS) to filter data instances</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 🔹 Directly Call the Results Display Function Instead of Showing Buttons
    display_lightgbm_results()


def display_lightgbm_results():
    st.markdown('<h3 class="sub-header animated-box">Model Performance</h3>', unsafe_allow_html=True)
    
    # Results container
    st.markdown("""<div class="animated-box">""", unsafe_allow_html=True)

    # Performance Metrics
    st.markdown(""" 
    <div class="metrics-container animated-box">
        <h4>Optimized LightGBM Performance Metrics:</h4>
        <p>🔹 <span class="metric-value">Accuracy: 0.8747</span></p>
        <p>🔹 <span class="metric-value">Precision: 0.5431</span></p>
        <p>🔹 <span class="metric-value">Recall: 0.7063</span></p>
        <p>🔹 <span class="metric-value">F1 Score: 0.6140</span></p>
        <p>🔹 <span class="metric-value">AUC-ROC: 0.8917</span></p>
    </div>
    """, unsafe_allow_html=True)

    # Classification Report - Improved Table Format
    st.markdown('<h4>📊 Classification Report (Optimized LightGBM)</h4>', unsafe_allow_html=True)

    # Creating the classification report as a dataframe
    data_lgbm = {
        "Class": ["0", "1", "Accuracy", "Macro Avg", "Weighted Avg"],
        "Precision": [0.95, 0.54, "", 0.75, 0.89],
        "Recall": [0.90, 0.71, "", 0.80, 0.87],
        "F1-Score": [0.93, 0.61, 0.87, 0.77, 0.88],
        "Support": [44249, 7267, 51516, 51516, 51516]
    }
    df_lgbm = pd.DataFrame(data_lgbm)

    table_html_lgbm = f"""
    <style>
        .styled-table {{
            width: 100%;
            max-width: 800px;
            margin: auto;
            border-collapse: collapse;
            font-family: 'Arial', sans-serif;
            font-size: 14px;
            text-align: center;
            background-color: #f4f4f4;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
        }}
        .styled-table th, .styled-table td {{
            padding: 8px;
            border: 1px solid #ddd;
        }}
        .styled-table th {{
            background-color: #2c3e50;
            color: white;
            text-transform: uppercase;
            font-weight: bold;
        }}
        .styled-table td {{
            color: black;
        }}
        .styled-table tr:nth-child(even) {{
            background-color: #ebebeb;
        }}
        .styled-table tr:hover {{
            background-color: #d5e6f3;
            transition: 0.3s;
        }}
    </style>
    <table class="styled-table">
        <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
        </tr>
        {''.join(f"<tr><td>{row['Class']}</td><td>{row['Precision']}</td><td>{row['Recall']}</td><td>{row['F1-Score']}</td><td>{row['Support']}</td></tr>" for _, row in df_lgbm.iterrows())}
    </table>
    """
    st.markdown(table_html_lgbm, unsafe_allow_html=True)


    # ROC Curve
    st.markdown('<h4>ROC Curve:</h4>', unsafe_allow_html=True)
    roc_curve = load_image(get_image_path("light_roc.png"))
    if roc_curve:
        st.image(roc_curve, use_column_width=True)
    else:
        st.info("ROC curve visualization not available")

    # LIME Explanation
    st.markdown('<h3 class="sub-header animated-box">Model Explainability (LIME)</h3>', unsafe_allow_html=True)

    lime_image = load_image(get_image_path("light_lime.png"))
    if lime_image:
        st.image(lime_image, use_column_width=True)
    else:
        st.info("LIME explanation visualization not available")

    # Interpretation
    st.markdown(""" 
    <div class="info-box animated-box">
        <h4>Interpretation:</h4>
        <p>The LightGBM model achieves a good balance between precision and recall for both classes. With an accuracy of 87.47% and 
        an AUC-ROC score of 0.8917, it demonstrates solid overall performance. Unlike the Random Forest model, LightGBM shows better 
        recall (0.7063) for the minority class, though at the cost of lower precision (0.5431).</p>
        <p>This trade-off might be preferable in scenarios where identifying as many positive cases as possible is more important than 
        avoiding false positives. The LIME explanation provides insight into feature importance for individual predictions, showing 
        which factors most heavily influenced the model's decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def display_xgboost():
    # Header with animation
    st.markdown('<h1 class="model-header animated-box">XGBoost - Our MVP Model</h1>', unsafe_allow_html=True)
    
    # Model information and facts
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div class="animated-box">
            <p>XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, 
            flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework with a focus on performance.</p>
            <p>Our implementation leverages Optuna for hyperparameter optimization to achieve maximum performance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="fact-box animated-box">
            <h4>Did you know?</h4>
            <ul>
                <li>XGBoost was developed by Tianqi Chen in 2014</li>
                <li>It's been used in over 60% of winning solutions in machine learning competitions</li>
                <li>Optuna is a hyperparameter optimization framework that uses Bayesian optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # XGBoost Variants
    xgboost_types = st.selectbox(
        "Select XGBoost Variant:",
        ["XGBoost with Optuna Optimization"]
    )
    
    if xgboost_types == "XGBoost with Optuna Optimization":
        display_xgboost_optuna()
    else:
        st.info(f"Details for {xgboost_types} will be added soon.")

def display_xgboost_optuna():
    st.markdown('<h3 class="sub-header animated-box">XGBoost with Optuna Optimization</h3>', unsafe_allow_html=True)
    
    # Optuna explanation
    st.markdown("""
    <div class="info-box animated-box">
        <h4 style="color: #1e3a5c;">About Optuna:</h4>
        <p style="color: #333;">Optuna is an automatic hyperparameter optimization framework that efficiently searches for the best hyperparameter settings. 
        It uses advanced algorithms like Bayesian optimization to find the optimal configuration that maximizes model performance. 
        For our XGBoost model, Optuna helped us find the best learning rate, maximum depth, subsample ratio, and other critical parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model upload and execution
    st.markdown('<h4 class="sub-header">Model Execution</h4>', unsafe_allow_html=True)
    
    # File uploader for model pickle
    uploaded_model = st.file_uploader("Upload XGBoost model (.pkl file)", type=["pkl"])
    
    start_button = st.button("Start Model Training/Evaluation")
    
    if start_button and uploaded_model is not None:
        with st.spinner("Running XGBoost model on training, validation, and test datasets..."):
            # Save the uploaded model to a temporary file
            with open("temp_xgb_model.pkl", "wb") as f:
                f.write(uploaded_model.getbuffer())

            # Execute the XGBoost evaluation code
            try:
                with st.expander("Execution Log", expanded=True):
                    st.markdown('<h5 style="color: #1e3a5c;">Training Dataset Evaluation:</h5>', unsafe_allow_html=True)
                    train_results = run_xgboost_on_train_data()
                    
                    st.markdown('<h5 style="color: #1e3a5c;">Test Dataset Evaluation:</h5>', unsafe_allow_html=True)
                    test_results = run_xgboost_on_test_data()
                    
                    # Store results in session state for display
                    st.session_state.xgb_train_results = train_results
                    st.session_state.xgb_test_results = test_results
                
                st.success("XGBoost model evaluation completed successfully!")
                
                # Display results automatically
                display_xgboost_results(train_results, test_results)

                # Display additional visuals (Top Features, SHAP Explanation)
                display_xgboost_visuals()

            except Exception as e:
                st.error(f"An error occurred during model execution: {str(e)}")


def display_xgboost_visuals():
    """Display XGBoost Model Insights including Feature Importance and SHAP Explanation."""
    
    st.markdown('<h3 class="sub-header animated-box">Model Insights</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4 style="color: #1e3a5c;">Top 20 Features in XGBoost</h4>', unsafe_allow_html=True)
        st.image(r"C:\Users\Siddhant Nijhawan\Downloads\git2\Frontend\pages\xgboost_top20.png", caption="Top 20 Most Important Features", use_column_width=True)
    
    with col2:
        st.markdown('<h4 style="color: #1e3a5c;">SHAP Explanation Summary</h4>', unsafe_allow_html=True)
        st.image(r"https://github.com/Siddhant2107/Clinical_trials/blob/master/Frontend/pages/xgboost_shap.png", caption="SHAP Summary Plot", use_column_width=True)
    
    st.markdown('<h4 style="color: #1e3a5c;">SHAP Dependence Plot</h4>', unsafe_allow_html=True)
    st.image(r"C:\Users\Siddhant Nijhawan\Downloads\git2\Frontend\pages\xgboost_shap2.png", caption="SHAP Dependence Plot", use_column_width=True)

def run_xgboost_on_train_data():
    """Run XGBoost model on training and validation datasets"""
    results = {}
    
    # Capture print outputs
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    with redirect_stdout(output):
        import pandas as pd
        import numpy as np
        import pickle
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Loading training and validation datasets...</span></div>', unsafe_allow_html=True)
        
        # Load the training and validation datasets
        X_train_df = pd.read_csv(r"C:\Users\Siddhant Nijhawan\Downloads\Nest images of model webpage\train_data_reduced.csv")
        X_val_df = pd.read_csv(r"C:\Users\Siddhant Nijhawan\Downloads\Nest images of model webpage\val_data_reduced.csv")

        # Extract true labels
        y_train = X_train_df['Study Status']
        y_val = X_val_df['Study Status']

        # Drop target column from features
        X_train_reduced = X_train_df.drop(columns=['Study Status'])
        X_val_reduced = X_val_df.drop(columns=['Study Status'])

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Handling missing data...</span></div>', unsafe_allow_html=True)
        # Handle missing data (replace Inf and NaN values)
        for df in [X_train_reduced, X_val_reduced]:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(df.mean(), inplace=True)

        st.markdown('<div style="background-color: #f2feeb; padding: 10px; border-radius: 5px; border-left: 5px solid #4caf50;"><span style="color: #1e3a5c; font-weight: bold;"> Replaced Inf values and handled missing data in training and validation sets.</span></div>', unsafe_allow_html=True)

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Loading and applying scaler...</span></div>', unsafe_allow_html=True)
        # Load the trained StandardScaler used during training
        with open(r"C:\Users\Siddhant Nijhawan\Downloads\Nest images of model webpage\scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)

        # Apply the same Standard Scaling transformation
        X_train_reduced = pd.DataFrame(scaler.transform(X_train_reduced), columns=X_train_reduced.columns)
        X_val_reduced = pd.DataFrame(scaler.transform(X_val_reduced), columns=X_val_reduced.columns)

        st.markdown('<div style="background-color: #f2feeb; padding: 10px; border-radius: 5px; border-left: 5px solid #4caf50;"><span style="color: #1e3a5c; font-weight: bold;"> Applied Standard Scaling to Training and Validation Features.</span></div>', unsafe_allow_html=True)

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Loading XGBoost model...</span></div>', unsafe_allow_html=True)
        # Load the trained XGBoost model (using the uploaded model)
        with open("temp_xgb_model.pkl", 'rb') as f:
            xgb_model = pickle.load(f)

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Making predictions...</span></div>', unsafe_allow_html=True)
        # Predict on training data
        y_train_pred = xgb_model.predict(X_train_reduced)  # Class labels
        y_train_pred_proba = xgb_model.predict_proba(X_train_reduced)[:, 1]  # Probability scores

        # Predict on validation data
        y_val_pred = xgb_model.predict(X_val_reduced)  # Class labels
        y_val_pred_proba = xgb_model.predict_proba(X_val_reduced)[:, 1]  # Probability scores

        # Evaluate model performance on training data
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)

        # Evaluate model performance on validation data
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Evaluating model performance...</span></div>', unsafe_allow_html=True)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Calculate additional metrics for training data
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        
        # Calculate additional metrics for validation data
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        # Create classification reports
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        val_report = classification_report(y_val, y_val_pred, output_dict=True)
        
        # Store results
        results['train_accuracy'] = train_accuracy
        results['train_precision'] = train_precision
        results['train_recall'] = train_recall
        results['train_f1'] = train_f1
        results['train_roc_auc'] = train_roc_auc
        results['train_report'] = train_report
        
        results['val_accuracy'] = val_accuracy
        results['val_precision'] = val_precision
        results['val_recall'] = val_recall
        results['val_f1'] = val_f1
        results['val_roc_auc'] = val_roc_auc
        results['val_report'] = val_report
        
        # Save predictions to CSV files
        train_predictions_df = pd.DataFrame({'True Label': y_train, 'Predicted': y_train_pred, 'Probability': y_train_pred_proba})
        val_predictions_df = pd.DataFrame({'True Label': y_val, 'Predicted': y_val_pred, 'Probability': y_val_pred_proba})
        
        # Print prediction summary
        st.markdown(f'<div style="background-color: #f2feeb; padding: 10px; border-radius: 5px; border-left: 5px solid #4caf50;"><span style="color: #1e3a5c; font-weight: bold;"> Generated predictions for {len(y_train)} training samples and {len(y_val)} validation samples.</span></div>', unsafe_allow_html=True)
    
    return results

def run_xgboost_on_test_data():
    """Run XGBoost model on test dataset"""
    results = {}
    
    # Capture print outputs
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    with redirect_stdout(output):
        import pandas as pd
        import numpy as np
        import pickle
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Loading test dataset...</span></div>', unsafe_allow_html=True)
        # Load the X_test data (already preprocessed and reduced)
        x_test_val_df = pd.read_csv(r"C:\Users\Siddhant Nijhawan\Downloads\Nest images of model webpage\test_data_reduced.csv")

        # Extract true labels
        y_true_new_data = x_test_val_df['Study Status']
        X_test_reduced = x_test_val_df.drop(columns=['Study Status'])  # Drop target column from features

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Handling missing data...</span></div>', unsafe_allow_html=True)
        # Handle missing data (replace Inf and NaN values)
        X_test_reduced.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test_reduced.fillna(X_test_reduced.mean(), inplace=True)

        st.markdown('<div style="background-color: #f2feeb; padding: 10px; border-radius: 5px; border-left: 5px solid #4caf50;"><span style="color: #1e3a5c; font-weight: bold;"> Replaced Inf values and handled missing data.</span></div>', unsafe_allow_html=True)

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Loading and applying scaler...</span></div>', unsafe_allow_html=True)
        # Load the trained StandardScaler used during training
        with open(r"C:\Users\Siddhant Nijhawan\Downloads\Nest images of model webpage\scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)

        # Apply the same Standard Scaling transformation
        X_test_reduced = pd.DataFrame(scaler.transform(X_test_reduced), columns=X_test_reduced.columns)

        st.markdown('<div style="background-color: #f2feeb; padding: 10px; border-radius: 5px; border-left: 5px solid #4caf50;"><span style="color: #1e3a5c; font-weight: bold;"> Applied Standard Scaling to Features.</span></div>', unsafe_allow_html=True)

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Loading XGBoost model...</span></div>', unsafe_allow_html=True)
        # Load the trained XGBoost model (using the uploaded model)
        with open("temp_xgb_model.pkl", 'rb') as f:
            xgb_model = pickle.load(f)

        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;"> Making predictions...</span></div>', unsafe_allow_html=True)
        # Predict on the new data
        y_pred_new_data = xgb_model.predict(X_test_reduced)  # Predict the class labels
        y_pred_proba_new_data = xgb_model.predict_proba(X_test_reduced)[:, 1]  # Predict probabilities for AUC

        # Evaluate model performance on the new data
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        test_accuracy = accuracy_score(y_true_new_data, y_pred_new_data)
        test_roc_auc = roc_auc_score(y_true_new_data, y_pred_proba_new_data)
        test_precision = precision_score(y_true_new_data, y_pred_new_data)
        test_recall = recall_score(y_true_new_data, y_pred_new_data)
        test_f1 = f1_score(y_true_new_data, y_pred_new_data)
        
        st.markdown('<div style="background-color: #f0f5ff; padding: 10px; border-radius: 5px; border-left: 5px solid #4361ee;"><span style="color: #1e3a5c; font-weight: bold;">⏳ Evaluating model performance...</span></div>', unsafe_allow_html=True)
        
        # Create classification report
        test_report = classification_report(y_true_new_data, y_pred_new_data, output_dict=True)
        
        # Store results
        results['test_accuracy'] = test_accuracy
        results['test_precision'] = test_precision
        results['test_recall'] = test_recall
        results['test_f1'] = test_f1
        results['test_roc_auc'] = test_roc_auc
        results['test_report'] = test_report
        
        # Save predictions to a DataFrame
        predictions_df = pd.DataFrame({'True Label': y_true_new_data, 'Predicted': y_pred_new_data, 'Probability': y_pred_proba_new_data})
        
        # Print prediction summary
        st.markdown(f'<div style="background-color: #f2feeb; padding: 10px; border-radius: 5px; border-left: 5px solid #4caf50;"><span style="color: #1e3a5c; font-weight: bold;">✅ Generated predictions for {len(y_true_new_data)} test samples.</span></div>', unsafe_allow_html=True)
    
    return results

def display_xgboost_results(train_results, test_results):
    """Display XGBoost results in a professional format"""
    
    st.markdown('<h3 class="sub-header animated-box">XGBoost Model Performance</h3>', unsafe_allow_html=True)
    
    # Create tabs for Train/Validation and Test results
    tab1, tab2 = st.tabs(["Training & Validation Results", "Test Results"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Training metrics
            st.markdown(f"""
            <div class="metrics-container animated-box" style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 5px solid #4361ee; margin-bottom: 20px;">
                <h4 style="color: #1e3a5c;">Training Performance Metrics:</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">Accuracy</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{train_results['train_accuracy']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">Precision</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{train_results['train_precision']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">Recall</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{train_results['train_recall']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">F1 Score</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{train_results['train_f1']:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; color: #333; font-weight: bold;">AUC-ROC</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{train_results['train_roc_auc']:.4f}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Validation metrics
            st.markdown(f"""
            <div class="metrics-container animated-box" style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 5px solid #4361ee; margin-bottom: 20px;">
                <h4 style="color: #1e3a5c;">Validation Performance Metrics:</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">Accuracy</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{train_results['val_accuracy']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">Precision</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{train_results['val_precision']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">Recall</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{train_results['val_recall']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">F1 Score</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{train_results['val_f1']:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; color: #333; font-weight: bold;">AUC-ROC</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{train_results['val_roc_auc']:.4f}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Training classification report
            st.markdown('<h4 style="color: #1e3a5c;">Training Classification Report:</h4>', unsafe_allow_html=True)
            create_classification_report_table(train_results['train_report'])
            
            # Validation classification report
            st.markdown('<h4 style="color: #1e3a5c;">Validation Classification Report:</h4>', unsafe_allow_html=True)
            create_classification_report_table(train_results['val_report'])
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Test metrics
            st.markdown(f"""
            <div class="metrics-container animated-box" style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 5px solid #4361ee; margin-bottom: 20px;">
                <h4 style="color: #1e3a5c;">Test Performance Metrics:</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">Accuracy</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{test_results['test_accuracy']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">Precision</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{test_results['test_precision']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">Recall</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{test_results['test_recall']:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; color: #333; font-weight: bold;">F1 Score</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{test_results['test_f1']:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; color: #333; font-weight: bold;">AUC-ROC</td>
                        <td style="padding: 8px; color: #333; text-align: right;">{test_results['test_roc_auc']:.4f}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature importance explanation (static content since we don't have actual feature importance)
            st.markdown("""
            <div class="info-box animated-box" style="background-color: #f0f5ff; border-radius: 8px; padding: 15px; border-left: 5px solid #4361ee; margin-bottom: 20px;">
                <h4 style="color: #1e3a5c;">Key Performance Insights:</h4>
                <p style="color: #333;">The XGBoost model demonstrates exceptional performance across all datasets, with particularly strong results
                on the test data. This indicates good generalization capability and minimal overfitting. The high AUC-ROC score
                shows excellent discriminative ability between classes.</p>
                <p style="color: #333;">Top influential features are automatically identified by the XGBoost algorithm, suggesting these variables have the 
                strongest predictive power for the target variable.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Test classification report
            st.markdown('<h4 style="color: #1e3a5c;">Test Classification Report:</h4>', unsafe_allow_html=True)
            create_classification_report_table(test_results['test_report'])
            
    
    # Model interpretation
    st.markdown('<h3 class="sub-header animated-box">Model Interpretation</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box animated-box" style="background-color: #f0f5ff; border-radius: 8px; padding: 15px; border-left: 5px solid #4361ee;">
        <h4 style="color: #1e3a5c;">Why XGBoost with Optuna Outperforms Other Models:</h4>
        <p style="color: #333;">Our XGBoost model, optimized with Optuna, demonstrates superior performance compared to other models for several reasons:</p>
        <ul style="color: #333;">
            <li><strong>Hyperparameter Optimization:</strong> Optuna systematically explored the hyperparameter space to find the optimal 
            configuration, resulting in improved generalization capability.</li>
            <li><strong>Gradient Boosting Advantage:</strong> XGBoost's sequential building of trees, each correcting errors of previous ones, 
            creates a powerful ensemble that captures complex patterns in the data.</li>
            <li><strong>Regularization:</strong> The L1 and L2 regularization in XGBoost helps prevent overfitting, which is particularly 
            important for our dataset.</li>
            <li><strong>Feature Importance:</strong> XGBoost automatically handles feature importance, effectively utilizing the most 
            predictive variables while minimizing noise from less important ones.</li>
        </ul>
        <p style="color: #333;">The model achieves a balance between precision and recall across both classes, making it suitable for real-world deployment 
        where both false positives and false negatives have significant consequences.</p>
    </div>
    """, unsafe_allow_html=True)

import pandas as pd
import streamlit as st

def create_classification_report_table(report_dict):
    """Create a well-formatted classification report table for Streamlit."""
    
    # Convert the classification report dictionary into a DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    
    # Round numerical values for better readability
    df_report = df_report.round(4)

    # Ensure support is consistent for macro avg, weighted avg, and accuracy
    if "accuracy" in df_report.index:
        common_support = int(df_report.loc["macro avg", "support"])  # Use support from macro avg
        df_accuracy = pd.DataFrame({
            "precision": [round(report_dict["accuracy"], 4)],
            "recall": ["-"], 
            "f1-score": ["-"], 
            "support": [common_support]  # Keeping same support as macro avg and weighted avg
        }, index=["Accuracy"])

        df_report = df_report.drop(index="accuracy")
        df_report = pd.concat([df_report, df_accuracy])

    # Improve presentation with markdown formatting
    st.markdown("### Classification Report")
    st.markdown("""
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 10px;
                text-align: center;
            }
            th {
                background-color: #778899;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #2E8B57;
            }
            tr:hover {
                background-color: #191970;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display the classification report table in Streamlit
    st.table(df_report)



if __name__ == "__main__":
    main()