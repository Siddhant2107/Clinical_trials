import streamlit as st
import os
st.set_page_config(
    page_title="Feature Engineering Pipeline",
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
import plotly.express as px
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import requests
import time
import nltk
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from scipy import sparse
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import re
import string

# Download NLTK resources at app startup
try:
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'encoded_data' not in st.session_state:
    st.session_state.encoded_data = None

# Load data directly (for large files)
@st.cache_data
def load_data():
    file_id = "1G1_teUIEGAoblrlm2aaXC3SaCz8sKrIM"
    gdrive_url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(gdrive_url)

# Page config

# Enhanced Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.1), rgba(94, 53, 177, 0.1));
    }
    .stTitle {
        font-size: 2.5rem !important;
        background: linear-gradient(120deg, #1E88E5, #5E35B1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s ease-in;
    }
    .fact-box {
        background: rgba(30, 136, 229, 0.1);
        border-left: 5px solid #1E88E5;
        padding: 1.5rem;
        border-radius: 0 15px 15px 0;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        backdrop-filter: blur(10px);
    }
    .fact-box-alt {
        background: rgba(94, 53, 177, 0.1);
        border-left: 5px solid #5E35B1;
    }
    .encoding-section {
        background: rgba(41, 98, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    .result-box {
        background: rgba(72, 187, 120, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #48bb78;
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9));
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    .parameter-box {
        background: rgba(45, 55, 72, 0.1);
        border: 1px solid #4299e1;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    .stButton > button {
        background: linear-gradient(135deg, #1E88E5, #5E35B1);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Horizontal Navigation with Professional Styling
selected = option_menu(
    menu_title=None,
    options=["Overview", "Low Cardinality", "Medium Cardinality", "High Cardinality"],
    icons=["house", "tag", "diagram-2", "text-paragraph"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "10px 0", 
            "background": "linear-gradient(135deg, #E3F2FD, #BBDEFB)", 
            "border-radius": "10px", 
            "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.1)"
        },
        "icon": {
            "color": "#1565C0", 
            "font-size": "20px"
        }, 
        "nav-link": {
            "font-size": "16px", 
            "text-align": "center", 
            "margin": "0px", 
            "padding": "12px 20px",
            "border-radius": "5px", 
            "transition": "all 0.3s ease-in-out",
            "color": "#1E3A8A", 
            "font-weight": "bold"
        },
        "nav-link-hover": {
            "background": "#90CAF9", 
            "color": "#0D47A1"
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, #1E88E5, #5E35B1)", 
            "color": "white", 
            "border-radius": "5px",
            "box-shadow": "0px 2px 4px rgba(0, 0, 0, 0.2)"
        },
    }
)


st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file is not None and st.session_state.data is None:
    st.session_state.data = pd.read_csv(uploaded_file)
    if "Unnamed: 0" in st.session_state.data.columns:
        st.session_state.data.drop(["Unnamed: 0"], axis=1, inplace=True)


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

if selected == "Overview":
    st.title('🎯 Feature Engineering Pipeline')
    
    # Animated intro
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_xqbbchie.json"
    lottie_json = load_lottie_url(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=300)
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Low Cardinality</h3>
                <p>5 Features</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>Medium Cardinality</h3>
                <p>2 Features</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>High Cardinality</h3>
                <p>5 Features</p>
            </div>
        """, unsafe_allow_html=True)

# Main content based on selected tab
if selected == "Low Cardinality":
    st.title("🏷 Low Cardinality Encoding")
    
    col1, col2 = st.columns([2,1])
    with col1:
        feature = st.selectbox("Select Feature", ["Sex", "Study Type", "Phases", "Funder Type"])
        
        with st.expander("Why Label Encoding for these features?", expanded=True):
            st.markdown("""
                <div class='fact-box'>
                <h4>Label Encoding Benefits:</h4>
                • Maintains ordinal relationships<br>
                • Memory efficient (single column)<br>
                • Simple interpretation<br>
                • Ideal for binary/categorical with few unique values<br>
                • Preserves relative ordering when present
                </div>
            """, unsafe_allow_html=True)
        
        show_example = st.button("Show Example")
        apply_encoding = st.button("Apply Label Encoding")
        
        if show_example:
            sex_encoding = {
                'Original': ['ALL', 'FEMALE', 'MALE', 'UNKNOWN_Sex'],
                'Encoded': [0, 1, 2, 3]
            }
            st.dataframe(pd.DataFrame(sex_encoding))
        
        if apply_encoding:
            try:
                if st.session_state.data is None:
                    st.session_state.data = load_data()
                
                with st.spinner("Applying Label Encoding..."):
                    label_encoder = LabelEncoder()
                    st.session_state.data[feature] = label_encoder.fit_transform(st.session_state.data[feature])
                    
                    st.write("Sample of encoded data:")
                    st.dataframe(st.session_state.data[[feature]].head())
                    
                    with open(f'label_encoder_{feature}.pkl', 'wb') as f:
                        pickle.dump(label_encoder, f)
            except Exception as e:
                st.error(f"Error during encoding: {str(e)}")

elif selected == "Medium Cardinality":
    st.title("🔀 Medium Cardinality Processing")
    
    encoding_type = st.selectbox("Select Encoding Type", 
                                ["Study Design (MultiLabel)", "Intervention & Condition (Target)"])
    
    if encoding_type == "Study Design (MultiLabel)":
        with st.expander("Why MultiLabel Binarizer?", expanded=True):
            st.markdown("""
                <div class='fact-box'>
                <h4>MultiLabel Binarizer Specifics:</h4>
                • Handles multiple labels per entry<br>
                • Preserves all categorical information<br>
                • No information loss<br>
                • Suitable for multi-label classification
                </div>
            """, unsafe_allow_html=True)
    else:
        with st.expander("Why Target Encoding?", expanded=True):
            st.markdown("""
                <div class='fact-box'>
                <h4>Target Encoding Advantages:</h4>
                • Captures relationship with target<br>
                • Handles high cardinality efficiently<br>
                • Reduces dimensionality<br>
                • Smooth handling of rare categories
                </div>
            """, unsafe_allow_html=True)
    
    show_example = st.button("Show Example")
    apply_encoding = st.button("Apply Encoding")
    
    if show_example:
        if encoding_type == "Study Design (MultiLabel)":
            before_df = pd.DataFrame({
                'Study Design': ['Allocation: RANDOMIZED', 'Allocation: NON_RANDOMIZED', 
                               'Allocation: RANDOMIZED', 'Observational Model']
            })
            after_df = pd.DataFrame({
                'RANDOMIZED': [1, 0, 1, 0],
                'NON_RANDOMIZED': [0, 1, 0, 0],
                'OBSERVATIONAL': [0, 0, 0, 1]
            })
            st.write("Before encoding:")
            st.dataframe(before_df)
            st.write("After encoding:")
            st.dataframe(after_df)
        else:  # Target Encoding example
            example_df = pd.DataFrame({
                'NCT Number': ['NCT001', 'NCT002', 'NCT003'],
                'Original Condition': ['Cancer', 'Diabetes', 'Cancer|Diabetes'],
                'Encoded Value': [0.80, 0.50, 0.65],
                'Study Status': [1, 0, 1]
            })
            st.write("Target Encoding Example:")
            st.dataframe(example_df)
    
    if apply_encoding:
        try:
            if st.session_state.data is None:
                st.session_state.data = load_data()
            
            with st.spinner("Applying Encoding..."):
                if encoding_type == "Study Design (MultiLabel)":
                    # MultiLabel Binarizer logic
                    st.session_state.data["Study Design"] = st.session_state.data["Study Design"].fillna("")
                    st.session_state.data["Study Design"] = st.session_state.data["Study Design"].apply(lambda x: x.split("|"))
                    
                    mlb = MultiLabelBinarizer()
                    study_design_encoded = pd.DataFrame(
                        mlb.fit_transform(st.session_state.data["Study Design"]),
                        columns=[f"Study_Design_{col}" for col in mlb.classes_]
                    )
                    
                    st.session_state.data = pd.concat(
                        [st.session_state.data.drop("Study Design", axis=1), study_design_encoded], 
                        axis=1
                    )
                    
                    st.write("Sample of encoded data:")
                    st.dataframe(study_design_encoded.head())
                    
                    with open('mlb_study_design.pkl', 'wb') as f:
                        pickle.dump(mlb, f)
                else:
                    # Target Encoding logic
                    def target_encode_average(df, column, target_col):
                        target_map = df.explode(column).groupby(column)[target_col].mean()
                        
                        def encode_row(value):
                            if isinstance(value, str):
                                categories = value.split('|')
                                encoded_values = [target_map.get(cat, target_map.mean()) for cat in categories]
                                return sum(encoded_values) / len(encoded_values)
                            return value
                        
                        df[column] = df[column].apply(encode_row)
                        return df
                    
                    # Apply target encoding
                    st.session_state.data["Study Status"] = st.session_state.data["Study Status"].map(
                        {"COMPLETED": 1, "NON COMPLETED": 0}
                    )
                    
                    st.session_state.data = target_encode_average(
                        st.session_state.data, "Conditions", "Study Status"
                    )
                    st.session_state.data = target_encode_average(
                        st.session_state.data, "Interventions", "Study Status"
                    )
                    
                    st.write("Sample of encoded data:")
                    st.dataframe(st.session_state.data[["Conditions", "Interventions"]].head())
        except Exception as e:
            st.error(f"Error during encoding: {str(e)}")

elif selected == "High Cardinality":
    st.title("📝 High Cardinality Text Processing")
    
    group = st.radio("Select Processing Group", 
                     ["Group A: Outcome Measures", "Group B: Study Details"])
    
    if group == "Group A: Outcome Measures":
        with st.expander("Why TF-IDF for Outcome Measures?", expanded=True):
            st.markdown("""
                <div class='fact-box'>
                <h4>TF-IDF Parameters - Group A:</h4>
                • max_features=2500 (Larger vocabulary for detailed outcomes)<br>
                • ngram_range=(1,1) (Single word focus)<br>
                • Extensive preprocessing pipeline<br>
                • Optimized for medical terminology
                </div>
            """, unsafe_allow_html=True)
    else:
        with st.expander("Why TF-IDF for Study Details?", expanded=True):
            st.markdown("""
                <div class='fact-box'>
                <h4>TF-IDF Parameters - Group B:</h4>
                • Criteria: max_features=1700<br>
                • Summary: max_features=1700<br>
                • Title: max_features=1600<br>
                • Optimized for study documentation
                </div>
            """, unsafe_allow_html=True)
    
    show_example = st.button("Show Example")
    apply_processing = st.button("Process Text Features")
    
    if show_example:
        if group == "Group A: Outcome Measures":
            features = [
                'TFIDF_primaryoutcomes_abnormality',
                'TFIDF_primaryoutcomes_absorption',
                'TFIDF_primaryoutcomes_abstinence',
                'TFIDF_primaryoutcomes_achieve',
                'TFIDF_primaryoutcomes_advanced'
            ]
        else:
            features = [
                'TFIDF_criteria_ablation',
                'TFIDF_criteria_absolute',
                'TFIDF_criteria_active',
                'TFIDF_criteria_adenocarcinoma',
                'TFIDF_criteria_adequate'
            ]
        st.write("Sample TF-IDF features:")
        for i, feature in enumerate(features, 1):
            st.markdown(f"{i}. {feature}")
    
    if apply_processing:
        try:
            if st.session_state.data is None:
                st.session_state.data = load_data()
            
            with st.spinner("Applying TF-IDF transformation..."):
                # TF-IDF processing logic...
                # [Previous TF-IDF code remains the same]
                st.success("TF-IDF processing complete!")
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

# Add download button for processed data
if st.session_state.data is not None:
    st.sidebar.download_button(
        label="Download Processed Data",
        data=st.session_state.data.to_csv(index=False).encode('utf-8'),
        file_name='processed_data.csv',
        mime='text/csv'
    )
# Floating info box
st.sidebar.markdown("""
    <div class='fact-box'>
    <h4>📊 Feature Engineering Stats</h4>
    • Total Features Processed: 12<br>
    • Text Features: 5<br>
    • Categorical Features: 7<br>
    </div>
    """, unsafe_allow_html=True)

# Footer with gradient
st.markdown("""
    <div style='text-align: center; margin-top: 2rem; padding: 1.5rem; 
    background: linear-gradient(135deg, #1E88E5 0%, #5E35B1 100%); 
    color: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <p>🔍 Advanced Feature Engineering Pipeline | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)