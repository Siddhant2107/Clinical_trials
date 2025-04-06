import streamlit as st
import os
st.set_page_config(page_title='Adverse Event Clustering', layout='wide')



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
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from streamlit_lottie import st_lottie
import requests

# Must be first Streamlit command

# Function to load Lottie files
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation
lottie_medical = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_5njp3vgg.json')

# Custom CSS with more vibrant styling
st.markdown("""
    <style>
    /* Global styles */
    .main {
        background: linear-gradient(135deg, #1a2980, #26d0ce);
        color: white;
    }
    
    /* Title styling */
    .super-title {
        font-size: 3.5rem !important;
        font-weight: 700;
        background: linear-gradient(120deg, #f6d365, #fda085);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    /* Section headers */
    .gradient-text {
        font-size: 2.2rem;
        font-weight: 600;
        background: linear-gradient(120deg, #84fab0, #8fd3f4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
        margin: 1.5rem 0;
    }
    
    /* Custom containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 3.5rem;
        font-weight: 700;
        color: #f6d365;
        text-align: center;
    }
    
    .metric-label {
        font-size: 1.2rem;
        color: #ffffff;
        text-align: center;
        opacity: 0.9;
    }
    
    /* Description text */
    .description {
        font-size: 1.2rem;
        line-height: 1.6;
        color: white;
        opacity: 0.9;
    }
    
    /* Chart customization */
    .chart-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
    }
    
    /* Select box styling */
    .stSelectbox {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Title and Animation
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<h1 class="super-title">🩺 Adverse Event Clustering & Insights</h1>', unsafe_allow_html=True)
with col2:
    st_lottie(lottie_medical, height=200)

# Step 1
st.markdown('<h2 class="gradient-text">The Challenge of Clinical Trials</h2>', unsafe_allow_html=True)
st.markdown("""
    <div class="glass-container">
        <p class="description">
            Clinical trials generate massive amounts of data, especially when tracking adverse events. 
            With <span style='color: #f6d365; font-weight: bold;'>101,000</span> unique disease terms, 
            identifying patterns manually is impossible. To make sense of this complexity, we used 
            <span style='color: #f6d365; font-weight: bold;'>HDBSCAN clustering</span> to group similar events, 
            helping researchers uncover meaningful insights.
        </p>
    </div>
""", unsafe_allow_html=True)

# Metrics
num_clusters = 453  # Reduced by 1 to account for removal of -1 cluster
num_unique_events = 135520  # Added new metric
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
        <div class="glass-container">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Clusters Identified</div>
        </div>
    """.format(num_clusters), unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="glass-container">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Unique Events</div>
        </div>
    """.format(num_unique_events), unsafe_allow_html=True)

# Cluster Distribution
st.markdown('<h2 class="gradient-text">Cluster Distribution Analysis</h2>', unsafe_allow_html=True)
st.markdown("""
    <div class="glass-container">
        <p class="description">
            Each bar represents a group of similar adverse events. The height shows the number of terms in each cluster.
        </p>
    </div>
""", unsafe_allow_html=True)

# Removed -1 cluster from the dictionary
cluster_sizes = {
    234: 253, 184: 152, 229: 182, 238: 491, 335: 340, 94: 540,
    439: 198, 311: 213, 442: 134, 22: 145, 140: 272, 443: 168, 125: 230
}
cluster_df = pd.DataFrame(list(cluster_sizes.items()), columns=['Cluster', 'Count'])

# Custom chart styling
chart = alt.Chart(cluster_df).mark_bar().encode(
    x=alt.X('Cluster:N', title='Cluster ID'),
    y=alt.Y('Count:Q', title='Number of Terms'),
    color=alt.Color('Count:Q', scale=alt.Scale(scheme='viridis'))
).properties(height=400)

st.altair_chart(chart, use_container_width=True)

# UMAP Visualization
st.markdown('<h2 class="gradient-text">Cluster Visualization</h2>', unsafe_allow_html=True)
st.markdown("""
    <div class="glass-container">
        <p class="description">
            High-dimensional medical text transformed into a visual map using <span style='color: #f6d365; font-weight: bold;'>UMAP</span>. 
            Each point represents an adverse event, with colors indicating different clusters.
        </p>
    </div>
""", unsafe_allow_html=True)

BASE_URL = "https://raw.githubusercontent.com/Siddhant2107/Clinical_trials/refs/heads/master/Frontend/pages/"
def get_image_url(file_name):
        return f"{BASE_URL}{file_name}"  # Properly construct the full URL

output_url = get_image_url("output.jpg")
st.image(output_url, use_column_width=True)

#st.image(r"C:\Users\Siddhant Nijhawan\Downloads\git2\Frontend\pages\output.jpg", use_column_width=True)

# Cluster Explorer - Removed -1 cluster
st.markdown('<h2 class="gradient-text">Explore Clusters</h2>', unsafe_allow_html=True)
sample_clusters = {
    0: ["elevated alt", "sgpt (serum glutamic pyruvic transaminase)", "elevated ast", "abnormal glutamate-pyruvate transaminase........"],
    1: ["diverticulum", "diverticulitis", "diverticulum intestinal", "diverticulosis......"],
    2: ["cytomegalovirus infection", "pneumonia cytomegaloviral", "cytomegalovirus colitis....."],
    3: ["death not associated with ctcae term", "death nos", "sudden death", "disease progression nos......."]
}

selected_cluster = st.selectbox("Select Cluster ID", list(sample_clusters.keys()))
st.markdown(f"""
    <div class="glass-container">
        <h3 style='color: #f6d365; font-size: 1.5rem; margin-bottom: 1rem;'>
            Cluster {selected_cluster} 
        </h3>
        <p class="description">{", ".join(sample_clusters[selected_cluster])}</p>
    </div>
""", unsafe_allow_html=True)

# Conclusion
st.markdown('<h2 class="gradient-text">Impact & Conclusions</h2>', unsafe_allow_html=True)
st.markdown("""
    <div class="glass-container">
        <p class="description">
            Through advanced clustering techniques, we've transformed complex medical data into actionable insights,
            enabling faster analysis and more informed decision-making in clinical trials.
        </p>
    </div>
""", unsafe_allow_html=True)