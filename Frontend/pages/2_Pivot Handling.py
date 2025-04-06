import streamlit as st
import os
import streamlit as st

# ✅ Only one set_page_config call
st.set_page_config(
    page_title='Pivot Handling',
    layout='wide',
    initial_sidebar_state='expanded'
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
import altair as alt
import time
from streamlit_lottie import st_lottie
import requests
import plotly.express as px
import gdown
import os



# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 3rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        background-color: #1E88E5;
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer p {
        margin: 0;
        font-size: 1rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load Lottie animation
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load datasets with progress bar
with st.spinner('Loading datasets...'):
    # For usecase3_updated.csv
    file_id_usecase3 = '1RMFbwjz62F_Ie6_5tKnfTzYRMONFH__p'  # Replace with actual ID
    output_path_usecase3 = 'usecase3_updated.csv'
    url_usecase3 = f'https://drive.google.com/uc?id={file_id_usecase3}'
    gdown.download(url_usecase3, output_path_usecase3, quiet=False)
    csv_df = pd.read_csv(output_path_usecase3)
    
    # For facilities_drop.txt
    file_id_facilities = '1FKLcD43Cx_d9YQiQ1Ey9m03LEqrBeGig'  # Replace with actual ID
    output_path_facilities = 'facilities_drop.txt'
    url_facilities = f'https://drive.google.com/uc?id={file_id_facilities}'
    gdown.download(url_facilities, output_path_facilities, quiet=False)
    facilities_df = pd.read_csv(output_path_facilities, sep="|")
    
    # Data preprocessing
    facilities_df.drop(['status', 'name', 'state'], axis=1, inplace=True)
    csv_subset = csv_df[['NCT Number', 'Study Status']]
    merged_df = pd.merge(facilities_df, csv_subset, how="inner", left_on="nct_id", right_on="NCT Number")
    
    
# Main title with animation
st.title('🔍 Pivot Handling')

# Load and display data analysis animation
lottie_data = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
if lottie_data:
    with st.container():
        st_lottie(lottie_data, height=200, key="data_animation")

# Dashboard tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🌍 Country Analysis", "📈 Results"])

with tab1:
    st.header("Understanding the Raw Data")
    st.write("Clinical trials are conducted worldwide, and our dataset contains valuable information about study locations.")
    
    # Display dataset info with metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(merged_df), delta="100%")
    with col2:
        st.metric("Countries", len(merged_df['country'].unique()), "Unique")
    with col3:
        st.metric("Study Statuses", len(merged_df['Study Status'].unique()), "Types")

with tab2:
    st.header("Country-wise Analysis")
    
    # Create pivot table
    selected_feature = "country"
    pivot_table = pd.pivot_table(
        merged_df,
        index=[selected_feature],
        columns=['Study Status'],
        aggfunc='size',
        fill_value=0
    )
    
    pivot_table['total'] = pivot_table.sum(axis=1)
    pivot_table = pivot_table.sort_values(by='total', ascending=False)
    pivot_table['overall_cumulative'] = pivot_table['total'].cumsum()
    pivot_table['overall_cumulative_percentage'] = (pivot_table['overall_cumulative'] / pivot_table['total'].sum()) * 100

    # Interactive choropleth map
    fig = px.choropleth(
        pivot_table.reset_index(),
        locations="country",
        locationmode="country names",
        color="total",
        hover_name="country",
        color_continuous_scale="Viridis",
        title="Global Distribution of Clinical Trials"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display pivot table with styling
    st.subheader("Detailed Country Statistics")
    st.dataframe(
        pivot_table[['total', 'overall_cumulative', 'overall_cumulative_percentage']].style.background_gradient(cmap='Blues'),
        height=400
    )

with tab3:
    st.header("Results and Insights")
    
    # Apply categorization
    pivot_table['country_class'] = pivot_table.index.to_series()
    pivot_table.loc[pivot_table['overall_cumulative_percentage'] > 91, 'country_class'] = 'country with high cumsum value'
    final_unique_countries = pivot_table['country_class'].nunique()

    # Display metrics with animations
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Original Unique Countries",
            len(pivot_table),
            delta=f"-{len(pivot_table) - final_unique_countries} after reduction"
        )
    with col2:
        st.metric(
            "After Categorization",
            final_unique_countries,
            delta="Optimized"
        )

    # Enhanced reduction visualization
    reduction_data = pd.DataFrame({
        "Step": ["Original", "After 91% Threshold"],
        "Count": [len(pivot_table), final_unique_countries]
    })

    fig = px.bar(
        reduction_data,
        x="Step",
        y="Count",
        color="Step",
        title="Column Reduction Analysis",
        color_discrete_map={"Original": "#FF6B6B", "After 91% Threshold": "#4ECB71"}
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>📊 Data Preprocessing & Analysis Dashboard | Created with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)