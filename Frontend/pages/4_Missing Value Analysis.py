import streamlit as st
import os
st.set_page_config(page_title='🔍 Missing Value Analysis', layout='wide')


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
import altair as alt
import numpy as np
import time
from streamlit_lottie import st_lottie
import json
import gdown
import os

# Load dataset
file_id = '1aCdIUjkVVM0yYkfVLOHEGPpCaMCZIfc6'

# Path to save the downloaded file
output_path = 'final_result5.xlsx'

# Download the file
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, output_path, quiet=False)

# Now read the downloaded file
final_result5 = pd.read_excel(output_path)

# Page Configuration
st.title('✨ Handling Missing Values in Clinical Trials')
st.markdown("---")

# Load Lottie Animation
def load_lottieurl(url):
    return url

lottie_data = load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json')
st_lottie(lottie_data, speed=1, height=250, key="loading")

# Step 1: Identifying Missing Values
st.subheader("📊 Step 1: Identifying Missing Values")
st.write("Missing values can significantly impact clinical trial data. Let's visualize their distribution.")

missing_values = final_result5.isnull().sum() / len(final_result5) * 100
missing_values = missing_values.sort_values()

# Animated loading effect
with st.spinner("Analyzing missing values..."):
    time.sleep(2)

# Create an interactive Altair bar chart
missing_df = pd.DataFrame({'Column': missing_values.index, 'Missing %': missing_values.values})
chart = alt.Chart(missing_df).mark_bar().encode(
    x=alt.X('Column:N', sort='-y', title='Columns'),
    y=alt.Y('Missing %:Q', title='Missing Values (%)'),
    color=alt.condition(
        alt.datum['Missing %'] > 50,
        alt.value("#ff6961"),  # Red for high missing percentage
        alt.value("#77dd77")   # Green for low missing percentage
    ),
    tooltip=['Column', 'Missing %']
).interactive()
st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# Step 2: Removing Columns with Excessive Missing Data
st.subheader("🗑 Step 2: Removing Columns with Excessive Missing Data")
st.write("Columns with more than *90% missing values* are removed to improve dataset quality.")
columns_to_drop = missing_values[missing_values > 90].index.tolist()
st.success(f"✅ Removed Columns: {columns_to_drop}")
final_result5 = final_result5.drop(columns=columns_to_drop)

st.markdown("---")

# Step 3: Missing Value Distribution in Completed vs Non-Completed Studies
st.subheader("📌 Step 3: Comparing Missing Values in Study Groups")
st.write("We analyze the missing value distribution in *Completed vs Non-Completed Studies*.")

def calculate_missing_values_by_group(group):
    return group.isnull().sum() / len(group) * 100

completed_group = final_result5[final_result5['Study Status'] == 'COMPLETED']
non_completed_group = final_result5[final_result5['Study Status'] != 'COMPLETED']

missing_completed = calculate_missing_values_by_group(completed_group)
missing_non_completed = calculate_missing_values_by_group(non_completed_group)

missing_comparison = pd.DataFrame({
    'Column': missing_completed.index,
    'COMPLETED (%)': missing_completed.values,
    'NON COMPLETED (%)': missing_non_completed.values
})
missing_comparison = missing_comparison.sort_values(by='COMPLETED (%)', ascending=False)

# Create an Altair chart for comparison
comparison_chart = alt.Chart(missing_comparison.melt(id_vars=['Column'], var_name='Study Status', value_name='Missing %'))\
    .mark_bar().encode(
    x=alt.X('Column:N', sort='-y', title='Columns'),
    y=alt.Y('Missing %:Q', title='Missing Values (%)'),
    color='Study Status:N',
    tooltip=['Column', 'Missing %']
).interactive()

st.altair_chart(comparison_chart, use_container_width=True)

st.markdown("---")

# Step 4: Why We Use Median for Missing Values
st.subheader("🧠 Step 4: Why We Use Median for Missing Values?")
with st.expander("🔎 Click to see why median is preferred over mean"):
    st.markdown("""
    - *Resistant to Outliers: Unlike the mean, median **ignores extreme values* that are common in medical datasets.
    - *Better Representation: For **skewed distributions*, the median better captures the central tendency.
    - *Preserves Data Integrity: Ensures that our imputed values **don’t distort statistical properties*.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/35/Mean_vs_median.png", width=500)

st.success("🎯 Median is our best choice for imputing missing values!")