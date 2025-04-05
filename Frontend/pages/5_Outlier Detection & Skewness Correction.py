import streamlit as st
import os
st.set_page_config(page_title='Outlier Detection & Skewness Correction', layout='wide')


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
import altair as alt
import matplotlib.pyplot as plt
import gdown

# Load dataset

def download_and_load_csv(file_id, output_path):
    """Download a CSV file from Google Drive and load it into a DataFrame."""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    df = pd.read_csv(output_path)
    
    # Drop the unnamed column if it exists
    if "Unnamed: 0" in df.columns:
        df.drop(["Unnamed: 0"], axis=1, inplace=True)
        
    return df

null_values_file_id = "1fESesydxwvWkY3Ftqe70JNN616cJLest"  

output_path = "null_values_dealt.csv"
sidd = download_and_load_csv(null_values_file_id, output_path)









st.title('📊 Outlier Detection & Skewness Handling')

st.write("## Step 1: Understanding Skewness in Data")
st.write("Skewed distributions can mislead models and affect predictions. Below is the skewness of selected numerical features before any transformation.")

# Define selected numerical columns
columns = ['Enrollment', 'subjects_at_risk', 'subjects_affected', 'duration', 'minimum_age', 'child', 'maximum_age', 'older_adult', 'adult']
skewness = sidd[columns].skew().reset_index()
skewness.columns = ['Feature', 'Skewness']

# Create an interactive Altair bar chart
skew_chart = alt.Chart(skewness).mark_bar().encode(
    x=alt.X('Feature', sort='-y', title='Numerical Features'),
    y=alt.Y('Skewness', title='Skewness Value'),
    color=alt.condition(
        alt.datum.Skewness > 0, alt.value("#1f77b4"), alt.value("#d62728")
    ),
    tooltip=['Feature', 'Skewness']
).interactive()

st.altair_chart(skew_chart, use_container_width=True)

st.write("## Step 2: Handling Outliers")
st.write("Outliers can distort statistical analysis and impact model performance. We used different strategies to cap outliers based on column characteristics:")

outlier_cap_methods = {
    'Enrollment': "Capped between 1st and 99th percentiles to remove extreme enrollments.",
    'duration': "Capped between 1st and 99th percentiles to manage unusual study lengths.",
    'minimum_age': "Set lower bound at 18 since clinical trials do not involve infants.",
    'maximum_age': "Set lower bound at 18 and upper bound at 100 to align with medical guidelines.",
    'subjects_at_risk': "Retained without changes to preserve variance.",
    'subjects_affected': "Retained without changes to maintain variability.",
    'child': "Kept as-is since it represents binary data.",
    'older_adult': "Retained without modifications.",
    'adult': "Kept as is since it classifies participant age groups."
}

st.write("### Outlier Capping Methods Used:")
for col, method in outlier_cap_methods.items():
    st.write(f"- **{col}**: {method}")

def cap_outliers_custom(df, column, lower_bound=None, upper_bound=None):
    if lower_bound is not None:
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    if upper_bound is not None:
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

for col in ['Enrollment', 'duration']:
    lower = np.percentile(sidd[col], 1)
    upper = np.percentile(sidd[col], 99)
    sidd = cap_outliers_custom(sidd, col, lower, upper)
for col in ['minimum_age', 'maximum_age']:
    if col == 'minimum_age':
        sidd = cap_outliers_custom(sidd, col, lower_bound=18)
    elif col == 'maximum_age':
        sidd = cap_outliers_custom(sidd, col, lower_bound=18, upper_bound=100)

# Boxplot visualization
selected_col = st.selectbox("Select a column to visualize outlier treatment", columns)
fig, ax = plt.subplots()
sidd[selected_col].plot(kind='box', ax=ax)
ax.set_title(f"After Outlier Processing: {selected_col}")
st.pyplot(fig)

st.write("## Step 3: Log Transformation for Skewed Features")
st.write("To correct skewness, we applied log transformation to numerical variables heavily affected by outliers.")

selected_skew_col = st.selectbox("Select a column for log transformation", ['Enrollment', 'subjects_at_risk', 'subjects_affected', 'duration'])
sidd[selected_skew_col] = np.log1p(sidd[selected_skew_col])

before_after_skewness = pd.DataFrame({
    'Stage': ['Before Transformation', 'After Transformation'],
    'Skewness': [skewness.loc[skewness['Feature'] == selected_skew_col, 'Skewness'].values[0], sidd[selected_skew_col].skew()]
})
st.bar_chart(before_after_skewness.set_index('Stage'))

st.write("## Conclusion")
st.write("By handling outliers with appropriate capping methods and correcting skewness using log transformation, we created a more stable dataset for clinical trial analysis.")
