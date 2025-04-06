import streamlit as st
import os
import streamlit as st

st.set_page_config(
    page_title="Clinical Trials Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Must be first Streamlit command
import pandas as pd
import altair as alt
from streamlit_lottie import st_lottie
import requests
import plotly.express as px




# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load data function remains the same
def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='|')
        else:
            return None
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

# Predefined file paths
import pandas as pd
import gdown
import os
import streamlit as st

# Define file IDs and local paths
file_info = {
    "usecase3_updated.csv": {
        "file_id": "1uMG_FaqUEmZ9MbZ7DNxbW-2Cs7oLEPL3",
        "output_path": "usecase3_updated.csv"  
    },
    "drop_withdrawals_drop.txt": {
        "file_id": "1eooUrLIePZ1iygJRqnEEa4GORXuPGyzs",
        "output_path": "drop_withdrawals_drop.txt"
    },
    "facilities_drop.txt": {
        "file_id": "1FKLcD43Cx_d9YQiQ1Ey9m03LEqrBeGig",
        "output_path": "facilities_drop.txt"
    },
    "reported_events_drop.txt": {
        "file_id": "134HKNOrP8hlh43-atf6uKz0M5I4V97te",
        "output_path": "reported_events_drop.txt"
    },
    "eligibilities_drop.txt": {
        "file_id": "1AQ5At-_IkonBMhjte188cqbILRYQYz_E",
        "output_path": "eligibilities_drop.txt"
    }
}

# Download all files from Google Drive
with st.spinner('Downloading files from Google Drive...'):
    for file_name, info in file_info.items():
        url = f"https://drive.google.com/uc?id={info['file_id']}"
        gdown.download(url, info['output_path'], quiet=False)

# Now you can use these local paths for your data processing
file_paths = {
    file_name: info["output_path"] for file_name, info in file_info.items()
}

# Load data animation
lottie_data = load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json')

# Custom CSS with modern styling
st.markdown("""
    <style>
    /* Global styles */
    .main {
        background: linear-gradient(135deg, #141E30, #243B55);
        color: white;
    }
    
    /* Title styling */
    .super-title {
        font-size: 3rem !important;
        font-weight: 700;
        background: linear-gradient(120deg, #00F5A0, #00D9F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        background: linear-gradient(120deg, #00F5A0, #00D9F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.8rem 0;
        margin: 1.2rem 0;
    }
    
    /* Card container */
    .card-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00F5A0;
        text-align: center;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #ffffff;
        text-align: center;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: rgba(20, 30, 48, 0.95);
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00F5A0 0%, #00D9F5 100%);
        color: #141E30;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 245, 160, 0.3);
    }
    
    /* DataFrame styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Animation
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="super-title">🔍 Clinical Trials Data Explorer</h1>', unsafe_allow_html=True)
with col2:
    st_lottie(lottie_data, height=150)

# Sidebar with gradient background
st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, rgba(20, 30, 48, 0.95), rgba(36, 59, 85, 0.95)); padding: 1rem; border-radius: 10px;'>
        <h2 style='color: #00F5A0; margin-bottom: 1rem;'>🛠 Data Controls</h2>
    </div>
""", unsafe_allow_html=True)

# Dataset Selection
selected_file = st.sidebar.selectbox("📂 Select Dataset", list(file_paths.keys()))

# Load selected dataset
selected_df = load_data(file_paths[selected_file])

if selected_df is not None:
    # Create session state
    if 'df' not in st.session_state or st.session_state.selected_file != selected_file:
        st.session_state.df = selected_df.copy()
        st.session_state.selected_file = selected_file

    df = st.session_state.df

    # Dataset Overview Card
    st.markdown("""
        <div class="card-container">
            <h3 style='color: #00F5A0; margin-bottom: 1rem;'>📊 Dataset Overview</h3>
            <div style='display: flex; justify-content: space-around; margin: 1rem 0;'>
                <div class="metric-value">{:,}</div>
                <div class="metric-value">{}</div>
            </div>
            <div style='display: flex; justify-content: space-around;'>
                <div class="metric-label">Rows</div>
                <div class="metric-label">Columns</div>
            </div>
        </div>
    """.format(df.shape[0], df.shape[1]), unsafe_allow_html=True)

    # Column Management
    st.sidebar.markdown("""
        <div style='background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
            <h3 style='color: #00F5A0;'>✂ Column Management</h3>
        </div>
    """, unsafe_allow_html=True)
    
    columns_to_drop = st.sidebar.multiselect("Select columns to drop", df.columns)

    if st.sidebar.button("🗑 Drop Selected Columns"):
        if columns_to_drop:
            st.session_state.df = df.drop(columns=columns_to_drop)
            st.success("✨ Columns successfully dropped!")
        else:
            st.warning("⚠ Please select columns to drop")
        st.rerun()

    # Data Preview
    st.markdown('<h2 class="section-header">📋 Data Preview</h2>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    # Download Section
    if st.sidebar.button("💾 Export Processed Data"):
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"processed_{selected_file}",
            mime="text/csv"
        )

    # Data Insights Section
    st.markdown('<h2 class="section-header">🔍 Data Insights</h2>', unsafe_allow_html=True)
    
    # Memory Usage Card
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    st.markdown(f"""
        <div class="card-container">
            <h4 style='color: #00F5A0;'>💾 Memory Usage</h4>
            <div class="metric-value">{memory_usage:.2f} MB</div>
        </div>
    """, unsafe_allow_html=True)

     # Data Types Visualization
    st.markdown('<h3 class="section-header">📊 Data Types Distribution</h3>', unsafe_allow_html=True)
    # Convert dtypes to strings to ensure JSON serialization
    dtypes = df.dtypes.astype(str).value_counts()
    fig = px.pie(values=dtypes.values, names=dtypes.index, 
                 title='Column Data Types',
                 color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

    # Interactive Feature Explorer
    st.markdown('<h3 class="section-header">🔍 Feature Explorer</h3>', unsafe_allow_html=True)
    selected_col = st.selectbox("Select a feature to analyze", df.columns)

    if pd.api.types.is_numeric_dtype(df[selected_col]):
        fig = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}',color_discrete_sequence=['#00F5A0'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        # For categorical columns
        value_counts = df[selected_col].value_counts().head(10)
        fig = px.bar(x=value_counts.index, y=value_counts.values,
                    title=f'Top 10 Values in {selected_col}',
                    color_discrete_sequence=['#00F5A0'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Expandable Insights
    with st.expander("📈 Numerical Summary"):
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.write(df.describe())
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📊 Categorical Summary"):
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        categorical_cols = df.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            for col in categorical_cols:
                st.write(f"{col}: {df[col].nunique()} unique values")
        else:
            st.write("No categorical columns found")
        st.markdown('</div>', unsafe_allow_html=True)
        
# Define the navigation between pages
def add_navigation_buttons():
    st.markdown("""
    <style>
        .navigation-container {
            display: flex;
            justify-content: space-between;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
        }
        .nav-button {
            background: linear-gradient(90deg, #1e3c72, #2a5298);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 4px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }
        .nav-button:hover {
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .nav-button.disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .nav-button.home {
            background: #34495e;
        }
    </style>
    """, unsafe_allow_html=True)
    



# Add this at the end of each subpage file (e.g., app1.py, app2.py, etc.)

def add_navigation_buttons():
    # Determine if we're in the pages directory or root
    import os
    
    current_file = os.path.basename(__file__)
    
    # Define paths based on whether we're in the pages directory or not
    in_pages_dir = os.path.dirname(__file__).endswith('pages')
    
    if in_pages_dir:
        # We're in the pages directory
        app_prefix = ""
        home_path = "../Main_page.py"
    else:
        # We're in the root directory
        app_prefix = "pages/"
        home_path = "Main_page.py"
    
    # Map page filenames to their sequence
    page_sequence = {
        '1_Clinical Trials Data Explorer': 1,
        '2_Pivot Handling.py': 2,
        '3_Adverse Event Clustering.py': 3,
        '4_Missing Value Analysis.py': 4,
        '5_Outlier Detection & Skewness Correction.py': 5,
        '7_Feature Engineering Pipeline.py': 6,
        '8_Healthcare Data Feature Reduction Journey.py': 7,
        '9_ML Models Showcase.py': 8,
    }
    
    # Custom CSS for navigation buttons
    st.markdown("""
    <style>
        .navigation-container {
            display: flex;
            justify-content: space-between;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
        }
        .nav-button {
            background: linear-gradient(90deg, #1e3c72, #2a5298);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 4px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }
        .nav-button:hover {
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .nav-button.disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .nav-button.home {
            background: #34495e;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Get current position in sequence
    current_position = page_sequence.get(current_file, 0)
    
    # Create buttons container
    st.markdown('<div class="navigation-container">', unsafe_allow_html=True)
    
    # Previous button
    if current_position > 1:
        prev_page = next((k for k, v in page_sequence.items() if v == current_position - 1), None)
        if prev_page:
            st.markdown(f'<a href="{app_prefix}{prev_page}" target="_blank" class="nav-button">← Previous Step</a>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="nav-button disabled">← Previous Step</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span></span>', unsafe_allow_html=True)  # Empty span for flexbox spacing
    
    # Home button
    st.markdown(f'<a href="{home_path}" target="_blank" class="nav-button home">🏠 Home</a>', unsafe_allow_html=True)
    
    # Next button
    if current_position < max(page_sequence.values()):
        next_page = next((k for k, v in page_sequence.items() if v == current_position + 1), None)
        if next_page:
            st.markdown(f'<a href="{app_prefix}{next_page}" target="_blank" class="nav-button">Next Step →</a>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="nav-button disabled">Next Step →</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span></span>', unsafe_allow_html=True)  # Empty span for flexbox spacing
    
    st.markdown('</div>', unsafe_allow_html=True)

# Call the function at the end of your page
add_navigation_buttons()