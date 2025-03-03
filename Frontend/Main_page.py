import streamlit as st

# Set page configuration
st.set_page_config(
    page_title='Main page',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Define the correct paths to your pages
# If your files are in the root directory, use:
# page_paths = {
#     "Data Preprocessing": "/app1.py",
#     "Exploratory Data Analysis": "/app2.py",
#     ...
# }
# 
# If your files are in a 'pages' subdirectory, use:
page_paths = {
    "Clinical Trials Data Explorer": "pages/1_Clinical Trials Data Explorer.py",
    "Pivot Handling": "pages/2_Pivot Handling.py",
    "Adverse Event Clustering": "pages/3_Adverse Event Clustering.py",
    "Missing Value Analysis": "pages/4_Missing Value Analysis.py",
    "Outlier Detection & Skewness Correction": "pages/5_Outlier Detection & Skewness Correction.py",
    "Feature Engineering Pipeline": "pages/7_Feature Engineering Pipeline.py",
    "Healthcare Data Feature Reduction Journey": "pages/8_Healthcare Data Feature Reduction Journey.py",
    "ML Models Showcase": "pages/9_ML Models Showcase.py"
}

# Custom CSS for animations and styling
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header styling with gradient */
    .header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxkZWZzPjxwYXR0ZXJuIGlkPSJwYXR0ZXJuIiB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHBhdHRlcm5Vbml0cz0idXNlclNwYWNlT25Vc2UiIHBhdHRlcm5UcmFuc2Zvcm09InJvdGF0ZSg0NSkiPjxyZWN0IGlkPSJwYXR0ZXJuLWJhY2tncm91bmQiIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9InRyYW5zcGFyZW50Ij48L3JlY3Q+PGNpcmNsZSBjeD0iMjAiIGN5PSIyMCIgcj0iMSIgZmlsbD0icmdiYSgyNTUsMjU1LDI1NSwwLjA3KSI+PC9jaXJjbGU+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI3BhdHRlcm4pIj48L3JlY3Q+PC9zdmc+') repeat;
        opacity: 0.2;
    }
    
    .header h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .header p {
        font-size: 1.2rem;
        max-width: 800px;
        opacity: 0.95;
    }
    
    /* Card container */
    .card-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    /* Card styling with hover effects */
    .card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        height: 240px;
    }
    
    .card:hover {
        transform: translateY(-10px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
    }
    
    .card:hover .card-overlay {
        opacity: 1;
    }
    
    .card-icon {
        font-size: 2.8rem;
        color: #4a6fa5;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.8rem;
    }
    
    .card-description {
        font-size: 0.9rem;
        color: #7f8c8d;
        line-height: 1.5;
    }
    
    .card-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(to top, rgba(74, 111, 165, 0.9), rgba(74, 111, 165, 0.7));
        padding: 1rem;
        color: white;
        opacity: 0;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .card-overlay-content {
        text-align: center;
    }
    
    .card-button {
        background: #ffffff;
        color: #4a6fa5;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 0.5rem;
    }
    
    .card-button:hover {
        background: #f8f9fa;
        transform: scale(1.05);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        font-size: 0.9rem;
        color: #95a5a6;
        border-top: 1px solid #ecf0f1;
        animation: fadeIn 2s ease-in-out;
    }
    
    .footer-logo {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #34495e;
    }
    
    /* General animations */
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    
    @keyframes slideInUp {
        from {
            transform: translateY(50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .animated {
        animation: slideInUp 0.5s ease forwards;
    }
    
    /* Responsive styles */
    @media screen and (max-width: 768px) {
        .card-container {
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        }
    }
</style>
""", unsafe_allow_html=True)

# Main content container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Animated header
st.markdown("""
<div class="header animated">
    <h1>🏥 Clinical Trials Analysis Dashboard</h1>
    <p>Welcome to the advanced Clinical Trials Prediction and Analysis platform. This interactive tool helps researchers, clinicians, and pharmaceutical companies gain actionable insights from clinical trial data, optimize study designs, and predict outcomes with machine learning.</p>
</div>
""", unsafe_allow_html=True)

# Define pages with enhanced descriptions and icons
pages = [
    {
        "title": "Clinical Trials Data Explorer",
        "icon": "📊",
        "description": "Comprehensive analysis dashboard for clinical trial datasets",
    },
    {
        "title": "Pivot Handling",
        "icon": "🔄",
        "description": "Dynamic data transformation tools for cross-tabulation analysis",
    },
    {
        "title": "Adverse Event Clustering",
        "icon": "🔍",
        "description": "Automated identification of related adverse event patterns",
    },
    {
        "title": "Missing Value Analysis",
        "icon": "❓",
        "description": "Statistical assessment of data completeness and imputation methods",
    },
    {
        "title": "Outlier Detection & Skewness Correction",
        "icon": "📉",
        "description": "Automated identification and normalization of statistical anomalies",
    },
    {
        "title": "Feature Engineering Pipeline",
        "icon": "⚙️",
        "description": "Systematic workflow for creating optimized model variables",
    },
    {
        "title": "Healthcare Data Feature Reduction Journey",
        "icon": "🧩",
        "description": "Dimensionality reduction techniques for clinical datasets",
    },
    {
        "title": "ML Models Showcase",
        "icon": "🚀",
        "description": "Performance visualization of deployed clinical prediction models",
    }
]
# Generate cards for workflow steps
st.markdown('<div class="card-container">', unsafe_allow_html=True)

for i, page in enumerate(pages):
    # Get the correct path for this page
    page_path = page_paths[page['title']]
    
    st.markdown(f"""
    <div class="card animated" onclick="window.open('{page_path}', '_blank')">
        <div class="card-icon">{page['icon']}</div>
        <div class="card-title">{page['title']}</div>
        <div class="card-description">{page['description']}</div>
        <div class="card-overlay">
            <div class="card-overlay-content">
                <p>Click to explore</p>
                <button class="card-button" onclick="window.open('{page_path}', '_blank'); event.stopPropagation();">Open {page['title']}</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Enhanced footer
st.markdown("""
<div class="footer">
    <div class="footer-logo">Clinical Trials Analysis Platform</div>
    <p>Developed with ❤️ by Siddhant Nijhawan</p>
    <p>© 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)

# Add JavaScript for interactivity
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Animate cards on scroll
    const cards = document.querySelectorAll('.card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = 1;
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, {threshold: 0.1});
    
    cards.forEach(card => {
        card.style.opacity = 0;
        card.style.transform = 'translateY(50px)';
        observer.observe(card);
        
        // Ensure clicking anywhere on the card opens the page in a new tab
        card.addEventListener('click', function(e) {
            const link = this.getAttribute('onclick').match(/'([^']+)'/)[1];
            window.open(link, '_blank');
        });
    });
});
</script>
""", unsafe_allow_html=True)