### 🧠 Predicting Completion of Clinical Trials with Explainability

> 🚀 **Top 3 in Use Case | Finalist (Top 12) - NEST 2025**

A data science project that predicts the likelihood of clinical trial completion using a blend of structured and unstructured data, advanced feature engineering, machine learning models, and explainability tools. This project was developed as part of the **NEST (National Educational Startup Tour)** initiative and emerged as one of the **Top 3 teams in its use case**, making it to the **Top 12 finalists** overall.

---

## 📁 Repository Structure


clinical-trial-prediction/
├── frontend/                  # Streamlit app and frontend code
├── backend/                   # Model code, pipelines, preprocessing, training scripts
├── csv/                       # All datasets used in the project
├── presentation_and_reports/  # Final reports, presentations, and documentation
└── README.md                  # This file

---

## 🧾 Project Overview

Clinical trials are crucial in the development of new drugs and treatments. However, a large number of them are terminated midway due to various operational and scientific reasons. This project aims to build a predictive system that:

- 📊 Forecasts whether a clinical trial will be completed or not  
- 🔍 Identifies key factors influencing trial success  
- 🧠 Provides transparent and interpretable results using SHAP & LIME  

---

## 📌 Key Highlights

- ✅ Merged structured + unstructured data using `nct_id`
- ✅ Feature engineering across 281+ columns using TF-IDF, Label Encoding, Target Encoding, and Feature Hashing
- ✅ Multi-label encoded study designs with `MultiLabelBinarizer`
- ✅ Dimensionality reduction and selection using LightGBM
- ✅ Final model: **Optuna-tuned XGBoost** with high accuracy
- ✅ Explainability with **SHAP** and **LIME**
- ✅ Clean, interactive **Streamlit app** to demonstrate functionality

---
## 🖥️ Live Application

> 🌐 *[https://clinicaltrials-vra2uu9pt5terqvpjgc6mr.streamlit.app/]*

---

## 🔍 Streamlit App Pages

1. **Clinical Trials Data Explorer**  
2. **Pivot Handling**  
3. **Adverse Event Clustering** *(BERT + UMAP + HDBSCAN)*  
4. **Missing Value Analysis**  
5. **Outlier Detection & Skewness Correction**  
6. **Feature Engineering Pipeline**  
7. **Healthcare Feature Reduction Journey**  
8. **ML Model Showcase with Explainability**

---

## 🧠 Tech Stack

- **Python** (Pandas, Scikit-learn, XGBoost, LightGBM, SHAP, LIME)
- **Streamlit** – for interactive UI
- **Optuna** – for hyperparameter tuning
- **UMAP + HDBSCAN** – for unsupervised clustering
- **BERT** – for NLP-based text embeddings
- **TargetEncoder, MultiLabelBinarizer** – for complex categorical encoding

---

## 📂 Datasets

All datasets are located in the `csv/` directory.

- Main dataset with **unique `nct_id`s**
- Additional text files for:
  - **Conditions**
  - **Interventions**
  - **Enrollment**
  - **Reasons for Termination**
- Final processed dataset has **281 features**

---

## 📊 Results

| Model                 | Accuracy | AUC-ROC |
|----------------------|----------|---------|
| Logistic Regression  | 85.23%   | 0.8012  |
| Random Forest        | 88.76%   | 0.8615  |
| LightGBM             | 89.41%   | 0.8723  |
| **XGBoost (Tuned)**  | **91.88%** | **0.8909** |

---

## 🏆 Recognition

- 🥉 **Top 3 Teams in Use Case (Clinical Trial Completion Prediction)**
- 🏅 **Top 12 Finalists Overall at NEST 2025**

---

## 👩‍💻 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/clinical-trial-prediction.git
   cd clinical-trial-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run frontend/app.py
   ```

4. **Explore the app in your browser**
   - Visit: `http://localhost:8501`

---

## 🤝 Team

**Team Name:** Team Ben10  

### 👩‍💻 Members

- **Yuvika Mishra**  
  - 🔗 [LinkedIn](https://www.linkedin.com/in/yuvika-mishra-9b7991258/)  
  - 📧 : yuvikasfs@gmail.com  

- **Siddhant Nijhawan**  
  - 🔗 [LinkedIn](https://www.linkedin.com/in/siddhant-nijhawan-453075255/) 
  - 📧 : siddhantnijhawan111@gmail.com

Feel free to connect with us on LinkedIn!
