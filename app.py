import streamlit as st
import pandas as pd
import pickle

# =========================
# DEPENDENCY CHECK
# =========================
missing_packages = []
for pkg in ("xgboost", "imblearn", "sklearn"):
    try:
        __import__(pkg)
    except ModuleNotFoundError:
        missing_packages.append(pkg)

if missing_packages:
    st.error(
        f"Missing required packages: {', '.join(missing_packages)}.\n"
        "Please install them using requirements.txt."
    )
    st.stop()

# =========================
# LOAD MODEL
# =========================
try:
    with open("best_xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found. Make sure 'best_xgboost_model.pkl' is in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(page_title="Hiring Prediction App", layout="centered")

st.title("💼 Candidate Hiring Prediction")
st.markdown("### 📊 Enter Candidate Details")

# =========================
# INPUTS
# =========================

# Basic Info
age = st.number_input("Age", 18, 60, 22)

cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
skills_score = st.number_input("Skills Score", 0.0, 100.0, 50.0)
soft_skills_score = st.number_input("Soft Skills Score", 0.0, 100.0, 50.0)

# Experience
internships = st.number_input("Internships", 0, 20, 1)
projects = st.number_input("Projects", 0, 50, 2)
certifications = st.number_input("Certifications", 0, 50, 1)
experience_years = st.number_input("Experience (Years)", 0.0, 20.0, 0.0)

hackathons = st.number_input("Hackathons", 0, 20, 0)
research_papers = st.number_input("Research Papers", 0, 20, 0)
programming_languages = st.number_input("Programming Languages Known", 0, 20, 3)

# Resume
resume_length_words = st.number_input("Resume Length (Words)", 100, 2000, 500)

# Feature Engineering
total_experience_score = internships + projects + certifications

# Categorical
education_level = st.selectbox("Education Level", ["bachelors", "masters", "phd"])
university_tier = st.selectbox("University Tier", ["tier 1", "tier 2", "tier 3"])
company_type = st.selectbox("Company Type", ["startup", "mid-size", "mnc"])

# =========================
# THRESHOLD CONTROL
# =========================
st.markdown("### 🎯 Adjust Prediction Sensitivity")
threshold = st.slider("Decision Threshold", 0.3, 0.9, 0.7)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    
    input_data = pd.DataFrame([{
        "age": age,
        "cgpa": cgpa,
        "skills_score": skills_score,
        "soft_skills_score": soft_skills_score,
        "internships": internships,
        "projects": projects,
        "certifications": certifications,
        "experience_years": experience_years,
        "hackathons": hackathons,
        "research_papers": research_papers,
        "programming_languages": programming_languages,
        "resume_length_words": resume_length_words,
        "total_experience_score": total_experience_score,
        "education_level": education_level,
        "university_tier": university_tier,
        "company_type": company_type
    }])

    try:
        proba = model.predict_proba(input_data)[0][1]
    except:
        proba = None

    # Decision Logic
    if proba is not None:
        if proba > threshold:
            st.success("✅ Candidate is likely to be HIRED")
        else:
            st.error("❌ Candidate is NOT likely to be hired")

        st.write(f"📊 Probability of Hiring: {proba*100:.2f}%")

    else:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Candidate is likely to be HIRED")
        else:
            st.error("❌ Candidate is NOT likely to be hired")
