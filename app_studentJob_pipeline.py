import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# LOAD MODELS
# =========================
try:
    clf_model = joblib.load("student_placement_classifier.pkl")
    reg_model = joblib.load("student_placement_salary_regression.pkl")
    st.success("Models loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")


# =========================
# UI
# =========================
def main():
    st.title("Student Placement Prediction System")

    st.subheader("Input Student Features")


    gender = st.selectbox("Gender", ["Male", "Female"])
    ssc = st.number_input("SSC Percentage (%)", 0, 100)
    hsc = st.number_input("HSC Percentage (%)", 0, 100)
    degree = st.number_input("Degree Percentage (%)", 0, 100)
    cgpa = st.number_input("CGPA", 0.0, 10.0)
    entrance = st.number_input("Entrance Exam Score", 0, 100)
    tech = st.number_input("Technical Skill Score", 0, 100)
    soft = st.number_input("Soft Skill Score", 0, 100)
    internship = st.number_input("Internship Count", 0, 20)
    projects = st.number_input("Live Projects", 0, 20)
    work_exp = st.number_input("Work Experience (months)", 0, 60)
    cert = st.number_input("Certifications", 0, 20)
    attendance = st.number_input("Attendance Percentage", 0, 100)
    backlogs = st.number_input("Backlogs", 0, 10)
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

    # =========================
    # CREATE DATAFRAME
    # =========================
    data = {
        "gender": gender,
        "ssc_percentage": ssc,
        "hsc_percentage": hsc,
        "degree_percentage": degree,
        "cgpa": cgpa,
        "entrance_exam_score": entrance,
        "technical_skill_score": tech,
        "soft_skill_score": soft,
        "internship_count": internship,
        "live_projects": projects,
        "work_experience_months": work_exp,
        "certifications": cert,
        "attendance_percentage": attendance,
        "backlogs": backlogs,
        "extracurricular_activities": extracurricular
    }

    df = pd.DataFrame([data])

    # =========================
    # PREDICTION
    # =========================
    if st.button("Predict"):

        # 🔹 CLASSIFICATION
        placement = clf_model.predict(df)[0]

        #proba = clf_model.predict_proba(df)[0]

        st.subheader("Result")

        #st.write("Probability Not Placed (0):", proba[0])
        #st.write("Probability Placed (1):", proba[1])

        if placement == 1:
            st.success("Student is eligible for job placement")

            # 🔹 REGRESSION (salary prediction)
            salary = reg_model.predict(df)[0]
            st.info(f"Predicted Salary Package: {salary:.2f} LPA")

        else:
            st.error("Student is not eligible for job placement")
            st.warning("Salary prediction not available")


if __name__ == "__main__":
    main()