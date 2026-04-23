import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout="wide")

# cek error
try:
    clf_model = joblib.load("student_placement_classifier.pkl")
    reg_model = joblib.load("student_placement_salary_regression.pkl")
    st.success("Models loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")


def main():
    st.title("Student Placement and Salary Prediction")

    st.subheader("Input Student Features")

    col1, col2, col3, col4 = st.columns([1.2, 1, 1.2, 1.2])

    # Academic
    with col1:
        st.subheader("📘 Academic")
        ssc = st.number_input("SSC (%)", 0, 100)
        hsc = st.number_input("HSC (%)", 0, 100)
        degree = st.number_input("Degree (%)", 0, 100)
        cgpa = st.number_input("Cumulative GPA", 0.0, 10.0)

    # Skills
    with col2:
        st.subheader("💻 Skills")
        tech = st.number_input("Technical Skill", 0, 100)
        soft = st.number_input("Soft Skill", 0, 100)

    # Experience
    with col3:
        st.subheader("📈 Experience")
        internship = st.number_input("Internship(s)", 0, 20)
        projects = st.number_input("Projects", 0, 20)
        work_exp = st.number_input("Work Exp (months)", 0, 100)
        cert = st.number_input("Certifications", 0, 20)

    # Others
    with col4:
        st.subheader("📊 Others")
        gender = st.selectbox("Gender", ["Male", "Female"])
        entrance = st.number_input("Entrance Score", 0, 100)
        attendance = st.number_input("Attendance (%)", 0, 100)
        backlogs = st.number_input("Backlogs", 0, 10)
        extracurricular = st.selectbox("Extracurricular", ["Yes", "No"])


    st.divider()

    col_btn1, col_btn2, col_btn3 = st.columns([2,1,2])
    with col_btn2:
        predict_clicked = st.button("Predict")

    # Enkapsulasi fitur menjadi dataframe
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

    # Prediction
    if predict_clicked:

        placement = clf_model.predict(df)[0]

        st.subheader("Result")

        if placement == 1:
            st.success("Student is eligible for job placement")

            salary = reg_model.predict(df)[0]
            st.info(f"Predicted Salary Package: {salary:.2f} LPA")

        else:
            st.error("Student is not eligible for job placement")
            st.warning("Salary prediction not available")


if __name__ == "__main__":
    main()