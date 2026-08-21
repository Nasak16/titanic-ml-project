import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")

# ==========================================================
# โหลดโมเดลและไฟล์ที่ Train มาจาก Colab (อยู่ในโฟลเดอร์ model/)
# ==========================================================
@st.cache_resource
def load_artifacts():
    log_reg = joblib.load("model/logistic_regression.pkl")
    dtree = joblib.load("model/decision_tree.pkl")
    rf = joblib.load("model/random_forest.pkl")
    scaler = joblib.load("model/scaler.pkl")
    le_sex = joblib.load("model/le_sex.pkl")
    le_embarked = joblib.load("model/le_embarked.pkl")
    with open("model/feature_names.json") as f:
        feature_names = json.load(f)
    comparison_df = pd.read_csv("model/model_comparison.csv")
    return log_reg, dtree, rf, scaler, le_sex, le_embarked, feature_names, comparison_df

log_reg, dtree, rf, scaler, le_sex, le_embarked, feature_names, comparison_df = load_artifacts()

models = {
    "Logistic Regression": log_reg,
    "Decision Tree": dtree,
    "Random Forest": rf
}

# ==========================================================
# SIDEBAR: ข้อมูลผู้พัฒนา
# ==========================================================
with st.sidebar:
    st.header("ข้อมูลผู้พัฒนา")
    # เปลี่ยนรูป/ข้อมูลตรงนี้ให้เป็นของตัวเอง
    st.image("https://via.placeholder.com/150", caption="รูปผู้พัฒนา")
    st.markdown("""
    **รหัสนักศึกษา:** 664245007
    
    **ชื่อ นามสกุล:** ณศักดิ์ ฉายแสงรัตน์  
    **หมู่เรียน:** 66/43
    """)
    st.divider()
    page = st.radio("เมนู", [
        "1. Dataset",
        "2. Data Preprocessing",
        "3. โมเดล ML",
        "4. เปรียบเทียบโมเดล",
        "5. ทำนายผล (Prediction)"
    ])

st.title("🚢 Titanic Survival Prediction")

# ==========================================================
# หน้า 1: การกำหนดปัญหาและ Dataset
# ==========================================================
if page == "1. Dataset":
    st.header("1. การกำหนดปัญหาและ Dataset")
    st.markdown("""
    **โจทย์ปัญหา:** ทำนายว่าผู้โดยสารเรือ Titanic จะ**รอดชีวิต**หรือ**ไม่รอดชีวิต**
    จากข้อมูลส่วนตัว เช่น เพศ อายุ ชั้นโดยสาร เป็นต้น (ปัญหาแบบ Classification 2 กลุ่ม)

    **ที่มาของ Dataset:** Titanic Dataset (Kaggle / Seaborn built-in dataset)

    **Feature ที่ใช้ในการทำนาย:**
    - `pclass` : ชั้นโดยสาร (1, 2, 3)
    - `sex` : เพศ
    - `age` : อายุ
    - `sibsp` : จำนวนพี่น้อง/คู่สมรสที่มาด้วย
    - `parch` : จำนวนพ่อแม่/ลูกที่มาด้วย
    - `fare` : ค่าตั๋วโดยสาร
    - `embarked` : ท่าเรือที่ขึ้นเรือ

    **เหตุผลที่เลือก Dataset นี้:** มีข้อมูลทั้งตัวเลขและข้อความ (categorical) เหมาะสำหรับฝึก
    การทำ Data Preprocessing และเป็นปัญหาที่เข้าใจง่าย เห็นผลลัพธ์ชัดเจน
    """)

# ==========================================================
# หน้า 2: Data Preprocessing
# ==========================================================
elif page == "2. Data Preprocessing":
    st.header("2. Data Preprocessing")
    st.markdown("""
    ขั้นตอนการเตรียมข้อมูลก่อนนำไปเทรนโมเดล:

    1. **จัดการ Missing Values**
       - คอลัมน์ `age` ที่ขาดหาย → เติมด้วยค่ามัธยฐาน (median)
       - คอลัมน์ `embarked` ที่ขาดหาย → เติมด้วยค่าที่พบบ่อยที่สุด (mode)
       - คอลัมน์ `fare` ที่ขาดหาย → เติมด้วยค่ามัธยฐาน

    2. **Encoding**
       - `sex` : male/female → แปลงเป็น 0/1 ด้วย Label Encoding
       - `embarked` : แปลงตัวอักษรท่าเรือ → ตัวเลข ด้วย Label Encoding

    3. **Feature Scaling**
       - ใช้ `StandardScaler` ปรับค่าตัวเลข (age, fare ฯลฯ) ให้อยู่ในสเกลเดียวกัน
         (ใช้กับโมเดล Logistic Regression)

    4. **แบ่งข้อมูล Train/Test**
       - แบ่งเป็น Train 80% และ Test 20% แบบ Stratified (รักษาสัดส่วนคลาส)
    """)
    st.info(f"Feature ที่ใช้ทั้งหมด: {feature_names}")

# ==========================================================
# หน้า 3: การสร้างโมเดล ML
# ==========================================================
elif page == "3. โมเดล ML":
    st.header("3. การสร้างโมเดล Machine Learning")

    st.subheader("1) Logistic Regression")
    st.markdown("""
    โมเดลเชิงเส้นที่ใช้ทำนายความน่าจะเป็น โดยผ่านฟังก์ชัน **Sigmoid** ที่แปลงค่าผลลัพธ์
    ให้อยู่ระหว่าง 0-1 แล้วตั้ง threshold ที่ 0.5 เพื่อแบ่งเป็น 2 คลาส (รอด/ไม่รอด)
    """)

    st.subheader("2) Decision Tree")
    st.markdown("""
    โมเดลที่ทำงานแบบ**แตกกิ่งคำถาม (if-else)** ทีละขั้น เช่น "เพศเป็นหญิงหรือไม่"
    แล้ว "ชั้นโดยสารคือ 1 หรือไม่" ไปเรื่อยๆ จนได้คำตอบสุดท้าย ข้อดีคืออธิบายผลได้ง่าย เห็นภาพชัดเจน
    """)

    st.subheader("3) Random Forest")
    st.markdown("""
    โมเดลแบบ **Ensemble** ที่สร้าง Decision Tree หลายๆ ต้นแบบสุ่ม แล้วนำผลของทุกต้นมา**โหวต**
    เพื่อหาคำตอบสุดท้าย ช่วยลดปัญหา Overfitting และมักให้ความแม่นยำสูงกว่า Decision Tree เดี่ยวๆ
    """)

# ==========================================================
# หน้า 4: การประเมินและเปรียบเทียบโมเดล
# ==========================================================
elif page == "4. เปรียบเทียบโมเดล":
    st.header("4. การประเมินและเปรียบเทียบโมเดล")

    st.subheader("ตารางเปรียบเทียบ")
    st.dataframe(comparison_df, use_container_width=True)

    st.subheader("กราฟเปรียบเทียบ")
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", ax=ax
    )
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    best_model = comparison_df.sort_values("Accuracy", ascending=False).iloc[0]
    st.success(f"โมเดลที่ดีที่สุด: **{best_model['Model']}** (Accuracy = {best_model['Accuracy']})")

# ==========================================================
# หน้า 5: Streamlit Application - ทำนายผล
# ==========================================================
elif page == "5. ทำนายผล (Prediction)":
    st.header("5. ทำนายผลการรอดชีวิต")
    st.markdown("กรอกข้อมูลผู้โดยสารด้านล่าง แล้วกดปุ่มเพื่อทำนายผล")

    selected_model_name = st.selectbox("เลือกโมเดลที่ต้องการใช้ทำนาย", list(models.keys()))

    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("ชั้นโดยสาร (Pclass)", [1, 2, 3], index=2)
        sex = st.selectbox("เพศ", ["male", "female"])
        age = st.slider("อายุ", 0, 90, 30)
        fare = st.number_input("ค่าตั๋วโดยสาร (Fare)", min_value=0.0, max_value=600.0, value=32.0)
    with col2:
        sibsp = st.number_input("จำนวนพี่น้อง/คู่สมรสที่มาด้วย (SibSp)", min_value=0, max_value=10, value=0)
        parch = st.number_input("จำนวนพ่อแม่/ลูกที่มาด้วย (Parch)", min_value=0, max_value=10, value=0)
        embarked = st.selectbox("ท่าเรือที่ขึ้นเรือ (Embarked)", ["S", "C", "Q"])

    if st.button("🔮 ทำนายผล", type="primary"):
        sex_enc = le_sex.transform([sex])[0]
        embarked_enc = le_embarked.transform([embarked])[0]

        input_df = pd.DataFrame([{
            "pclass": pclass,
            "sex": sex_enc,
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "embarked": embarked_enc
        }])[feature_names]

        model = models[selected_model_name]

        if selected_model_name == "Logistic Regression":
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]
        else:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]

        st.divider()
        if pred == 1:
            st.success(f"### ผลการทำนาย: รอดชีวิต ✅ (ความมั่นใจ {prob[1]*100:.1f}%)")
        else:
            st.error(f"### ผลการทำนาย: ไม่รอดชีวิต ❌ (ความมั่นใจ {prob[0]*100:.1f}%)")
