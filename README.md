# Titanic Survival Prediction

โปรเจกต์ทำนายการรอดชีวิตของผู้โดยสารเรือ Titanic ด้วย Machine Learning (Classification)
พร้อม Web Application สำหรับใช้งานจริงด้วย Streamlit

## โครงสร้างโปรเจกต์

```
titanic-ml-project/
├── train_model_colab.py     # โค้ด train โมเดล (รันใน Google Colab)
├── app.py                    # Streamlit web application
├── requirements.txt          # รายชื่อ library ที่ต้องใช้
├── README.md
└── model/                    # ไฟล์โมเดลที่ train เสร็จแล้ว (ได้จาก Colab)
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    ├── le_sex.pkl
    ├── le_embarked.pkl
    ├── feature_names.json
    └── model_comparison.csv
```

## ขั้นตอนการทำงาน (Workflow)

### ขั้นที่ 1: Train โมเดลบน Google Colab
1. เปิด Google Colab (https://colab.research.google.com)
2. สร้าง Notebook ใหม่
3. เปิดไฟล์ `train_model_colab.py` แล้ว copy โค้ดแต่ละ `CELL` ไปวางเป็นคนละ cell ใน Colab
4. รันทีละ cell ตามลำดับ (CELL 1 → CELL 9)
5. Cell สุดท้ายจะ download ไฟล์ `model.zip` ที่รวมโมเดลทั้งหมดลงเครื่องอัตโนมัติ

### ขั้นที่ 2: เตรียมไฟล์เพื่อขึ้น GitHub
1. แตกไฟล์ `model.zip` ที่ได้ จะได้โฟลเดอร์ `model/` ที่มีไฟล์ .pkl, .json, .csv
2. นำโฟลเดอร์ `model/` มาวางในโปรเจกต์นี้ (ตำแหน่งเดียวกับ `app.py`)
3. สร้าง repository ใหม่บน GitHub แล้ว push ไฟล์ทั้งหมด:
   - `app.py`
   - `requirements.txt`
   - `model/` (ทั้งโฟลเดอร์)
   - `README.md`

```bash
git init
git add .
git commit -m "Initial commit: Titanic ML project"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

### ขั้นที่ 3: Deploy บน Streamlit Cloud
1. เข้า https://share.streamlit.io แล้ว login ด้วย GitHub
2. กด "New app" แล้วเลือก repository ที่เพิ่ง push ไป
3. ตั้งค่า Main file path เป็น `app.py`
4. กด Deploy รอสักครู่ก็จะได้ลิงก์เว็บไซต์ใช้งานจริง

### ขั้นที่ 4: รันทดสอบในเครื่องตัวเอง (ถ้าต้องการ)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## หมายเหตุ
- อย่าลืมแก้ไขข้อมูลผู้พัฒนาใน `app.py` (ส่วน sidebar) ให้เป็นข้อมูลของตัวเอง
- รูปภาพผู้พัฒนา แนะนำให้อัพโหลดไฟล์รูปใส่ในโปรเจกต์ (เช่น `images/profile.jpg`) แล้วแก้โค้ด
  `st.image("https://via.placeholder.com/150")` เป็น `st.image("images/profile.jpg")`
