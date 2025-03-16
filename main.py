import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import pickle

# กำหนดค่าเริ่มต้นให้ st.session_state.current_page
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Data & Development"

# กำหนดคลาส PlacementNN ในระดับ global
class PlacementNN(nn.Module):
    def __init__(self):
        super(PlacementNN, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# ฟังก์ชันสำหรับ sub pages ใน Page 1
def subpage_main():
    st.header("แหล่งที่มาของข้อมูล ")
    st.markdown("""
        <style>
        .custom-text {
            line-height: 1.5;
            margin-bottom: 0.5rem;
            text-align: justify;
        }
        </style>
        <div class="custom-text">
            ข้อมูลที่เราใช้ในการวิเคราะห์และฝึกโมเดลจะเป็นชุดข้อมูลเกี่ยวกับหุ้นของบริษัท Tesla ตั้งแต่ปี 2010 จนถึง 2022 ซึ่งจะมีข้อมูลของราคาเปิดและปิดของตลาดในแต่ละช่วงเวลา โดยได้นำชุดข้อมูลนี้มาจากเว็บไซต์ Kaggle และจะนำมาเพื่อนำไปทำนายจำนวนที่จะขายได้จากราคาเปิดและปิด โดยนำข้อมูลชุดนี้มาใช้เพื่อศึกษาและพัฒนาโมเดลเท่านั้น
        </div>
    """, unsafe_allow_html=True)

def subpage_dataset():
    st.header("ข้อมูลใน Dataset")
    st.subheader("ข้อมูลในไฟล์ CSV")
    st.markdown("""
        <div class="custom-text">
            - Date ข้อมูลวันที่โดยจะเป็น ปี เดือน และวันของข้อมูล <br>
            - Open ราคาเปิดของหุ้น ณ วันนั้น <br>
            - High ราคาสูงสุดของหุ้น ณ วันนั้น <br>
            - Low ราคาต่ำสุดของหุ้น ณ วันนั้น <br>
            - Close ราคาปิดของหุ้น ณ วันนั้น <br>
            - Adj Close ราคาปิดที่ปรับเพื่อสะท้อนมูลค่าหลังจากหักค่าดำเนินการแล้ว <br>
            - Volume จำนวนหุ้นที่ขายไป ณ วันนั้น <br>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-bottom: 0.5rem;'>ตัวอย่างข้อมูลจากไฟล์ CSV</h3>", unsafe_allow_html=True)
    try:
        df = pd.read_csv('TSLA.csv')
        st.dataframe(df.head(100), height=300, use_container_width=True)
    except FileNotFoundError:
        st.warning("ไม่พบ TSLA.csv")

def subpage_dl():
    st.header("Development Model")
    st.write("Content about Deep Learning models")

def subpage_eval():
    st.header("Evaluation")
    st.write("Content about model evaluation techniques")

# ฟังก์ชันสำหรับหน้า Page 1 (ใช้ Tabs)
def page1():
    st.title("Data & Development 🔥")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Main",
        "Dataset Info",
        "Development Model",
        "Evaluation"
    ])
    
    with tab1:
        subpage_main()
    with tab2:
        subpage_dataset()
    with tab3:
        subpage_dl()
    with tab4:
        subpage_eval()

# ฟังก์ชันสำหรับหน้า Page 2
def page2():
    st.title("Machine Learning Models 💀")
    
    tab1, tab2 = st.tabs([
        "SVM Linear Model",
        "SVR Model"
    ])
    
    with tab1:
        st.header("SVM Linear Model")
        st.write("""
            โมเดล SVM Linear นี้ถูกฝึกด้วยข้อมูลหุ้น Tesla (TSLA.csv) โดยใช้ฟีเจอร์ Year, Month, Day, Open, High, Low, Volume 
            เพื่อทำนายราคาปิด (Close) หรือข้อมูลอื่นตามการฝึกโมเดล เป็นโมเดลที่เหมาะสำหรับข้อมูลที่สามารถแยกได้ด้วยเส้นตรง
        """)
        
        try:
            with open('svm_linear_model.pkl', 'rb') as file:
                svm_model = pickle.load(file)
        except FileNotFoundError:
            st.error("ไม่พบไฟล์ svm_linear_model.pkl")
            svm_model = None
        
        st.subheader("กรอกข้อมูลเพื่อทำนาย")
        year = st.number_input("Year", min_value=2010, max_value=2025, value=2022, step=1, key="svm_year")
        month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1, key="svm_month")
        day = st.number_input("Day", min_value=1, max_value=31, value=1, step=1, key="svm_day")
        open_price = st.number_input("Open Price", min_value=0.0, value=100.0, step=0.1, key="svm_open")
        high_price = st.number_input("High Price", min_value=0.0, value=110.0, step=0.1, key="svm_high")
        low_price = st.number_input("Low Price", min_value=0.0, value=90.0, step=0.1, key="svm_low")
        volume = st.number_input("Volume", min_value=0, value=100000, step=1000, key="svm_volume")
        
        if st.button("ทำนายด้วย SVM Linear", key="svm_predict"):
            if svm_model:
                input_data = [[year, month, day, open_price, high_price, low_price, volume]]
                prediction = svm_model.predict(input_data)
                st.write(f"ผลการทำนาย (อาจเป็นราคาปิด): {prediction[0]:.2f}")
            else:
                st.warning("ไม่สามารถทำนายได้ เนื่องจากโมเดลไม่ถูกโหลด")

    with tab2:
        st.header("SVR Model")
        st.write("""
            โมเดล SVR นี้ถูกฝึกด้วยข้อมูลหุ้น Tesla (TSLA.csv) โดยใช้ฟีเจอร์ Year, Month, Day, Open, High, Low, Volume 
            เพื่อทำนายราคาปิด (Close) หรือข้อมูลต่อเนื่องอื่น เป็นโมเดลที่เหมาะสำหรับการทำนายค่าเชิงปริมาณ
        """)
        
        try:
            with open('svr_model.pkl', 'rb') as file:
                svr_model = pickle.load(file)
        except FileNotFoundError:
            st.error("ไม่พบไฟล์ svr_model.pkl")
            svr_model = None
        
        st.subheader("กรอกข้อมูลเพื่อทำนาย")
        year = st.number_input("Year", min_value=2010, max_value=2025, value=2022, step=1, key="svr_year")
        month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1, key="svr_month")
        day = st.number_input("Day", min_value=1, max_value=31, value=1, step=1, key="svr_day")
        open_price = st.number_input("Open Price", min_value=0.0, value=100.0, step=0.1, key="svr_open")
        high_price = st.number_input("High Price", min_value=0.0, value=110.0, step=0.1, key="svr_high")
        low_price = st.number_input("Low Price", min_value=0.0, value=90.0, step=0.1, key="svr_low")
        volume = st.number_input("Volume", min_value=0, value=100000, step=1000, key="svr_volume")
        
        if st.button("ทำนายด้วย SVR", key="svr_predict"):
            if svr_model:
                input_data = [[year, month, day, open_price, high_price, low_price, volume]]
                prediction = svr_model.predict(input_data)
                st.write(f"ผลการทำนาย (อาจเป็นราคาปิด): {prediction[0]:.2f}")
            else:
                st.warning("ไม่สามารถทำนายได้ เนื่องจากโมเดลไม่ถูกโหลด")

# ฟังก์ชันสำหรับหน้า Page 3
def page3():
    st.title("Placement Opportunities Predictor 😠")
    st.write("""
        โมเดล Neural Network นี้ถูกฝึกด้วยข้อมูลการวางตำแหน่งงาน โดยใช้ฟีเจอร์ CGPA, Internships, Projects, Workshops/Certifications, 
        AptitudeTestScore, SoftSkillsRating, ExtracurricularActivities, PlacementTraining, SSC_Marks, HSC_Marks 
        เพื่อทำนายโอกาสการได้งาน (Placed หรือ Not Placed)
    """)

    st.subheader("กรอกข้อมูลเพื่อทำนาย")
    cgpa = st.number_input("CGPA (เกรดเฉลี่ยสะสม)", min_value=0.0, max_value=10.0, value=7.0, step=0.1, key="nn_cgpa")
    internships = st.number_input("จำนวนครั้งที่ฝึกงาน (Internships)", min_value=0, max_value=10, value=1, step=1, key="nn_internships")
    projects = st.number_input("จำนวนโปรเจกต์ (Projects)", min_value=0, max_value=20, value=2, step=1, key="nn_projects")
    workshops = st.number_input("จำนวน Workshops/Certifications", min_value=0, max_value=20, value=3, step=1, key="nn_workshops")
    aptitude_score = st.number_input("คะแนนทดสอบความถนัด (Aptitude Test Score)", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="nn_aptitude")
    soft_skills = st.number_input("คะแนนทักษะด้านอ่อน (Soft Skills Rating)", min_value=0.0, max_value=10.0, value=7.0, step=0.1, key="nn_softskills")
    extracurricular = st.selectbox("เข้าร่วมกิจกรรมนอกหลักสูตร (Extracurricular Activities)", options=["No", "Yes"], key="nn_extracurricular")
    placement_training = st.selectbox("เข้าร่วมการฝึกอบรมเพื่อการจ้างงาน (Placement Training)", options=["No", "Yes"], key="nn_training")
    ssc_marks = st.number_input("คะแนน SSC (มัธยมต้น)", min_value=0.0, max_value=100.0, value=80.0, step=1.0, key="nn_ssc")
    hsc_marks = st.number_input("คะแนน HSC (มัธยมปลาย)", min_value=0.0, max_value=100.0, value=85.0, step=1.0, key="nn_hsc")

    if st.button("ทำนายด้วย Neural Network", key="nn_predict"):
        try:
            # โหลด LabelEncoder
            label_encoder = joblib.load('label_encoder.pkl')
            # โหลด Scaler
            scaler = joblib.load('scaler.pkl')
            # โหลดโมเดล PyTorch
            model = PlacementNN()
            model.load_state_dict(torch.load('placement_model.pkl', weights_only=False))
            model.eval()

            # แปลงข้อมูล categorical เป็น numerical
            extracurricular_num = label_encoder.transform([extracurricular])[0]
            placement_training_num = label_encoder.transform([placement_training])[0]

            # เตรียมข้อมูล input
            input_data = np.array([[cgpa, internships, projects, workshops, aptitude_score, 
                                   soft_skills, extracurricular_num, placement_training_num, ssc_marks, hsc_marks]])
            
            # ปรับสเกลข้อมูล
            input_data_scaled = scaler.transform(input_data)

            # แปลงเป็น Tensor
            input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
            
            # ทำนาย
            with torch.no_grad():
                output = model(input_tensor)
                probability = torch.sigmoid(output).item()
            
            result = "Placed" if probability >= 0.5 else "Not Placed"
            st.success(f"ผลการทำนาย: **{result}** (ความน่าจะเป็น: {probability:.2f})")
        except FileNotFoundError as e:
            st.error(f"ไม่พบไฟล์ที่ต้องการ: {str(e)}. กรุณาตรวจสอบว่าไฟล์ 'placement_model.pkl', 'scaler.pkl', และ 'label_encoder.pkl' อยู่ในโฟลเดอร์ที่ถูกต้อง")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")

# CSS เพื่อปรับแต่งทั้งปุ่มและ tabs ให้ minimal และกึ่งกลาง
st.markdown("""
    <style>
    .nav-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    .stButton > button {
        width: 200px;
        height: 50px;
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
        color: #333333;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #f0f0f0;
        border-color: #aaaaaa;
    }
    .stButton > button:active {
        background-color: #e0e0e0;
        border-color: #888888;
    }
    .stColumn > div {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stTabs [role="tab"] {
        min-width: 150px;
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
        color: #333333;
        margin: 0 5px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    .stTabs [role="tab"]:hover {
        background-color: #f0f0f0;
        border-color: #aaaaaa;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #e0e0e0;
        border-color: #888888;
        color: #333333;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# หัวข้อหลัก
st.markdown("<h1 style='text-align: center; color: #333;'>Welcome to Placement Prediction App</h1>", unsafe_allow_html=True)

# สร้างกรอบปุ่มด้วย container
with st.container():
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    cols = st.columns(3)
    
    with cols[0]:
        if st.button("🔥 Data & Development"):
            st.session_state.current_page = "Data & Development"
    
    with cols[1]:
        if st.button("💀 Machine Learning Models"):
            st.session_state.current_page = "Machine Learning Models"
    
    with cols[2]:
        if st.button("😠 Neural Network Model"):
            st.session_state.current_page = "Neural Network Model"
    
    st.markdown('</div>', unsafe_allow_html=True)

# แสดงเนื้อหาตามหน้าที่เลือก
if st.session_state.current_page == "Data & Development":
    page1()
elif st.session_state.current_page == "Machine Learning Models":
    page2()
elif st.session_state.current_page == "Neural Network Model":
    page3()