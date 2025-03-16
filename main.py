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

# กำหนดค่าเริ่มต้นให้ st.session_state.dataset_view
if 'dataset_view' not in st.session_state:
    st.session_state.dataset_view = "Tesla"  # ค่าเริ่มต้นเป็น Tesla Dataset

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
        <h1>Dataset หุ้น Tesla</h1><br>
        <div class="custom-text">
            ข้อมูลที่เราใช้ในการวิเคราะห์และฝึกโมเดลจะเป็นชุดข้อมูลเกี่ยวกับหุ้นของบริษัท Tesla ตั้งแต่ปี 2010 จนถึง 2022 ซึ่งจะมีข้อมูลของราคาเปิดและปิดของตลาดในแต่ละช่วงเวลา โดยได้นำชุดข้อมูลนี้มาจากเว็บไซต์ Kaggle และจะนำมาเพื่อนำไปทำนายจำนวนที่จะขายได้จากราคาเปิดและปิด โดยนำข้อมูลชุดนี้มาใช้เพื่อศึกษาและพัฒนาโมเดลเท่านั้น
        </div><br>
        <p>ที่มา : https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021/data</p>
        <h1>Dataset อัตราการจ้างงานจากผลการเรียน</h1><br>
        <div class="custom-text">
            ข้อมูลที่เราใช้ในการวิเคราะห์และฝึกโมเดลจะเป็นชุดข้อมูลเกี่ยวกับสถิติการเรียน การทำการบ้าน เกรดเฉลี่ย ผลการเรียนมาวิเคราะห์และทำนายว่าจากข้อมูลทั้งหมดมีสิทธิ์ได้รับการจ้างงานหรือไม่ โดยได้นำชุดข้อมูลนี้มาจากเว็บไซต์ Kaggle โดยนำข้อมูลชุดนี้มาใช้เพื่อศึกษาและพัฒนาโมเดลเท่านั้น
        </div><br>
        <p>ที่มา : https://www.kaggle.com/datasets/ruchikakumbhar/placement-prediction-dataset/data</p>
    """, unsafe_allow_html=True)

# ฟังก์ชันสำหรับแสดง Dataset หุ้น Tesla
def display_tesla_dataset():
    st.header("ข้อมูลใน Dataset หุ้นบริษัท Tesla")
    st.markdown("""
        <style>
        .custom-text {
            line-height: 1.5;
            margin-bottom: 0.5rem;
            text-align: justify;
        }
        .info-box {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        .feature-title {
            font-weight: bold;
            color: #2c3e50;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-size: 1.2em;
        }
        .feature-desc {
            margin-left: 1.5rem;
            color: #34495e;
            font-size: 1em;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("คอลัมน์ (Features) ใน Dataset")
    st.markdown("""
        <div class="info-box">
            <div class="custom-text">
                <div class="feature-title">Date</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> วันที่ของการซื้อขายหุ้น<br>
                    <strong>ประเภทข้อมูล:</strong> String (ในรูปแบบ YYYY-MM-DD)<br>
                    <strong>ตัวอย่าง:</strong> "2010-06-29"<br>
                    <strong>ความหมาย:</strong> ระบุวันที่ที่ข้อมูลราคาหุ้นถูกบันทึก
                </div>
                <div class="feature-title">Open</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> ราคาเปิดของหุ้นในวันนั้น<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 1.2666670083999634 (วันที่ 2010-06-29)<br>
                    <strong>ความหมาย:</strong> ราคาแรกที่หุ้นถูกซื้อขายในวันนั้น
                </div>
                <div class="feature-title">High</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> ราคาสูงสุดของหุ้นในวันนั้น<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 1.6666669845581057 (วันที่ 2010-06-29)<br>
                    <strong>ความหมาย:</strong> ราคาสูงสุดที่หุ้นทำได้ในระหว่างวัน
                </div>
                <div class="feature-title">Low</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> ราคาต่ำสุดของหุ้นในวันนั้น<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 1.1693329811096191 (วันที่ 2010-06-29)<br>
                    <strong>ความหมาย:</strong> ราคาต่ำสุดที่หุ้นทำได้ในระหว่างวัน
                </div>
                <div class="feature-title">Close</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> ราคาปิดของหุ้นในวันนั้น<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 1.5926669836044312 (วันที่ 2010-06-29)<br>
                    <strong>ความหมาย:</strong> ราคาสุดท้ายที่หุ้นถูกซื้อขายในวันนั้น
                </div>
                <div class="feature-title">Adj Close</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> ราคาปิดที่ปรับปรุงแล้ว (Adjusted Close)<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 1.5926669836044312 (วันที่ 2010-06-29)<br>
                    <strong>ความหมาย:</strong> ราคาปิดที่ถูกปรับให้สะท้อนการเปลี่ยนแปลง เช่น การจ่ายปันผลหรือการแตกหุ้น
                </div>
                <div class="feature-title">Volume</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> ปริมาณการซื้อขายหุ้นในวันนั้น<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 281494500.0 (วันที่ 2010-06-29)<br>
                    <strong>ความหมาย:</strong> จำนวนหุ้นที่ถูกซื้อขายในวันนั้น ช่วยบ่งบอกถึงสภาพคล่อง (Liquidity)
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("ลักษณะเด่นของ Dataset")
    st.markdown("""
        <div class="info-box">
            <div class="custom-text">
                <div class="feature-desc">
                    <strong>ระยะเวลา:</strong> ข้อมูลครอบคลุมตั้งแต่ปี 2010 ถึง 2025 (บางส่วนอาจเป็นข้อมูลสมมติในอนาคต)<br>
                    <strong>ความถี่:</strong> ข้อมูลมีความถี่รายวัน (Daily)<br>
                    <strong>ข้อมูลที่ขาดหาย:</strong> บางแถวมีคอลัมน์ว่าง ซึ่งอาจเกิดจากวันหยุดตลาด<br>
                    <strong>แนวโน้มราคา:</strong> ราคาหุ้นมีแนวโน้มเพิ่มขึ้นอย่างมากเมื่อเวลาผ่านไป
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("การใช้งาน Dataset")
    st.markdown("""
        <div class="info-box">
            <div class="custom-text">
                <div class="feature-desc">
                    <p><strong>Dataset นี้สามารถนำไปใช้ในการวิเคราะห์ทางการเงินได้หลากหลาย</strong></p>
                    <strong>การวิเคราะห์แนวโน้มราคา:</strong> ดูการเคลื่อนไหวของราคา Open, High, Low, Close<br>
                    <strong>การคำนวณความผันผวน:</strong> ใช้ High และ Low เพื่อประเมินความผันผวน (Volatility)<br>
                    <strong>การวิเคราะห์ปริมาณการซื้อขาย:</strong> ใช้ Volume เพื่อดูความสนใจของนักลงทุน<br>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='margin-bottom: 0.5rem;'>ตัวอย่างข้อมูลจากไฟล์ CSV</h3>", unsafe_allow_html=True)
    try:
        df = pd.read_csv('dataset/TSLA.csv')
        st.dataframe(df.head(100), height=300, use_container_width=True)
    except FileNotFoundError:
        st.warning("ไม่พบไฟล์ TSLA.csv")

# ฟังก์ชันสำหรับแสดง Dataset Placement Prediction
def display_placement_dataset():
    st.header("ข้อมูลใน Dataset อัตราการจ้างงานจากผลการเรียน")
    st.markdown("""
        <style>
        .custom-text {
            line-height: 1.5;
            margin-bottom: 0.5rem;
            text-align: justify;
        }
        .info-box {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        .feature-title {
            font-weight: bold;
            color: #2c3e50;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-size: 1.2em;
        }
        .feature-desc {
            margin-left: 1.5rem;
            color: #34495e;
            font-size: 1em;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("คอลัมน์ (Features) ใน Dataset")
    st.markdown("""
        <div class="info-box">
            <div class="custom-text">
                <div class="feature-title">CGPA</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> เกรดเฉลี่ยสะสม<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 7.5<br>
                    <strong>ความหมาย:</strong> คะแนนเฉลี่ยสะสมของนักเรียน
                </div>
                <div class="feature-title">Internships</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> จำนวนครั้งที่ฝึกงาน<br>
                    <strong>ประเภทข้อมูล:</strong> Integer<br>
                    <strong>ตัวอย่าง:</strong> 2<br>
                    <strong>ความหมาย:</strong> จำนวนครั้งที่นักเรียนเข้าร่วมฝึกงาน
                </div>
                <div class="feature-title">Projects</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> จำนวนโปรเจกต์<br>
                    <strong>ประเภทข้อมูล:</strong> Integer<br>
                    <strong>ตัวอย่าง:</strong> 3<br>
                    <strong>ความหมาย:</strong> จำนวนโปรเจกต์ที่นักเรียนทำ
                </div>
                <div class="feature-title">Workshops/Certifications</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> จำนวนการเข้าร่วมเวิร์กช็อปหรือใบรับรอง<br>
                    <strong>ประเภทข้อมูล:</strong> Integer<br>
                    <strong>ตัวอย่าง:</strong> 4<br>
                    <strong>ความหมาย:</strong> จำนวนครั้งที่เข้าร่วมเวิร์กช็อปหรือได้รับใบรับรอง
                </div>
                <div class="feature-title">AptitudeTestScore</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> คะแนนทดสอบความถนัด<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 75.0<br>
                    <strong>ความหมาย:</strong> คะแนนจากการทดสอบความถนัด
                </div>
                <div class="feature-title">SoftSkillsRating</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> คะแนนทักษะด้านอ่อน<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 8.0<br>
                    <strong>ความหมาย:</strong> การประเมินทักษะด้านอ่อน (เช่น การสื่อสาร)
                </div>
                <div class="feature-title">ExtracurricularActivities</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> การเข้าร่วมกิจกรรมนอกหลักสูตร<br>
                    <strong>ประเภทข้อมูล:</strong> String (Yes/No)<br>
                    <strong>ตัวอย่าง:</strong> "Yes"<br>
                    <strong>ความหมาย:</strong> ระบุว่านักเรียนเข้าร่วมกิจกรรมนอกหลักสูตรหรือไม่
                </div>
                <div class="feature-title">PlacementTraining</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> การเข้าร่วมการฝึกอบรมเพื่อการจ้างงาน<br>
                    <strong>ประเภทข้อมูล:</strong> String (Yes/No)<br>
                    <strong>ตัวอย่าง:</strong> "No"<br>
                    <strong>ความหมาย:</strong> ระบุว่านักเรียนเข้ารับการฝึกอบรมเพื่อการจ้างงานหรือไม่
                </div>
                <div class="feature-title">SSC_Marks</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> คะแนนสอบมัธยมต้น<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 85.0<br>
                    <strong>ความหมาย:</strong> คะแนนสอบระดับมัธยมต้น
                </div>
                <div class="feature-title">HSC_Marks</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> คะแนนสอบมัธยมปลาย<br>
                    <strong>ประเภทข้อมูล:</strong> Float<br>
                    <strong>ตัวอย่าง:</strong> 88.0<br>
                    <strong>ความหมาย:</strong> คะแนนสอบระดับมัธยมปลาย
                </div>
                <div class="feature-title">PlacementStatus</div>
                <div class="feature-desc">
                    <strong>คำอธิบาย:</strong> สถานะการจ้างงาน<br>
                    <strong>ประเภทข้อมูล:</strong> String (Placed/Not Placed)<br>
                    <strong>ตัวอย่าง:</strong> "Placed"<br>
                    <strong>ความหมาย:</strong> ผลลัพธ์ว่าสามารถหางานได้หรือไม่
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("ลักษณะเด่นของ Dataset")
    st.markdown("""
        <div class="info-box">
            <div class="custom-text">
                <div class="feature-desc">
                    <strong>จำนวนข้อมูล:</strong> ประกอบด้วยข้อมูลนักเรียนหลายร้อยรายการ<br>
                    <strong>ข้อมูลที่ขาดหาย:</strong> บางแถวอาจมีค่าสูญหายในบางคอลัมน์<br>
                    <strong>เป้าหมาย:</strong> ใช้เพื่อทำนายโอกาสการได้งานจากผลการเรียนและกิจกรรม
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("การใช้งาน Dataset")
    st.markdown("""
        <div class="info-box">
            <div class="custom-text">
                <div class="feature-desc">
                    <p><strong>Dataset นี้สามารถนำไปใช้ในการวิเคราะห์และทำนายโอกาสการจ้างงาน</strong></p>
                    <strong>การวิเคราะห์ผลการเรียน:</strong> ดูความสัมพันธ์ระหว่างเกรดและการได้งาน<br>
                    <strong>การประเมินทักษะ:</strong> วิเคราะห์ผลกระทบของทักษะและกิจกรรมต่อการจ้างงาน<br>
                    <strong>การสร้างโมเดล:</strong> ฝึกโมเดล Machine Learning เพื่อพยากรณ์สถานะการจ้างงาน<br>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='margin-bottom: 0.5rem;'>ตัวอย่างข้อมูลจากไฟล์ CSV</h3>", unsafe_allow_html=True)
    try:
        df = pd.read_csv('dataset/placementdata_miss.csv')
        st.dataframe(df.head(100), height=300, use_container_width=True)
    except FileNotFoundError:
        st.warning("ไม่พบไฟล์ placementdata_miss.csv")

# ฟังก์ชัน subpage_dataset() ที่รวมทั้งสอง Dataset
def subpage_dataset():
    st.header("Dataset")

    # ปุ่มเลือก Dataset
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Dataset หุ้น Tesla"):
            st.session_state.dataset_view = "Tesla"
    with col2:
        if st.button("Dataset Placement Prediction"):
            st.session_state.dataset_view = "Placement"

    # แสดง Dataset ตามที่เลือก
    if st.session_state.dataset_view == "Tesla":
        display_tesla_dataset()
    elif st.session_state.dataset_view == "Placement":
        display_placement_dataset()
    st.markdown('</div>', unsafe_allow_html=True)

def subpage_dl():
    st.header("แนวทางการสร้างโปรเจค")

    # CSS สำหรับการจัดรูปแบบ
    st.markdown("""
        <style>
        .custom-text {
            line-height: 1.5;
            margin-bottom: 0.5rem;
            text-align: justify;
        }
        .info-box {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        .section-title {
            font-weight: bold;
            color: #2c3e50;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-size: 1.3em;
        }
        .section-desc {
            margin-left: 1.5rem;
            color: #34495e;
            font-size: 1em;
        }
        .highlight {
            color: #2980b9;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # เนื้อหาแนวทางการพัฒนาโมเดลทั้ง 5 ข้อ
    st.markdown("""
        <div class="info-box">
            <div class="custom-text">
                <div class="section-title">1. การเลือกและพัฒนาโมเดล</div>
                <div class="section-desc">
                    ด้วยเป้าหมายที่แตกต่างกันของชุดข้อมูลทั้งสอง เราได้เลือกโมเดลที่เหมาะสมสำหรับการทำนาย ดังนี้:
                    <ul>
                        <li><span class="highlight">Tesla Stock Data</span>:<br>
                            - <strong>SVM Linear Model</strong>: เราเลือก Support Vector Machine แบบ Linear เพราะเหมาะกับข้อมูลที่อาจมีแนวโน้มแยกได้ด้วยเส้นตรง ใช้ฟีเจอร์ เช่น Year, Month, Day, Open, High, Low, Volume เพื่อทำนายราคา Close โมเดลนี้ถูกฝึกด้วย Scikit-learn และบันทึกเป็นไฟล์ <code>svm_linear_model.pkl</code><br>
                            - <strong>SVR Model</strong>: เพื่อเพิ่มความยืดหยุ่นในการทำนายข้อมูลต่อเนื่อง เราใช้ Support Vector Regression (SVR) ซึ่งสามารถจัดการกับความสัมพันธ์ที่ซับซ้อนขึ้นได้ โมเดลนี้ถูกฝึกและบันทึกเป็น <code>svr_model.pkl</code>
                        </li>
                        <li><span class="highlight">Placement Prediction Dataset</span>:<br>
                            - <strong>Neural Network (PlacementNN)</strong>: เราเลือกใช้โครงข่ายประสาทเทียมด้วย PyTorch เพราะข้อมูลนี้มีความซับซ้อนและมีฟีเจอร์หลายมิติ (เช่น CGPA, Internships, Projects) โมเดลประกอบด้วยเลเยอร์ 128, 64, 32 และ 1 พร้อม Dropout เพื่อป้องกัน Overfitting หลังจากฝึกด้วยข้อมูลที่ปรับสเกลแล้ว โมเดลถูกบันทึกเป็น <code>placement_model.pkl</code> พร้อม Scaler (<code>scaler.pkl</code>) และ LabelEncoder (<code>label_encoder.pkl</code>)
                        </li>
                    </ul>
                </div>
                <div class="section-title">2. การฝึกโมเดล</div>
                <div class="section-desc">
                    การฝึกโมเดลทั้งหมดทำใน Jupyter Notebook โดยมีการปรับแต่ง Hyperparameter เช่น:<br>
                    - <strong>Learning Rate</strong> และ <strong>Kernel Type</strong> สำหรับ SVM และ SVR<br>
                    - <strong>จำนวน Epochs</strong> และ <strong>Dropout Rate</strong> สำหรับ Neural Network<br>
                    ผลลัพธ์ถูกประเมินด้วยเมตริก เช่น <strong>Mean Squared Error (MSE)</strong> สำหรับ Tesla Stock Data และ <strong>Accuracy</strong> สำหรับ Placement Prediction Dataset
                </div>
                <div class="section-title">3. การบันทึกและนำไปใช้งาน</div>
                <div class="section-desc">
                    หลังจากฝึกเสร็จสิ้น โมเดลถูกบันทึกในรูปแบบไฟล์ที่สามารถโหลดได้ง่าย:<br>
                    - SVM และ SVR ใช้ <code>pickle</code> สำหรับบันทึกเป็น <code>.pkl</code><br>
                    - Neural Network ใช้ <code>torch.save</code> เพื่อบันทึกสถานะของโมเดลเป็น <code>placement_model.pkl</code><br>
                    ไฟล์เหล่านี้ถูกนำไปใช้ใน Streamlit โดยโหลดผ่าน <code>pickle.load()</code> หรือ <code>torch.load()</code> เพื่อทำนายผลลัพธ์ตามข้อมูลที่ผู้ใช้กรอก
                </div>
                <div class="section-title">4. การพัฒนาเว็บแอปพลิเคชันด้วย Streamlit</div>
                <div class="section-desc">
                    เมื่อโมเดลพร้อมใช้งาน เราได้พัฒนาเว็บแอปด้วย Streamlit ซึ่งมีโครงสร้างหน้าเว็บดังนี้:<br>
                    - <strong>หน้า "Data & Development"</strong>: แสดงแหล่งที่มาและรายละเอียด Dataset โดยมีแท็บ "Main", "Dataset Info" (พร้อมปุ่มสลับ Tesla และ Placement), "Development Model", และ "Evaluation"<br>
                    - <strong>หน้า "Machine Learning Models"</strong>: นำเสนอ SVM Linear และ SVR พร้อมช่องกรอกข้อมูลและปุ่มทำนาย<br>
                    - <strong>หน้า "Neural Network Model"</strong>: นำเสนอ PlacementNN พร้อมช่องกรอกข้อมูล 10 ฟีเจอร์และปุ่มทำนาย<br>
                    การออกแบบ UI ใช้ CSS เพื่อปรับแต่งปุ่มและแท็บให้มีสไตล์ Minimalist และจัดวางกึ่งกลาง
                </div>
                <div class="section-title">5. การทดสอบและปรับปรุง</div>
                <div class="section-desc">
                    หลังพัฒนาเสร็จ เราได้ทดสอบเว็บแอปโดยรันด้วยคำสั่ง <code>streamlit run main.py</code> และตรวจสอบว่า:<br>
                    - ปุ่มใน "Dataset Info" สลับ Dataset ได้ถูกต้อง<br>
                    - การกรอกข้อมูลและทำนายใน "Machine Learning Models" และ "Neural Network Model" ให้ผลลัพธ์ที่สอดคล้อง<br>
                    - การแสดงตารางข้อมูลจาก <code>TSLA.csv</code> และ <code>placementdata_miss.csv</code> ทำงานได้<br>
                    พบปัญหา เช่น ข้อมูลขาดหายใน Tesla Stock หรือการกรอกข้อมูลเกินขอบเขตใน Placement Prediction เราแก้ไขโดยเพิ่มคำเตือนและกำหนดขอบเขตในช่องกรอกข้อมูล (เช่น <code>min_value</code>, <code>max_value</code>)
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)



# ฟังก์ชันสำหรับหน้า Page 1 (ใช้ Tabs)
def page1():
    st.title("Data & Development 🔥")
    
    tab1, tab2, tab3= st.tabs([
        "Main",
        "Dataset Info",
        "Development Model",
 
    ])
    
    with tab1:
        subpage_main()
    with tab2:
        subpage_dataset()
    with tab3:
        subpage_dl()


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
            with open('models/ML/svm_linear_model.pkl', 'rb') as file:
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
            with open('models/ML/svr_model.pkl', 'rb') as file:
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
            label_encoder = joblib.load('models/NN/label_encoder.pkl')
            # โหลด Scaler
            scaler = joblib.load('models/NN/scaler.pkl')
            # โหลดโมเดล PyTorch
            model = PlacementNN()
            model.load_state_dict(torch.load('models/NN/placement_model.pkl', weights_only=False))
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
            st.success(f"ผลการทำนาย: **{result}**")
        except FileNotFoundError as e:
            st.error(f"ไม่พบไฟล์ที่ต้องการ: {str(e)}. กรุณาตรวจสอบว่าไฟล์ 'placement_model.pkl', 'scaler.pkl', และ 'label_encoder.pkl' อยู่ในโฟลเดอร์ที่ถูกต้อง")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")

# CSS เพื่อปรับแต่งทั้งปุ่มและ tabs ให้ minimal และกึ่งกลาง
st.markdown("""
    <style>
    .nav-container {
        # background-color: #f9f9f9;
        # padding: 20px;
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
st.markdown("<h1 style='text-align: center; color: #ffff;'>Welcome to Placement Prediction App</h1>", unsafe_allow_html=True)

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