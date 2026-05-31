# 🌿 Durian Leaf Disease Classification with CNN

> พร้อมระบบ Checkpoint ที่ช่วยให้ฝึกโมเดลต่อเนื่องได้บน Google Colab

---

## 🎯 ภาพรวมโปรเจค

โรคในใบทุเรียนเป็นปัญหาสำคัญที่ส่งผลกระทบต่อผลผลิตและรายได้ของเกษตรกร การวินิจฉัยโรคด้วยตาเปล่าต้องอาศัยผู้เชี่ยวชาญซึ่งไม่ได้มีอยู่ทั่วไป

โปรเจคนี้พัฒนาระบบ **Image Classification** โดยใช้ Convolutional Neural Network (CNN) ด้วยเทคนิค **Transfer Learning** เพื่อจำแนกโรคในใบทุเรียนได้อัตโนมัติ

---

## 🍃 คลาสที่จำแนก

| คลาส | ชื่อโรค | ลักษณะ |
|------|---------|--------|
| `healthy` | ใบปกติ | ใบสีเขียวสม่ำเสมอ ไม่มีรอยโรค |
| `rust` | ราสนิม (Rust) | จุดสีน้ำตาล-เหลืองบนใบ เกิดจากเชื้อรา |
| `anthracnose` | แอนแทคโนส (Anthracnose) | จุดสีดำหรือน้ำตาลเข้ม ขอบชัดเจน |

---

## 🤖 โมเดลที่ใช้

**MobileNetV2** ขนาด ~14 MB โดยมีจุดเด่นคือ น้ำหนักเบา เหมาะกับ mobile

---

## ⚙️ ปัญหาและวิธีแก้ไข

### 🔴 ปัญหา: Google Colab Disconnection

การฝึกโมเดลบน **Google Colab (Free Tier)** มีข้อจำกัดคือ:
- GPU session หมดเวลาโดยไม่แจ้งล่วงหน้า
- การเชื่อมต่อหลุดระหว่างการเทรนที่ใช้เวลานาน
- ต้องเริ่มการเทรนใหม่ตั้งแต่ Epoch 0 ทุกครั้ง ทำให้เสียเวลามาก

### 🟢 วิธีแก้: Epoch-Level Checkpoint System

ออกแบบระบบ **Model Checkpoint** ที่บันทึกสถานะการเทรนทุก epoch ดังนี้:

```python
# กำหนด callback สำหรับการบันทึกโมเดลและ log
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,  # บันทึกทั้งโมเดล
    save_best_only=False,     # บันทึกทุก epoch
    verbose=1
)

csv_logger = CSVLogger(log_path, append=True)  # บันทึก log ของการฝึก
```


**ผลลัพธ์:** ไม่ว่าจะหลุดการเชื่อมต่อกี่ครั้ง ระบบจะ resume จาก epoch ล่าสุดอัตโนมัติ ไม่ต้องเริ่มใหม่ตั้งแต่ต้น


## 📊 ผลลัพธ์

### Accuracy เปรียบเทียบ

| โมเดล | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| MobileNetV2 | 1.0000 | 0.9885 | 0.0297 |

> อัปเดตค่า accuracy จริงหลังเทรนเสร็จ

---

## 🛠 เทคโนโลยีที่ใช้

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?logo=keras)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

- **Framework**: TensorFlow / Keras
- **Computer Vision**: OpenCV, PIL
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Google Colab (Free Tier)
- **Version Control**: Git / GitHub

---

*โปรเจคนี้พัฒนาเพื่อวัตถุประสงค์ทางการศึกษา*
