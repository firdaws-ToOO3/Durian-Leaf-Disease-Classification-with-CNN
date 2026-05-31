# 🌿 Durian Leaf Disease Classification with CNN

> เปรียบเทียบ 5 Transfer Learning Models สำหรับการจำแนกโรคในใบทุเรียน พร้อมระบบ Checkpoint ที่ช่วยให้ฝึกโมเดลต่อเนื่องได้บน Google Colab

---

## 📋 สารบัญ

- [ภาพรวมโปรเจค](#-ภาพรวมโปรเจค)
- [คลาสที่จำแนก](#-คลาสที่จำแนก)
- [โมเดลที่เปรียบเทียบ](#-โมเดลที่เปรียบเทียบ)
- [ปัญหาและวิธีแก้ไข](#-ปัญหาและวิธีแก้ไข)
- [โครงสร้างโปรเจค](#-โครงสร้างโปรเจค)
- [วิธีใช้งาน](#-วิธีใช้งาน)
- [ผลลัพธ์](#-ผลลัพธ์)
- [เทคโนโลยีที่ใช้](#-เทคโนโลยีที่ใช้)

---

## 🎯 ภาพรวมโปรเจค

โรคในใบทุเรียนเป็นปัญหาสำคัญที่ส่งผลกระทบต่อผลผลิตและรายได้ของเกษตรกร การวินิจฉัยโรคด้วยตาเปล่าต้องอาศัยผู้เชี่ยวชาญซึ่งไม่ได้มีอยู่ทั่วไป

โปรเจคนี้พัฒนาระบบ **Image Classification** โดยใช้ Convolutional Neural Network (CNN) ด้วยเทคนิค **Transfer Learning** เพื่อจำแนกโรคในใบทุเรียนได้อัตโนมัติ โดยเปรียบเทียบประสิทธิภาพของ 5 สถาปัตยกรรมยอดนิยมและเลือกโมเดลที่ดีที่สุดสำหรับงานนี้

---

## 🍃 คลาสที่จำแนก

| คลาส | ชื่อโรค | ลักษณะ |
|------|---------|--------|
| `healthy` | ใบปกติ | ใบสีเขียวสม่ำเสมอ ไม่มีรอยโรค |
| `rust` | ราสนิม (Rust) | จุดสีน้ำตาล-เหลืองบนใบ เกิดจากเชื้อรา |
| `anthracnose` | แอนแทคโนส (Anthracnose) | จุดสีดำหรือน้ำตาลเข้ม ขอบชัดเจน |

---

## 🤖 โมเดลที่เปรียบเทียบ

โปรเจคนี้เปรียบเทียบ 5 สถาปัตยกรรม โดยใช้ weights ที่ pre-trained บน ImageNet:

| โมเดล | ขนาด | จุดเด่น |
|-------|------|---------|
| **VGG16** | ~528 MB | สถาปัตยกรรมดั้งเดิม เข้าใจง่าย |
| **MobileNetV2** | ~14 MB | น้ำหนักเบา เหมาะกับ mobile |
| **Xception** | ~88 MB | Depthwise separable convolutions |
| **NASNetMobile** | ~23 MB | Neural Architecture Search |
| **ResNet50V2** | ~98 MB | Residual connections ลด vanishing gradient |

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
# บันทึก checkpoint หลังทุก epoch
checkpoint_callback = ModelCheckpoint(
    filepath='checkpoints/model_{epoch:02d}.h5',
    save_best_only=False,      # บันทึกทุก epoch ไม่ใช่แค่ best
    save_weights_only=False,
    verbose=1
)

# บันทึก training log
csv_logger = CSVLogger(
    'logs/training_log.csv',
    append=True                # append ต่อเมื่อ resume
)
```

```python
# ตรวจสอบและ resume จาก checkpoint ล่าสุด
def resume_training(model_name):
    checkpoint_dir = f'checkpoints/{model_name}/'
    log_file = f'logs/{model_name}_log.csv'
    
    # หา checkpoint ล่าสุด
    checkpoints = sorted(glob.glob(checkpoint_dir + '*.h5'))
    
    if checkpoints:
        latest = checkpoints[-1]
        last_epoch = int(re.search(r'epoch_(\d+)', latest).group(1))
        model = load_model(latest)
        print(f"✅ Resuming from epoch {last_epoch + 1}")
        return model, last_epoch
    else:
        print("🆕 Starting fresh training")
        return build_model(model_name), 0
```

**ผลลัพธ์:** ไม่ว่าจะหลุดการเชื่อมต่อกี่ครั้ง ระบบจะ resume จาก epoch ล่าสุดอัตโนมัติ ไม่ต้องเริ่มใหม่ตั้งแต่ต้น

---

## 📁 โครงสร้างโปรเจค

```
durian-disease-cnn/
├── 📓 notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── 📊 dataset/
│   ├── train/
│   │   ├── healthy/
│   │   ├── rust/
│   │   └── anthracnose/
│   ├── val/
│   └── test/
├── 💾 checkpoints/          # Model checkpoints (ไม่ถูก track ใน git)
├── 📈 logs/                 # Training logs CSV
├── 📉 results/
│   ├── accuracy_comparison.png
│   ├── confusion_matrices/
│   └── training_curves/
└── 📄 README.md
```

---

## 🚀 วิธีใช้งาน

### 1. Clone โปรเจค

```bash
git clone https://github.com/yourusername/durian-disease-cnn.git
```

### 2. เปิดใน Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/durian-disease-cnn/blob/main/notebooks/02_model_training.ipynb)

### 3. ติดตั้ง Dependencies

```python
pip install tensorflow==2.x opencv-python matplotlib scikit-learn
```

### 4. เทรนโมเดล (พร้อม Auto-Resume)

```python
# เทรนโมเดลทั้งหมด 5 ตัว
models = ['VGG16', 'MobileNetV2', 'Xception', 'NASNetMobile', 'ResNet50V2']

for model_name in models:
    model, initial_epoch = resume_training(model_name)
    train_model(model, model_name, initial_epoch=initial_epoch)
```

### 5. เปรียบเทียบผลลัพธ์

```python
# โหลด training logs และเปรียบเทียบ
compare_all_models(logs_dir='logs/')
```

---

## 📊 ผลลัพธ์

### Accuracy เปรียบเทียบ

| โมเดล | Train Acc | Val Acc | Test Acc | Parameters |
|-------|-----------|---------|----------|------------|
| VGG16 | -% | -% | -% | 138M |
| MobileNetV2 | -% | -% | -% | 3.4M |
| Xception | -% | -% | -% | 22.9M |
| NASNetMobile | -% | -% | -% | 5.3M |
| **ResNet50V2** | **-**% | **-**% | **-**% | 25.6M |

> 📝 อัปเดตค่า accuracy จริงหลังเทรนเสร็จ

### สิ่งที่ค้นพบ

- **โมเดลที่ดีที่สุด**: [ชื่อโมเดล] ด้วย accuracy [X]% บน test set
- **โมเดลที่เบาที่สุด**: MobileNetV2 เหมาะสำหรับ deployment บนอุปกรณ์ขนาดเล็ก
- ปัญหา class imbalance ส่งผลต่อการจำแนกคลาส [X] — แก้โดยใช้ class weights

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

## 👤 ผู้พัฒนา

**[ชื่อของคุณ]**  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

*โปรเจคนี้พัฒนาเพื่อวัตถุประสงค์ทางการศึกษา*
