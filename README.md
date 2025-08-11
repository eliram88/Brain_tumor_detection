# Men-Women Detection using Transfer Learning & Data Augmentation

🎯 هدف پروژه: ایجاد مدل بهینه و دقیق برای تشخیص تومور مغزی در تصاویر MRI با استفاده از مدل EfficientNetB0 از پیش آموزش‌دیده و دیگر تکنیک‌های deep learning



## 🌐 لینک ها

- [دیتاست پروژه](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)  
  دیتاست MRI تومور مغزی و تصاویر سالم

- [مشاهده پروژه در Google Colab](https://colab.research.google.com/drive/1yTaL8_Fqk3TbfHazISD6DFJFajpOq_mV?usp=sharing)  
  اجرای آنلاین کد پروژه در محیط Google Colab

- [مشاهده پروژه در GitHub](https://github.com/eliram88/Brain_tumor_detection)
   سورس کد و مستندات پروژه در GitHub 



## 🔧 ابزارهای استفاده‌شده

- Python (TensorFlow, Keras, NumPy, Matplotlib, Seaborn, scikit-learn)  
- Google Colab
- Google Drive
- Kaggle Dataset
- GitHub for version control



## 📊  دیتاست

- **Source:** Brain Tumor MRI Dataset from Kaggle  
- **Samples:**  
  - Train: 3152 images  
  - Validation: 682 images  
  - Test: 680 images  
- **Classes:**  
  - `0` →  brain tumor  
  - `1` → healthy



## 📊 مراحل پروژه


### 🛠 Preprocessing | پیش‌پردازش

- ایجاد ساختار پوشه برای تقسیم‌بندی داده به train/val/test  
- بارگذاری داده‌ها با `image_dataset_from_directory` و اندازه تصویر `(224, 224)`  
- اعمال Prefetch برای افزایش سرعت خواندن داده‌ها  
- Data Augmentation شامل:  
  - Random Flip  
  - Random Rotation (0.1)  
  - Random Zoom (0.1)  
- نمایش نمونه تصاویر همراه با برچسب


### 🧠 Model Design | طراحی مدل

- استفاده از مدل EfficientNetB0 (وزن‌های ImageNet، بدون لایه‌های بالایی)  
- لایه‌های مدل:  
  - Data Augmentation  
  - Preprocess Input مخصوص EfficientNet  
  - EfficientNetB0 (غیرقابل آموزش در ابتدا)  
  - Global Average Pooling  
  - Dropout (0.3)  
  - Dense (128 واحد، relu)  
  - Dropout (0.2)  
  - Dense (1, activation="sigmoid")


### ⚙ Training | آموزش
 
**Phase 1 — Transfer Learning**  
- Base model frozen  
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Metrics: Accuracy  
- Epochs: 30  
- Callback: ModelCheckpoint & EarlyStopping

**Phase 2 — Fine-Tuning**  
- Unfreeze last layers of EfficientNetB0
- Reduce learning rate → `1e-5`  
- Epochs: 5 
- Callback: ModelCheckpoint & EarlyStopping


### 📈 Results | نتایج

- **Test Accuracy:** 0.9074  
- **Test Loss:** 0.1990  
- ماتریس درهم‌ریختگی (Confusion Matrix) و گزارش طبقه‌بندی (Classification Report) نشان‌دهنده عملکرد دقیق مدل در تشخیص تومور و سالم بودن است.  
- مدل به خوبی توانسته با دقت بالا و کمترین خطا تصاویر را طبقه‌بندی کند.



## 🚀 نحوه اجرا

1) Install dependencies | نصب کتابخانه‌ها
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

2) Run Jupyter Notebook | اجرای نوت‌بوک
```bash
jupyter notebook
```
Open the file Men_women_detection.ipynb and run all cells.



## 📁 ساختار فایل‌ها

```bash
Brain_tumor_detection/
│
├── 📁 datasets/                         
│   ├── train/
│   ├── val/
│   └── test/
│
├── 📁 notebook/
│   └── Brain_tumor_detection.ipynb       # Data analysis & model training
│
├── 📄 README.md                          # Project documentation
│
├── requirements.txt                      # Project Libraries
```



## 🧑‍💻 توسعه‌دهنده

این پروژه توسط یک علاقه‌مند به یادگیری عمیق و بینایی ماشین طراحی و اجرا شده،
با هدف ارتقا مهارت در تحلیل داده‌های پزشکی و پیاده‌سازی مدل‌های پیشرفته.

✨ هدف: تولید نمونه‌کار قابل ارائه و یادگیری تخصصی حوزه تشخیص تصاویر پزشکی
