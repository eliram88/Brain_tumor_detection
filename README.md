# Brain Tumor Detection using EfficientNetB0 and Transfer Learning 

🎯 Project Goal: Building an optimized and accurate model for brain tumor detection in MRI scans using the pre-trained EfficientNetB0 model and advanced deep learning techniques.


## 🌐 Links

- [Dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)  
  Brain Tumor MRI Dataset (Healthy & Tumor images)

- [View project in Google Colab](https://colab.research.google.com/drive/1yTaL8_Fqk3TbfHazISD6DFJFajpOq_mV?usp=sharing)  
  Google Colab Notebook: Run the project online in Google Colab

- [View project در GitHub](https://github.com/eliram88/Brain_tumor_detection)  
  GitHub Repository: Source code and project documentation



## 🔧 Tools & Libraries

- Python (TensorFlow, Keras, NumPy, Matplotlib, Seaborn, scikit-learn)  
- Google Colab
- Google Drive
- Kaggle Dataset
- GitHub for version control



## 📊 Dataset

- **Source:** Brain Tumor MRI Dataset from Kaggle  
- **Samples:**  
  - Train: 3152 images  
  - Validation: 682 images  
  - Test: 680 images  
- **Classes:**  
  - `0` →  brain tumor  
  - `1` → healthy




## 📊 Project stages


### 🛠 Preprocessing 

- Structured dataset into train/val/test directories
- Loaded images with image_dataset_from_directory at size (224, 224)
- Applied Prefetching for faster loading
- Data Augmentation:
  - Random Flip  
  - Random Rotation (0.1)  
  - Random Zoom (0.1) 
- Displayed sample images with labels


### 🧠 Model Design

- Base Model: EfficientNetB0 (ImageNet weights, top layers removed)
- Layers:
  - Data Augmentation  
  - Preprocess Input مخصوص EfficientNet  
  - EfficientNetB0 (frozen initially)  
  - Global Average Pooling  
  - Dropout (0.3)  
  - Dense (28 units, ReLU)  
  - Dropout (0.2)  
  - Dense (1, activation="sigmoid")


### ⚙ Training 
 
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


### 📈 Results

- **Test Accuracy:** 0.9074  
- **Test Loss:** 0.1990
  
- Confusion Matrix & Classification Report show strong performance in distinguishing tumor vs. healthy cases.
- The model achieves high accuracy with minimal errors.



## 🚀 How to Run

1) Install dependencies 
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

2) Run Jupyter Notebook  
```bash
jupyter notebook
```
Open the file Brain_tumor_detection.ipynb and run all cells.



## 📁 Project Structure

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



## 🧑‍💻 Developer

This project was developed by a deep learning and computer vision enthusiast with the goal of:

  - Advancing skills in medical image analysis
  - Building a professional portfolio project
  - Applying state-of-the-art deep learning models for healthcare

