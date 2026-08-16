# Diabetes_Prediction_ML_Project_2

End-to-end supervised learning project that predicts whether a patient has diabetes using the Pima Indians Diabetes Dataset. This repo is a work-in-progress

## 🎯 Goal

Build a classifier that takes medical features (pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age) and predicts the `Outcome` (0 = non-diabetic, 1 = diabetic).

## 📦 Dataset

- **Source:** Pima Indians Diabetes Dataset [Kaggle]
- **Samples:** 768
- **Features:** 8 numeric medical attributes
- **Target:** `Outcome` (binary: 0/1)

## 🧠 Approach (so far)

- **Model:** Support Vector Machine (SVM) classifier with linear kernel 
- **Preprocessing:** Standardization using `StandardScaler`
- **Split:** 80% train / 20% test with stratification on `Outcome`
- **Metrics:** Accuracy on train and test sets

### Current Results

- Training accuracy: ~0.78–0.79
- Test accuracy: ~0.76–0.78  
*(Update with your exact numbers once you finalize the run.)*



