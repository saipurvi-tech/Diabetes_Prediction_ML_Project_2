

A machine learning model built with Python and Scikit-Learn that predicts whether a female patient is diabetic based on diagnostic medical measurements.

---

## 📌 Project Overview

This project uses a Support Vector Classifier (`SVC`) with a linear kernel to perform binary classification on the Pima Indians Diabetes Dataset. It standardizes diagnostic metrics, trains a supervised learning model, evaluates performance, and provides a pipeline to test custom single-instance patient data.

---

## 📂 Features & Dataset Schema

The model takes 8 clinical input parameters to predict the target outcome:

| Feature Name | Description |
| :--- | :--- |
| **Pregnancies** | Number of times pregnant |
| **Glucose** | Plasma glucose concentration (2 hours in an oral glucose tolerance test) |
| **BloodPressure** | Diastolic blood pressure ($mm\ Hg$) |
| **SkinThickness** | Triceps skin fold thickness ($mm$) |
| **Insulin** | 2-Hour serum insulin ($\mu U/ml$) |
| **BMI** | Body mass index ($weight\ in\ kg / (height\ in\ m)^2$) |
| **DiabetesPedigreeFunction** | Diabetes pedigree function (genetic score) |
| **Age** | Age in years |
| **Outcome** *(Target)* | `0`: Non-diabetic, `1`: Diabetic |

---

## ⚙️ Prerequisites & Dependencies

Ensure you have Python 3.8+ installed. Install the required libraries:

```bash
pip install numpy pandas scikit-learn
