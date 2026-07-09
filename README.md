# Medical Data Analysis and Disease Prediction

This repository contains a comprehensive data analysis and machine learning project focused on analyzing patient medical features and predicting disease status. The project utilizes popular Python data science libraries to explore relationships between clinical metrics and trains classification models (Logistic Regression and Support Vector Machines) to classify health outcomes.

## 🚀 Features
* **Exploratory Data Analysis (EDA):** Detailed visualizations using `matplotlib` and `seaborn` to understand feature distributions (Age, BMI, Blood Pressure, Glucose, etc.) and their impact on different diseases.
* **Correlation Analysis:** Identification of linear relationships among numeric medical attributes.
* **Predictive Modeling:** Implementation and evaluation of **Logistic Regression** and **Support Vector Classifiers (SVC)** for disease prediction.
* **Performance Evaluation:** Training vs. Testing accuracy metrics.

---

## 📊 Visualizations Included
The project walks through several statistical plots to uncover hidden patterns:
1. **Scatter Plot:** Age vs. Glucose Level categorized by Disease Type.
2. **Count Plot:** Disease distribution broken down by Smoking habits.
3. **Heatmap:** Correlation matrix highlighting relationships between numeric features.
4. **Violin Plot:** BMI distribution grouped by Gender and Disease status.
5. **Pair Plot:** Pairwise relationships between key clinical markers (`Age`, `BMI`, `BloodPressure`, `GlucoseLevel`).
6. **Box Plots:** Distribution comparison across diseases for `GlucoseLevel`, `BMI`, `Age`, `BloodPressure`, and `HeartRate`.
7. **Bar Chart:** Aggregated mean values of numeric medical features grouped by disease category.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`

---

## 📂 Project Workflow

### 1. Data Exploration & Visualization
Before feeding data into models, the data is analyzed using multi-variable plots to understand the boundary limits and spread of parameters like BMI, Heart Rate, and Blood Pressure relative to a patient's diagnosed condition.

### 2. Model Training & Evaluation
Two predictive models are trained on the dataset using a standard train-test split:

#### **Logistic Regression (LR)**
```python
# Check Accuracy
print("Accuracy of test data : ", lr.score(x_test, y_test) * 100)
print("Accuracy of train data : ", lr.score(x_train, y_train) * 100)

# Predictions
pred = lr.predict(x_test)