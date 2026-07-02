# Heart Disease Dataset

## 1. Project Overview
This project focuses on building and evaluating six machine learning models to predict the presence of heart disease based on 13 clinical health indicators. Early diagnosis of cardiovascular issues is critical, and these predictive pipelines serve as a reliable decision-support tool for healthcare workers.

This notebook presents a complete data science workflow featuring data cleaning (duplicate removal to prevent overfitting), standardization via `StandardScaler`, pipeline-driven model training, and performance evaluation emphasizing clinical safety metrics like **Recall**.

## 2. Dataset Overview & Features
The data is based on the famous **UCI Cleveland Heart Disease Dataset**. While the original file contained **1,025 rows**, rigorous cleaning identified **723 duplicate records** created via resampling. These duplicates were dropped to eliminate data leakage and ensure an honest evaluation on unseen data, leaving a clean dataset of **302 unique patients**.

### Feature Attributes:
1. age: Patient age in years.
2. sex: Biological sex (1 = Male, 0 = Female).
3. cp: Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic).
4. trestbps: Resting blood pressure (mm Hg).
5. chol: Serum cholesterol level (mg/dl).
6. fbs: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False).
7. restecg: Resting ECG results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy).
8. thalach: Maximum heart rate achieved.
9. exang: Exercise-induced angina (1 = Yes, 0 = No).
10. oldpeak: ST depression induced by exercise relative to rest.
11. slope: Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping).
12. ca: Number of major vessels (0–3) colored by fluoroscopy.
13. thal: Thalassemia status (1: Fixed defect, 2: Normal, 3: Reversible defect).
14. target: The diagnosis of heart disease (The classification target):
    * **0 = No Disease (Normal / Healthy)**
    * **1 = Heart Disease (Cardiovascular disease present)**


**Machine Learning Models**

Six machine learning models were trained and evaluated in this project to compare their performance on the heart disease classification task.

Logistic Regression is a simple and interpretable linear model that estimates the probability of a patient having heart disease. It serves as a strong baseline model for binary classification tasks.

Support Vector Machine (SVM) finds the optimal hyperplane that best separates patients with and without heart disease in the feature space. It works well even with a relatively small number of training samples, which suited this dataset after duplicate removal.

K-Nearest Neighbors (KNN) classifies a patient based on the majority class of their K nearest neighbours in the feature space. In this project, K was set to 5, meaning the 5 most similar patients in the training data were used to make each prediction.

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It assumes that all features are independent of each other, which makes it fast and surprisingly effective, even on a dataset with mixed categorical and numerical health indicators like this one.

Random Forest is an ensemble method that builds multiple decision trees and combines their predictions to produce a more accurate and stable result. In this project, 100 trees were used with a random state of 42 to ensure reproducibility.

Decision Tree is a simple tree-based model that splits the data based on feature values to make predictions. It is the most interpretable model among the six, as the decision path can be visualised directly. A maximum depth of 5 was set to prevent overfitting.


**Data Cleaning and Train/Test Split**

An important step in this project was identifying that the original dataset of 1025 rows contained 723 duplicate records, leaving only 302 unique patient entries. These duplicates were removed before splitting the data, since training and testing on repeated rows would allow a model to "memorise" patients rather than genuinely learn to generalise, artificially inflating accuracy scores.

After cleaning, the dataset was divided into training and testing sets using an 80/20 split, resulting in 241 training samples and 61 testing samples. A fixed random state of 42 was used to ensure reproducibility of results across all models.

Before training, all features were standardised using StandardScaler to ensure that no single feature dominated the model due to differences in scale. The scaler was fitted only on the training data and then applied to the test data, preventing data leakage and ensuring a fair evaluation of model performance on unseen data.
All six models were trained on the same training set and evaluated on the same test set, making the performance comparison fair and consistent across all models.



### ROC Curve and Area Under the Curve (AUC) Analysis

To evaluate how effectively each machine learning model distinguishes between patients with heart disease and those who are healthy, a Receiver Operating Characteristic (ROC) curve analysis was conducted. The performance was quantified using the Area Under the Curve (AUC) score, where a score closer to 1.0 indicates near-perfect classification capability.
Random Forest achieved the highest AUC score at 0.8788, followed closely by Support Vector Machine (SVM) at 0.8696 and Naive Bayes at 0.8556. These scores indicate roughly an 85–88% probability that the models will correctly rank a randomly chosen heart disease patient higher than a healthy individual, reflecting strong discriminative ability given the limited size of the cleaned dataset. K-Nearest Neighbors (0.8389) and Logistic Regression (0.8341) performed similarly to one another in the mid-range, while the Decision Tree recorded the lowest AUC at 0.7161, indicating a noticeably weaker ability to separate the two classes compared to the other models.

### Feature Importance Analysis

A permutation feature importance analysis was conducted to understand which clinical risk factors contributed most significantly to the final classification decisions across the models. This analysis provides essential clinical transparency, transforming the machine learning pipelines from complex "black boxes" into interpretable decision-support tools for healthcare professionals.

Across the highest-performing models, **chest pain type (cp)** emerged consistently as the most dominant predictive indicator for heart disease. This aligns perfectly with real-world clinical knowledge, as specific types of chest discomfort (such as typical or atypical angina) are major presenting symptoms of cardiovascular distress. Other critical features that heavily influenced model decisions included **maximum heart rate achieved (thalach)**, **the number of major vessels colored by fluoroscopy (ca)**, and **ST depression induced by exercise relative to rest (oldpeak)**. Interestingly, traditional standalone risk factors such as fasting blood sugar (fbs) and resting blood pressure (trestbps) carried relatively lower weight compared to dynamic exercise-induced cardiac indicators. This insight highlights the power of machine learning to look beyond isolated baseline measurements and prioritize complex, interconnected physiological markers when assessing a patient's overall cardiac risk profile.


**Results Summary**

This project evaluated six machine learning models on the UCI Heart Disease Dataset to classify patients as having heart disease or not, based on 13 clinical features. Among all models, Naive Bayes achieved the highest overall accuracy at 85.25%, with an F1-Score of 84.75%. Random Forest followed closely with 83.61% accuracy but achieved the highest ROC AUC (0.8788) and the highest recall alongside Logistic Regression, with both models missing only 3 out of 29 actual heart disease cases in the test set. Support Vector Machine achieved 78.69% accuracy, while Logistic Regression, despite tying for the best recall (89.66%), had noticeably lower precision (70.27%), meaning it flagged more healthy patients as at-risk than the other top models. K-Nearest Neighbors and Decision Tree were the weakest performers, at 73.77% accuracy each, with Decision Tree also recording the most missed disease cases (9 false negatives).

It is important to note that in healthcare applications, Recall on the positive (disease) class is often more critical than overall Accuracy, since a false negative — a patient with heart disease incorrectly classified as healthy — could delay necessary treatment. While both Random Forest and Logistic Regression tied for the fewest missed cases (3 false negatives, 89.66% recall), Random Forest is considered the more suitable model overall, as it paired this strong recall with substantially higher precision (78.79% vs. 70.27%), F1-Score (83.87% vs. 78.79%), and ROC AUC (0.8788 vs. 0.8341) — meaning it catches nearly as many true cases while producing far fewer false alarms.
It should also be noted that after removing 723 duplicate records, the dataset was reduced to 302 unique patients, which is relatively small for training machine learning models. This likely explains why accuracy scores in this project (74–85%) are noticeably lower than scores often reported on the unfiltered 1025-row version of this dataset, where duplicate patient records overlapping between the training and test sets artificially inflated performance to over 95%. The results reported here are considered more reliable, as they reflect genuine model performance on unseen data.