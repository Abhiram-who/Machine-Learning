### Welcome to a collection of beginner-friendly Machine Learning projects aimed at solving real-world problems with predictive modeling.

This section contains two complete projects:

- **Heart Disease Prediction**

- **Spam Email Detection**

#
# 📁 Projects Overview



### 1. **❤️ Heart Disease Detection**

**Objective:**

Predict whether a patient has heart disease based on clinical parameters.

**Dataset:**

Structured data containing patient attributes like age, cholesterol, blood pressure, etc.

**Methodology:**

- Data Cleaning: Handle missing values, normalize features.

- Exploratory Data Analysis (EDA):

  - Correlation matrix to identify important features.

  - Visualization of feature distributions (e.g., cholesterol levels, age distribution).

- Feature Selection:

  - Removal of less impactful variables.

  - Use of domain knowledge and correlation scores.

- Model Training:

  - Algorithms used: Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees.

  - Model comparison using accuracy, precision, recall, and F1-score.

- Evaluation:

  - Confusion Matrix

  - ROC-AUC Curve

**Results:**

- Achieved an accuracy score of around 85-90% (depending on model).

- Logistic Regression performed consistently well across different validation sets.

![](Data\pics\output1.png)
![](Data\pics\output2.png)
##
### 2. 📧 Spam Email Detection



**Objective:**

Classify emails as either spam or not spam using Natural Language Processing (NLP) and Machine Learning.

**Dataset:**

Collection of labeled email messages (Spam vs Ham).

**Methodology:**

- Text Preprocessing:

  - Lowercasing, removal of punctuation, stopwords, and stemming.

- Feature Extraction:

  - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

- Model Training:

  - Algorithms used: Naive Bayes, Support Vector Machines (SVM).

  - Hyperparameter tuning for optimal performance.

- Evaluation:

  - Confusion Matrix

  - Precision-Recall tradeoff

  - ROC Curve

**Results:**

- Achieved an accuracy score of 97-98% with Naive Bayes.

- TF-IDF improved the classification compared to simple bag-of-words approaches.
##

# 🛠️ Technologies Used
**Languages:** Python 3

**Libraries:**

- Scikit-learn

- Pandas

- NumPy

- Matplotlib

- Seaborn

- NLTK (for text processing)


##

# 📊 Results Summary


| Project                 | Best Model          | Accuracy | Highlights                                |
|--------------------------|---------------------|----------|-------------------------------------------|
| Heart Disease Detection  | Logistic Regression | ~88%     | ROC Curve AUC > 0.85                      |
| Spam Email Detection     | Naive Bayes         | ~98%     | High precision on spam classification     |


