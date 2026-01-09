# Decision Tree Machine Learning Project

## 1. Project Title

**Customer Purchase Prediction Using Decision Tree Algorithm**

---

## 2. Problem Statement

Businesses want to predict whether a customer will purchase a product based on demographic and behavioral attributes. The objective of this project is to build a **Decision Tree Classification model** that predicts customer purchase decisions accurately and interpretably.

---

## 3. Objectives

* Understand and implement the Decision Tree algorithm
* Perform data preprocessing and exploratory analysis
* Manually compute split criteria for understanding
* Build a Decision Tree model using a library
* Evaluate the model using appropriate metrics
* Interpret results and model behavior

---

## 4. Dataset Description

**Dataset:** Social Network Ads (commonly used for classification)

| Feature         | Description                       |
| --------------- | --------------------------------- |
| Age             | Age of the customer               |
| EstimatedSalary | Estimated annual salary           |
| Purchased       | Target variable (0 = No, 1 = Yes) |

---

## 5. Tools and Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 6. Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Manual Decision Tree Split Calculation
5. Model Training (Decision Tree)
6. Model Evaluation
7. Result Interpretation

---

## 7. Data Preprocessing

* Handle missing values (if any)
* Feature selection
* Train-test split
* Feature scaling (optional for Decision Trees)

---

## 8. Manual Calculation Example (Entropy & Information Gain)

### Sample Data

| Purchased | Count |
| --------- | ----- |
| Yes (1)   | 9     |
| No (0)    | 5     |

### Entropy

H(S) = -[(9/14)log2(9/14) + (5/14)log2(5/14)]

### Information Gain

IG(S, Age) = H(S) - Weighted Entropy after split

This manual step validates how the algorithm chooses optimal splits.

---

## 9. Model Implementation (Decision Tree Classifier)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# Load dataset
data = pd.read_csv('Social_Network_Ads.csv')

X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 10. Model Evaluation Metrics

### Accuracy

Measures overall correctness of predictions.

### Confusion Matrix

Provides insight into False Positives and False Negatives.

### Precision, Recall, F1-score

Used due to possible class imbalance and business risk sensitivity.

```python
print('Accuracy:', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 11. Why These Metrics Are Used

* **Accuracy:** Baseline performance indicator
* **Precision:** Important when false positives are costly
* **Recall:** Important when missing positive cases is risky
* **F1-score:** Balanced metric for uneven class distribution

---

## 12. Visualization (Optional)

* Decision Tree structure visualization
* Feature importance plot

---

## 13. Results and Discussion

* Decision Tree achieved interpretable rules
* Model performance depends heavily on depth
* Overfitting observed for deeper trees

---

## 14. Limitations

* Sensitive to noisy data
* High variance
* Requires pruning or depth control

---

## 15. Conclusion

The Decision Tree model provides a transparent and interpretable approach to customer purchase prediction. With proper depth control and evaluation metrics, it performs effectively on structured tabular data.

---

## 16. Future Enhancements

* Apply pruning techniques
* Compare with Random Forest
* Hyperparameter tuning
* Cross-validation

---

## 17. References

* Scikit-learn Documentation
* Machine Learning by Tom Mitchell
* Pattern Recognition and Machine Learning by Bishop
