 Supervised Learning with Python

This repository helps you understand Supervised Learning using Python with simple and practical examples.

We cover the two main types of supervised learning:

 Regression

 Classification

All examples use Python + Scikit-learn, making them easy for beginners.

 What is Supervised Learning?

Supervised Learning is a type of Machine Learning where:

We have input data (X)

We have known output labels (y)

The model learns a relationship between X → y

 Because the data is labeled, it is called supervised learning.

 Example 1: Linear Regression (Regression Problem)

Goal: Predict continuous values

🔹 Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

🔹 Step 2: Create Labeled Data
# Input data (X)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Output data (Y)
y = np.array([2, 4, 6, 8, 10])


 Here:

X = input

y = known output
 This makes it supervised learning

🔹 Step 3: Train the Model
model = LinearRegression()
model.fit(X, y)

🔹 Step 4: Make Predictions
prediction = model.predict([[6]])
print("Predicted value:", prediction)

🔹 Step 5: Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Example")
plt.show()


 This graph shows:

Actual data points

Best-fit regression line

 Example 2: Classification using Logistic Regression

Goal: Predict categories (0/1, Yes/No)

🔹 Step 1: Import Libraries
import numpy as np
from sklearn.linear_model import LogisticRegression

🔹 Step 2: Create Labeled Data
# Hours studied
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Pass (1) or Fail (0)
y = np.array([0, 0, 0, 1, 1, 1])

🔹 Step 3: Train the Model
model = LogisticRegression()
model.fit(X, y)

🔹 Step 4: Make Prediction
result = model.predict([[4]])
print("Prediction (1 = Pass, 0 = Fail):", result)

 Train–Test Split (Important Concept)

Used to test model performance on unseen data.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

 Model Evaluation
 For Regression (R² Score)
from sklearn.metrics import r2_score

y_pred = model.predict(X)
print("R² Score:", r2_score(y, y_pred))

 For Classification (Accuracy)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y, model.predict(X))
print("Accuracy:", accuracy)

 Why These Examples Matter

✔ Show complete supervised learning workflow
✔ Beginner & college-friendly
✔ Useful for projects, exams, and interviews
✔ Can be reused in Medium blogs & GitHub portfolios
✔ Explain training, prediction, and evaluation

 Conclusion

Supervised learning becomes easy when you:

✔ Have labeled data

✔ Train a machine learning model

✔ Evaluate the results

With Python + Scikit-Learn, supervised learning is simple, powerful, and practical 🚀
