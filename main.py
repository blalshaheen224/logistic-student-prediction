# Import necessary libraries
import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
import seaborn as ns

# Load the dataset
data = pd.read_csv("students_data_annoynmous_names.csv")

# Convert percentage column from string (with %) to float
data["TotalPercentage"] = data["TotalPercentage"].str.replace('%', '').astype(float)

# Encode the target variable: "ناجح" -> 1, "راسب"/"دور ثاني" -> 0
data['Status_code'] = data['Status'].map({
    'راسب': 0,
    'ناجح': 1,
    'دور ثاني': 0,
})

# Define features (X) and target (Y)
X = data[['TotalGrades', 'TotalPercentage']]
Y = data['Status_code']

# Visualize data distribution: Pass vs. Fail
ns.scatterplot(data=data, x='TotalGrades', y='TotalPercentage', hue='Status_code')
plt.title("Grades vs Percentage by Pass/Fail")
plt.show()

# Plot histogram of TotalPercentage with KDE
ns.histplot(data['TotalPercentage'], kde=True)
plt.title("Distribution of Total Percentage")
plt.show()

# Split the dataset into training and testing sets
x_trian, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(x_trian, y_train)

# Make predictions
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]  # Get probability of class 1 (pass)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print()

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print()

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guess
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Print model coefficients
print()
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict single student outcome based on input
def predict_student_status(grade, percentage):
    pred = model.predict([[grade, percentage]])
    return "ناجح" if pred[0] == 1 else "راسب/دور ثاني"

# Test with user input
grade = float(input("أدخل مجموع الطالب: "))
perc = float(input("أدخل النسبة المئوية: "))
result = predict_student_status(grade, perc)
print("النتيجة:", result)
