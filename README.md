# Logistic Regression: Student Pass/Fail Prediction

This project uses **Logistic Regression** to predict whether a student is likely to **pass or fail** based on their **Total Grades** and **Total Percentage**.

---

## 📂 Dataset

* **Input File**: `students_data_annoynmous_names.csv`
* **Columns Used**:

  * `TotalGrades`
  * `TotalPercentage`
  * `Status` (Target: 'ناجح', 'راسب', 'دور ثاني')

---

## 🧪 Features

* Converts percentage from string to float.
* Maps target labels:

  * 'ناجح' → `1`
  * 'راسب', 'دور ثاني' → `0`
* Uses `LogisticRegression` from `sklearn`.

---

## 📊 Visualizations

* Scatter plot of Grades vs. Percentage colored by result.
* Histogram + KDE of `TotalPercentage` distribution.
* ROC Curve for evaluating classification quality.

---

## 🧠 Model Training

* Model: `sklearn.linear_model.LogisticRegression`
* Train/Test Split: `67% train / 33% test`
* Metrics:

  * Confusion Matrix
  * Accuracy
  * Precision, Recall, F1 (via `classification_report`)
  * ROC Curve + AUC

---

## 🔍 Prediction Demo

The script allows user input to test the model:

```bash
أدخل مجموع الطالب: 300
أدخل النسبة المئوية: 60
```

---

## 📈 Example Output

```
النتيجة: ناجح
---

## 🚀 Requirements


* Python 3.7+
* scikit-learn
* pandas
* matplotlib
* seaborn

Install with:

```bash
pip install scikit-learn pandas matplotlib seaborn
```

---

## 🧾 License

This project is for educational purposes only.

---

## 📌 Notes

* Improve model by trying more features (subject-wise grades).
* Try handling class imbalance with `class_weight='balanced'` or oversampling.

---

## 🤝 Contribution

Pull requests and suggestions welcome!
