import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

iris = datasets.load_iris(as_frame= True)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1) Decision Tree
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)
y_dt = clf_dt.predict(X_test)
print("Decision Tree\n", classification_report(y_test, y_dt))

# Plot tree
plt.figure(figsize=(8,6))
plot_tree(clf_dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree")
plt.show()

# 2) Naive Bayes
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
y_nb = clf_nb.predict(X_test)
print("Naive Bayes\n", classification_report(y_test, y_nb))

# 3) SVM
clf_svm = SVC(kernel='rbf', probability=True, random_state=42)
clf_svm.fit(X_train, y_train)
y_svm = clf_svm.predict(X_test)
print("SVM\n", classification_report(y_test, y_svm))

# Confusion Matrices
for name, y_pred in [("DT", y_dt), ("NB", y_nb), ("SVM", y_svm)]:
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                                  display_labels=iris.target_names)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix {name}")
    plt.show()