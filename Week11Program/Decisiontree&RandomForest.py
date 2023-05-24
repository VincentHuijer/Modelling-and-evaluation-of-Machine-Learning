import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_excel('Outdoordb.xlsx')

correlation = df.corr()
# Decision Tree model
# X bevat de features en y bevat de target
X = df.drop('RETURN_REASON_CODE', axis=1)
y = df['RETURN_REASON_CODE']

# Data splitsen in trainingsdata en testdata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier maken en trainen
clf_dt = DecisionTreeClassifier(max_depth=2) #in mijn document heb ik een max depth van 3 gehanteerd, maar 2 geeft een betere accuracy van 53%
clf_dt.fit(X_train, y_train)
# Voorspellingen maken voor de testdata
y_pred_dt = clf_dt.predict(X_test)

# Accuracy van het Decision Tree model berekenen
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print('Accuracy van Decision Tree model:', accuracy_dt)

# Visualize Decision Tree
plt.figure(figsize=(12, 6))
tree.plot_tree(clf_dt, filled=True)
plt.title('Decision Tree Classifier')
plt.show()
# Random Forest model
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
# Voorspellingen maken voor de testdata
y_pred_rf = clf_rf.predict(X_test)
# Accuracy van het Random Forest model berekenen
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print('Accuracy van Random Forest model:', accuracy_rf)
# Bar chart comparing accuracy of Decision Tree and Random Forest models
accuracy_scores = [accuracy_dt, accuracy_rf]
models = ['Decision Tree', 'Random Forest']
plt.figure(figsize=(8, 6))
plt.bar(models, accuracy_scores)
plt.title('Accuracy Comparison: Decision Tree vs Random Forest')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()
