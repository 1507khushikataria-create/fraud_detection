import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix

data=pd.read_csv("creditcard.csv")

print(data.head())

sns.heatmap(data.sample(10000).corr(),cmap="coolwarm")
plt.title("correlation map")
plt.show()

sns.countplot(x='Class',data=data)
plt.title("distribution graph")
plt.show()

X=data.drop('Class',axis=1)
y=data['Class']

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model=RandomForestClassifier(class_weight="balanced")
model.fit(X_train,y_train)

predictions=model.predict(X_test)

score=f1_score(y_test,predictions)
print("f1 score:",score)

recall=recall_score(y_test,predictions)
print("recall score:",recall)

precision=precision_score(y_test,predictions)
print("preciction score:",precision)

matrix=confusion_matrix(y_test,predictions)
print("confusion matrix:",matrix)