#%% packages
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
# import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
# add naive classifier
from sklearn.dummy import DummyClassifier
import seaborn as sns


# %% data import
data_path = 'diabetes.csv'
diabetes = pd.read_csv(data_path)
diabetes.describe()

# %% correlation matrix
corr = diabetes.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True)

# %% Separating indepedent (X) / dependent features (y)
X = diabetes.drop(['Outcome'], axis=1)
y = diabetes['Outcome']

# %% Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42)

# approach for train, validation, and test
# 1. split data into train_validation and test
# 2. split train_validation into train and validation

# %% Modeling
steps = [
    ('scaler', StandardScaler()), 
    ('random_forest', RandomForestClassifier(n_estimators = 100, random_state = 42, bootstrap=True))
    # ('decision_tree', DecisionTreeClassifier())
]

pipeline = Pipeline(steps)

# %% train the model
pipeline.fit(X_train, y_train)


# %% create prediction
y_pred_test = pipeline.predict(X_test)

# %% check model performance
cm = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
sns.heatmap(cm, annot=True) 

# %%
accuracy_score(y_true=y_test, y_pred=y_pred_test)

# %% Naive Classifier
DummyClassifier(strategy='most_frequent').fit(X_train, y_train).score(X_test, y_test)

# %% visualise train / test split of data 
# only based on Outcome
diabetes['Outcome']


# %% visualise decision tree
# create a shall decision tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# %% visualise decision tree
import matplotlib.pyplot as plt
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(model, 
               feature_names=X_train.columns,  
               class_names=['0','1'],
               filled=True)
plt.show()
# %% Variable Importance
feat_importances = pd.Series(pipeline.steps[1][1].feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')

# %% 
