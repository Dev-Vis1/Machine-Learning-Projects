from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'iris_model.pkl')

#print succesful
print("Model trained and saved successfully.")