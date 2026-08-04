import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("car_data.csv")

# Label Encoding
le = LabelEncoder()

data["Gender"] = le.fit_transform(data["Gender"])
data["Marital_Status"] = le.fit_transform(data["Marital_Status"])
data["Has_Driving_License"] = le.fit_transform(data["Has_Driving_License"])
data["Owns_House"] = le.fit_transform(data["Owns_House"])
data["Buy_Car"] = le.fit_transform(data["Buy_Car"])

# Features
X = data[[
    "Age",
    "Gender",
    "Annual_Income",
    "Marital_Status",
    "Has_Driving_License",
    "Owns_House"
]]

# Target
y = data["Buy_Car"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Predict a new customer
# Age, Gender(Male=1,Female=0), Income,
# Marital(Married=0,Single=1),
# Driving License(Yes=1),
# Own House(Yes=1)

new_customer = [[33,1,700000,0,1,1]]

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nCustomer is likely to BUY a car.")
else:
    print("\nCustomer is NOT likely to buy a car.")

# Display Decision Tree
plt.figure(figsize=(15,8))

tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No","Yes"],
    filled=True
)

plt.show()