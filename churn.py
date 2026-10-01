import pandas as pd
import matplotlib.pyplot as plt
# Load customer data
data = pd.read_csv("customers.csv")

# Display the first rows
print("CUSTOMER DATA")
print(data.head())

# Display dataset information
print("\nDATA INFORMATION")
print(data.info())
# Prepare data for Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Select features and target
X = data[["Age", "MonthlyCharges", "ContractMonths", "SupportCalls"]]
y = data["Churn"].map({"Yes": 1, "No": 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nMODEL RESULTS")
print("Accuracy:", accuracy)
# Predict churn for a new customer
new_customer = pd.DataFrame(
    [[27, 145, 4, 6]],
    columns=["Age", "MonthlyCharges", "ContractMonths", "SupportCalls"]
)

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nNew Customer Prediction: Likely to Churn")
else:
    print("\nNew Customer Prediction: Likely to Stay")
# Churn distribution chart
churn_counts = data["Churn"].value_counts()

churn_counts.plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig("churn_distribution.png")
plt.show()