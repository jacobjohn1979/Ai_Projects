from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib


# Example transaction dataset
data = {
    "amount": [100,120,95,110,10,115,5000,90,102,108],
    "hour": [10,11,9,10,11,10,3,12,10,9]
}

df = pd.DataFrame(data)

# Train Isolation Forest
model = IsolationForest(contamination=0.1)

model.fit(df)

# Test new transaction
test_transaction = [[105,10]]

prediction = model.predict(test_transaction)

if prediction[0] == -1:
    print("⚠️ Anomaly detected")
else:
    print("Transaction looks normal")

joblib.dump(model, "anomaly_model.pkl")
print("Model saved successfully")