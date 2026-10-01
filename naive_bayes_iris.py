from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import joblib
import os


# Load the Iris dataset
X, y = load_iris(return_X_y=True)

print("Iris dataset:")
print(X)


# Create and train the Naive Bayes model
clf = GaussianNB()
clf.fit(X, y)

print("\nModel trained successfully.")


# Save the trained model to a file
model_path = "naive_bayes_iris_model.pkl"
joblib.dump(clf, model_path)

print(f"Model saved to: {model_path}")


# Check that the model file exists
if os.path.exists(model_path):
    print("Model file created successfully.")


# Load the model from the file
loaded_model = joblib.load(model_path)

print("Model loaded successfully.")


# Make a prediction using the loaded model
sample = [[5.0, 3.4, 1.5, 0.4]]
prediction = loaded_model.predict(sample)

print("\nPrediction:")
print(prediction)
