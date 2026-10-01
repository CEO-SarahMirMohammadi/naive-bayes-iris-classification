# Naive Bayes Iris Classification

A machine learning project that demonstrates **Naive Bayes classification** using the Iris dataset. The project trains a Gaussian Naive Bayes model, saves the trained model to a file, loads the model back into memory, and uses the loaded model to make a prediction.

## Overview

This project is based on Example 3.7, **Naive Bayes Iris**, and extends the original program by adding model persistence.

The original workflow is:

```text
Load Dataset → Train Model → Predict
```

This project extends it to:

```text
Load Dataset
      ↓
Train Gaussian Naive Bayes Model
      ↓
Save Trained Model
      ↓
Load Model from File
      ↓
Make Prediction
```

The main purpose is to demonstrate how a trained machine learning model can be saved and reused without having to retrain it every time.

## Algorithm

### Gaussian Naive Bayes

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem.

The **GaussianNB** implementation assumes that the features follow a Gaussian (normal) distribution within each class.

For the Iris dataset, the model uses four features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The model estimates the probability of each Iris species based on these feature values and predicts the class with the highest probability.

## Dataset

The project uses the **Iris dataset** provided by Scikit-learn.

The dataset contains:

* **150 samples**
* **4 numerical features**
* **3 classes**

The three Iris species are:

```text
0 → Setosa
1 → Versicolor
2 → Virginica
```

Each sample contains:

```text
[sepal length, sepal width, petal length, petal width]
```

## Model Persistence

One of the main purposes of this project is demonstrating model persistence.

After training, the model is saved using `joblib`:

```python
joblib.dump(clf, "naive_bayes_iris_model.pkl")
```

The saved model can then be loaded:

```python
loaded_model = joblib.load("naive_bayes_iris_model.pkl")
```

The loaded model is then used to make a prediction:

```python
prediction = loaded_model.predict([[5.0, 3.4, 1.5, 0.4]])
```

This means the model does not need to be trained again before making a prediction.

## Project Structure

```text
naive-bayes-iris-classification/
│
├── naive_bayes_iris.py
├── requirements.txt
├── README.md
└── .gitignore
```

After running the program, an additional local file will be generated:

```text
naive_bayes_iris_model.pkl
```

This file is ignored by Git because it is a generated model artifact.

## Requirements

| Requirement  | Purpose                              |
| ------------ | ------------------------------------ |
| Python 3.x   | Programming language                 |
| scikit-learn | Dataset and Naive Bayes model        |
| joblib       | Saving and loading the trained model |

## Installation

Clone the repository:

```bash
git clone https://github.com/CEO-SarahMirMohammadi/naive-bayes-iris-classification.git
```

Move into the project directory:

```bash
cd naive-bayes-iris-classification
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Program

Run:

```powershell
python naive_bayes_iris.py
```

The program will:

1. Load the Iris dataset.
2. Create a Gaussian Naive Bayes classifier.
3. Train the classifier.
4. Save the trained model as `naive_bayes_iris_model.pkl`.
5. Load the saved model.
6. Use the loaded model to predict the class of a new sample.

## Example Prediction

The program predicts the class of:

```text
[5.0, 3.4, 1.5, 0.4]
```

The prediction returned by Scikit-learn is:

```text
[0]
```

Class `0` corresponds to:

```text
Iris Setosa
```

## Workflow

```text
                 Iris Dataset
                      │
                      ▼
              Load X and y
                      │
                      ▼
          Create GaussianNB Model
                      │
                      ▼
                Train Model
                      │
                      ▼
             Save Model (.pkl)
                      │
                      ▼
             Load Model (.pkl)
                      │
                      ▼
            New Iris Observation
                      │
                      ▼
                  Prediction
```

## Key Concepts

### 1. Naive Bayes

Naive Bayes is a supervised classification algorithm based on probability.

### 2. GaussianNB

`GaussianNB` is the Scikit-learn implementation of Gaussian Naive Bayes.

```python
clf = GaussianNB()
```

### 3. Model Training

The model learns the relationship between the Iris features and their corresponding classes:

```python
clf.fit(X, y)
```

### 4. Model Persistence

The trained model is serialized to a file using Joblib:

```python
joblib.dump(clf, model_path)
```

### 5. Model Loading

The saved model can later be restored:

```python
loaded_model = joblib.load(model_path)
```

### 6. Prediction

The restored model can be used exactly like the original trained model:

```python
loaded_model.predict(sample)
```

## Why Save a Machine Learning Model?

Saving a trained model is useful when:

* Training takes significant time.
* The model needs to be reused later.
* A model needs to be deployed as an application or API.
* Predictions need to be made without retraining.
* The trained model needs to be transferred between environments.

For a small dataset such as Iris, retraining is extremely fast. However, the same concept becomes important for larger machine learning systems.

## Output

A typical execution will produce output similar to:

```text
Iris dataset:
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 ...
 [6.2 3.4 5.4 2.3]]

Model trained successfully.
Model saved to: naive_bayes_iris_model.pkl
Model file created successfully.
Model loaded successfully.

Prediction:
[0]
```

## Future Improvements

Possible improvements include:

* Add train/test dataset splitting.
* Calculate classification accuracy.
* Generate a confusion matrix.
* Display a classification report.
* Test multiple new Iris samples.
* Compare Naive Bayes with SVM, Decision Tree, and K-Nearest Neighbors.
* Create a prediction API using FastAPI.
* Add model versioning.
* Store model metadata alongside the serialized model.

## License

This project is licensed under the **MIT License**.

You are free to use, modify, distribute, and build upon this project under the terms of the MIT License.
