import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report

# Load and preprocess data
df = pd.read_csv("training_dataset.csv")
df = df.dropna()

X = df.drop(columns=["home_win"])
y = df["home_win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define parameter grid
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_features': ['sqrt', 'log2']
}

# Set up RandomizedSearchCV
rf = RandomForestClassifier(random_state=42)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',  # or 'f1'
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=1
)

# Run search
search.fit(X_train, y_train)

# Show best results
best_model = search.best_estimator_
print("Best Parameters:")
print(search.best_params_)

# Evaluate on test set
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
