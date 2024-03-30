import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from process import process
from visualization import plot_spam_ham_count, confusion_matrix_plot
import time
import psutil
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

start_time = time.time()

data_file = "enron_spam_data.csv"
df = pd.read_csv(data_file)
df = process(df)
plot_spam_ham_count(df)

x = df['Subject'] + ' ' + df['Message']
y = df['Label']

# Set seed for reproducibility
seed = 11

# Split train and test set for training and validation the performace
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

vectorizer = CountVectorizer()
print("Statr Training !!")

# Convert text to BoW features
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Logistic Regression 
cs = [0.01, 0.1, 1, 10]
logreg = LogisticRegressionCV(Cs=cs, cv=5, random_state=seed, n_jobs=2)
logreg.fit(x_train, y_train)

# Random Forest 
random_forest = RandomForestClassifier(random_state=seed)

# Define grid of hyperparameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300]
}

# Perform grid search for Random Forest
grid_search = GridSearchCV(random_forest, param_grid, cv=5, n_jobs=2)
grid_search.fit(x_train, y_train)

random_forest_best = grid_search.best_estimator_

# Train the best Random Forest model on the entire training set
random_forest_best.fit(x_train, y_train)

# Print best parameters from grid search
# Get the best parameters from both models
print("\nBest Parameters for Logistic Regression:")
print(logreg.C_)
print("\nBest Parameters for Random Forest:")
print(grid_search.best_params_, '\n')

# Make predictions 
logreg_train_pred = logreg.predict(x_train)
logreg_test_pred = logreg.predict(x_test)
rf_train_pred = random_forest_best.predict(x_train)
rf_test_pred = random_forest_best.predict(x_test)

# Calculate accuracy of the models
logreg_train_acc = accuracy_score(y_train, logreg_train_pred)
logreg_test_acc = accuracy_score(y_test, logreg_test_pred)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

# Calculate recall, precision, and confusion matrix 
logreg_test_recall = recall_score(y_test, logreg_test_pred)
rf_test_recall = recall_score(y_test, rf_test_pred)

logreg_test_precision = precision_score(y_test, logreg_test_pred)
rf_test_precision = precision_score(y_test, rf_test_pred)

logreg_test_cm = confusion_matrix(y_test, logreg_test_pred, normalize = 'all')
rf_test_cm = confusion_matrix(y_test, rf_test_pred, normalize = 'all')

# Average the coefficients across different iterations
coefficients = np.mean(logreg.coef_, axis=0)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Create a dictionary mapping words to their coefficients
word_coefficients = dict(zip(feature_names, coefficients))

# Sort the words based on their coefficients in descending order
sorted_words = sorted(word_coefficients.items(), key=lambda x: x[1], reverse=True)

# Print results for logistic regression
print("Logistic Regression:")
print("Train Accuracy:", logreg_train_acc)
print("Test Accuracy:", logreg_test_acc)
print("Test Recall:", logreg_test_recall)
print("Test Precision:", logreg_test_precision)
confusion_matrix_plot(logreg_test_cm, 'Logistic Regression')

# Print results for best random forest model
print("\nRandom Forest:")
print("Train Accuracy:", rf_train_acc)
print("Test Accuracy:", rf_test_acc)
print("Test Recall:", rf_test_recall)
print("Test Precision:", rf_test_precision)
confusion_matrix_plot(rf_test_cm, 'Random Forest')

# Print the top 10 most significant words
top_words = sorted_words[:10]
for word, coefficient in top_words:
    print(word, ":", coefficient)

cpu_utilization = psutil.cpu_percent(interval=None)
end_time = time.time() - start_time

print(f"\nTotal execution time: {end_time} seconds")
print(f"CPU utilization: {cpu_utilization}%")
