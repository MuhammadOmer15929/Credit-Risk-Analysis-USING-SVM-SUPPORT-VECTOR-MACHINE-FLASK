import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# Load and preprocess the dataset
data = pd.read_csv("credit_risk_dataset.csv")

# Identify columns with missing values
columns_with_missing_values = data.columns[data.isnull().any()].tolist()

# Impute missing values with appropriate strategies
imputer = SimpleImputer(strategy="median")  # Use median value to fill missing numerical values
data[columns_with_missing_values] = imputer.fit_transform(data[columns_with_missing_values])

# Select relevant features and target variable
features = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_intent',
            'loan_grade', 'loan_amnt', 'loan_int_rate', 'cb_person_default_on_file', 'cb_person_cred_hist_length']
target = 'loan_status'
data = data[features + [target]]

# Perform label encoding on categorical columns
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate the features and target variable
X = data.drop(target, axis=1)
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Evaluate the model on the testing set
accuracy = svm_model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save the trained model and label encoders
joblib.dump(svm_model, "svm_model.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")

# Generate histograms
plt.figure(figsize=(12, 8))
data.hist()
plt.tight_layout()
plt.savefig('static/histograms.png')

# Generate box plots
plt.figure(figsize=(12, 8))
data.boxplot()
plt.tight_layout()
plt.savefig('static/boxplots.png')

# Generate correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('static/correlation_matrix.png')

# Generate scatter plot matrix
plt.figure(figsize=(12, 8))
sns.pairplot(data)
plt.tight_layout()
plt.savefig('static/scatter_plot_matrix.png')

# Generate confusion matrix (dummy example)
confusion_matrix = np.array([[50, 10], [5, 35]])
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
