from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize base models
dt_model = DecisionTreeClassifier(random_state=42)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
lr_model = LogisticRegression(max_iter=200, random_state=42)

# Create the ensemble model using a Voting Classifier
ensemble_model = VotingClassifier(
    estimators=[('dt', dt_model), ('svm', svm_model), ('lr', lr_model)],
    voting='soft'  # Use soft voting to average predicted probabilities
)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Predict using the test set
y_pred = ensemble_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Ensemble Model Accuracy:", accuracy)
print("Classification Report:\n", report)
