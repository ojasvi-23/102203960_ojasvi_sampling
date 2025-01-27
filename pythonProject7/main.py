import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load Dataset
data = pd.read_csv('Creditcard_data.csv')


print(data.columns)
X = data.drop('Class', axis=1)  # Replace 'Class' with the actual column name
y = data['Class']

# Step 2: Balance the Dataset (Example: Oversampling using SMOTE)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Step 3: Apply Random Sampling for Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Step 4: Train Model using RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
