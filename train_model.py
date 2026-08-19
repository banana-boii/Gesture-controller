import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

csv_filename = "gesture_dataset.csv"
model_filename = "gesture_model.pkl"

print("loading data")
df = pd.read_csv(csv_filename)

#labels
y = df['label']
#everything else - data
x = df.drop('label', axis=1)

print(f"data loaded {len(df)} frames")

#split data into training and testing 80/20
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.2, random_state = 42, stratify = y)

model = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = -1)

model.fit(X_train, Y_train)

print("\nModel Evaluation")
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Overall accuracy: {accuracy * 100:.2f}%\n")


print("Classification Report:")
print(classification_report(Y_test, y_pred))

joblib.dump(model, model_filename)
print(f"\nModel successfuly saved as '{model_filename}'")