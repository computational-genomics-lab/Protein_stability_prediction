import pandas as pd

# Load the uploaded Excel file
file_path = 'initial_file_AI_ML.xlsx'
data = pd.read_excel(file_path)

# Drop unnecessary columns and inspect the relevant ones for cleaning
# Identify columns with significant missing data or irrelevant information
columns_to_drop = [col for col in data.columns if 'Unnamed' in col or data[col].isnull().mean() > 0.5]

# Drop the identified columns
cleaned_data = data.drop(columns=columns_to_drop)

# Display the remaining columns and the first few rows
cleaned_data.head(), cleaned_data.info()


#ANALYSIS TIME
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Step 1: Clean and encode target column
# Map 'stable' to 1 and 'unstable' to 0 in the Instability index
cleaned_data['Instability index'] = cleaned_data['Instability index'].map({'stable': 1, 'unstable': 0})

# print(cleaned_data.columns.tolist())

# Step 2: Select relevant numeric features and handle missing values
selected_features = ['Theoretical PI', 'Molecular weight ', 'No of amino acids', 'Alipathic Index' ]
cleaned_data = cleaned_data[selected_features + ['Instability index']].dropna()

# Split features and target
X = cleaned_data[selected_features]
y = cleaned_data['Instability index']

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("report :", report, '\n')
print("The accuracy is :", accuracy)
print("confusion matrix", '\n', conf_matrix)

#VISUALISATION OF THE RESULTS

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data from the confusion matrix
conf_matrix = np.array([[0, 8],
                        [1, 64]])

# Labels for the classes
class_names = ["Unstable (0.0)", "Stable (1.0)"]

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Bar plot for precision, recall, and f1-score
metrics = {
    "Metric": ["Precision", "Recall", "F1-Score"],
    "Unstable (0.0)": [0.0, 0.0, 0.0],
    "Stable (1.0)": [0.89, 0.98, 0.93],
}

# Convert to format for plotting
categories = metrics["Metric"]
unstable_scores = metrics["Unstable (0.0)"]
stable_scores = metrics["Stable (1.0)"]

x = np.arange(len(categories))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, unstable_scores, width, label="Unstable (0.0)", color='orange')
bars2 = ax.bar(x + width/2, stable_scores, width, label="Stable (1.0)", color='green')

# Add some text for labels, title, and axes ticks
ax.set_xlabel("Metrics")
ax.set_title("Performance Metrics by Class")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Display values on top of bars
for bar in bars1 + bars2:
    height = bar.get_height()
    if height > 0:  # Avoid displaying 0
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.show()