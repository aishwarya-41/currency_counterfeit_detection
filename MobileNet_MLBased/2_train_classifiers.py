import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from catboost import CatBoostClassifier
import pandas as pd

# --- Load Pre-computed Features ---
print("Loading pre-computed features and labels from the 'features/' directory...")
try:
    train_features = np.load('features/train_features.npy')
    val_features = np.load('features/val_features.npy')
    test_features = np.load('features/test_features.npy')

    y_train = np.load('features/train_labels.npy')
    y_val = np.load('features/val_labels.npy')
    y_test = np.load('features/test_labels.npy')
    print("Loading complete.")
except FileNotFoundError:
    print("\nERROR: Feature files not found in 'features/' directory.")
    print("Please run '1_extract_features.py' first to generate them.")
    exit()

# --- Define and Train Classifiers ---
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

results = {}
class_names = ['Counterfeit', 'Genuine']

for name, clf in classifiers.items():
    print(f"\n--- Training {name} ---")
    # Train the classifier on the training data
    clf.fit(train_features, y_train)
    
    # Evaluate on the unseen test set
    y_pred = clf.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    results[name] = {
        "Accuracy": accuracy,
        "Precision": report['weighted avg']['precision'],
        "Recall": report['weighted avg']['recall'],
        "F1-Score": report['weighted avg']['f1-score'],
        "Classifier": clf  # Store the trained classifier object
    }
    print(f"{name} Test Accuracy: {accuracy:.4f}")

# --- Comparative Analysis of Ensemble Classifiers ---
# Create a DataFrame for easy comparison, excluding the classifier object itself
results_df = pd.DataFrame(results).T.drop(columns=['Classifier'])
print("\n--- Performance Comparison of Ensemble Classifiers ---")
print(results_df.to_string())

# --- Visualize the Best Performing Model's Results ---
best_model_name = results_df['Accuracy'].idxmax()
best_classifier = results[best_model_name]['Classifier']
y_pred_best = best_classifier.predict(test_features)
cm_best = confusion_matrix(y_test, y_pred_best)

print(f"\n--- Detailed Report for Best Model: {best_model_name} ---")
print(classification_report(y_test, y_pred_best, target_names=class_names))

plt.figure(figsize=(7, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Purples', 
            xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
plt.title(f'Confusion Matrix for Best Model: {best_model_name}', fontsize=14)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig('best_model_confusion_matrix.png')
print(f"\nConfusion matrix for the best model saved as 'best_model_confusion_matrix.png'")
plt.show()
