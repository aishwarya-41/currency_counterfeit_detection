import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from catboost import CatBoostClassifier
import pandas as pd
import os

# --- Load Pre-computed Hybrid Features ---
print("Loading pre-computed hybrid features and labels...")
try:
    train_features = np.load('features/train_features.npy')
    test_features = np.load('features/test_features.npy')
    y_train = np.load('features/train_labels.npy')
    y_test = np.load('features/test_labels.npy')
    print("Loading complete.")
except FileNotFoundError:
    print("\nERROR: Feature files not found. Please run '1_extract_hybrid_features.py' first.")
    exit()

# --- Define and Train Ensemble Classifiers (as per your report) ---
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
    clf.fit(train_features, y_train)
    y_pred = clf.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    results[name] = {
        "Accuracy": accuracy,
        "Precision": report['weighted avg']['precision'],
        "Recall": report['weighted avg']['recall'],
        "F1-Score": report['weighted avg']['f1-score'],
        "Classifier": clf
    }
    print(f"{name} Test Accuracy: {accuracy:.4f}")

# --- Implement Ensemble Voting (as described in your report) ---
# For this implementation, we'll use a simple hard voting ensemble
print("\n--- Training Ensemble with Hard Voting ---")
voting_clf = VotingClassifier(
    estimators=[(name, results[name]['Classifier']) for name in classifiers],
    voting='hard' # Each model gets one vote
)
voting_clf.fit(train_features, y_train)
y_pred_ensemble = voting_clf.predict(test_features)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
results['Ensemble (Hard Vote)'] = {"Accuracy": ensemble_accuracy, "Classifier": voting_clf}


# --- Comparative Analysis ---
results_df = pd.DataFrame(results).T.drop(columns=['Classifier'])
print("\n--- Performance Comparison of All Classifiers ---")
print(results_df.to_string())

# --- Visualize the Best Performing Model's Results ---
best_model_name = results_df['Accuracy'].idxmax()
best_classifier = results[best_model_name]['Classifier']
y_pred_best = best_classifier.predict(test_features)
cm_best = confusion_matrix(y_test, y_pred_best)

print(f"\n--- Detailed Report for Best Model: {best_model_name} ---")
print(classification_report(y_test, y_pred_best, target_names=class_names))

plt.figure(figsize=(7, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
plt.title(f'Confusion Matrix for Best Model: {best_model_name}', fontsize=14)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig('best_hybrid_model_confusion_matrix.png')
print(f"\nConfusion matrix for the best model saved as 'best_hybrid_model_confusion_matrix.png'")
plt.show()
