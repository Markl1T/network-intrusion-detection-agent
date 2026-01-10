import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay
)

def evaluate_model(model, X_test, y_test, model_name="Model", binary=True, encoder=None):
    print(f"\n{'=' * 10}")
    print(f" {model_name} — Evaluation ")
    print(f"{'=' * 10}")

    # Predictions
    y_pred = model.predict(X_test)

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Display confusion matrix as a plot
    if not binary and encoder is not None:
        labels = encoder.classes_
    else:
        labels = ["Benign", "Attack"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f"{model_name} — Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    if binary and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC : {roc:.4f}")

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        print(f"PR-AUC  : {pr_auc:.4f}")

        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} — ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_name} — Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    if not binary:
        if encoder is not None and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test) 
            for i, class_name in enumerate(encoder.classes_):
                plt.figure()
                precision, recall, _ = precision_recall_curve((y_test==i).astype(int), y_prob[:, i])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, label=f"{class_name} PR-AUC={pr_auc:.4f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"{model_name} — Precision-Recall Curve ({class_name})")
                plt.legend()
                plt.grid(True)
                plt.show()

    print(f"{'=' * 20}\n")
