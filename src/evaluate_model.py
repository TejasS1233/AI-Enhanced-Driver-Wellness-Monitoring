import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    auc, precision_recall_curve, average_precision_score
)
from pathlib import Path
import logging

from drowsiness_model import DrowsinessLSTM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EvaluateModel')


def load_test_data(data_file='data/training_data.csv', test_size=0.2):
    if not Path(data_file).exists():
        logger.error(f"Test data not found: {data_file}")
        return None, None
    
    df = pd.read_csv(data_file)
    
    feature_columns = [
        'left_ear', 'right_ear', 'avg_ear', 'mar', 'blink_rate',
        'head_yaw', 'head_pitch', 'head_roll', 'ear_consecutive', 'mar_consecutive'
    ]
    
    X = df[feature_columns].values
    y = df['is_drowsy'].values
    
    split_idx = int(len(X) * (1 - test_size))
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    return X_test, y_test


def evaluate_model(model_path='models/drowsiness_lstm.h5'):
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Please train the model first using: python src/train_lstm.py")
        return
    
    logger.info("="*60)
    logger.info("LSTM Model Evaluation")
    logger.info("="*60)
    
    model = DrowsinessLSTM(sequence_length=30, n_features=10)
    model.load_model(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    X_test, y_test = load_test_data()
    
    if X_test is None:
        logger.error("Failed to load test data")
        return
    
    logger.info(f"Test set: {len(X_test)} samples")
    
    X_sequences, y_sequences = model.prepare_sequences(X_test.tolist(), y_test.tolist())
    
    logger.info(f"Created {len(X_sequences)} test sequences")
    
    logger.info("\nGenerating predictions...")
    y_pred_probs = []
    for i in range(len(X_sequences)):
        features_list = [
            {
                'left_ear': X_sequences[i][j][0],
                'right_ear': X_sequences[i][j][1],
                'avg_ear': X_sequences[i][j][2],
                'mar': X_sequences[i][j][3],
                'blink_rate': X_sequences[i][j][4],
                'head_yaw': X_sequences[i][j][5],
                'head_pitch': X_sequences[i][j][6],
                'head_roll': X_sequences[i][j][7],
                'ear_consecutive': X_sequences[i][j][8],
                'mar_consecutive': X_sequences[i][j][9]
            }
            for j in range(30)
        ]
        prob = model.predict(features_list)
        y_pred_probs.append(prob)
    
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    logger.info("\n" + "="*60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*60)
    print(classification_report(y_sequences, y_pred, target_names=['Alert', 'Drowsy']))
    
    cm = confusion_matrix(y_sequences, y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xticklabels(['Alert', 'Drowsy'])
    axes[0, 0].set_yticklabels(['Alert', 'Drowsy'])
    
    fpr, tpr, thresholds = roc_curve(y_sequences, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True)
    
    precision, recall, _ = precision_recall_curve(y_sequences, y_pred_probs)
    ap = average_precision_score(y_sequences, y_pred_probs)
    
    axes[1, 0].plot(recall, precision, color='blue', lw=2,
                    label=f'Precision-Recall curve (AP = {ap:.2f})')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(True)
    
    axes[1, 1].hist(y_pred_probs[y_sequences == 0], bins=50, alpha=0.5, 
                    label='Alert', color='green')
    axes[1, 1].hist(y_pred_probs[y_sequences == 1], bins=50, alpha=0.5, 
                    label='Drowsy', color='red')
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/evaluation_results.png', dpi=300)
    logger.info("\nEvaluation plots saved to: models/evaluation_results.png")
    plt.show()
    
    threshold_metrics = []
    for threshold in np.arange(0.3, 0.8, 0.05):
        y_pred_thresh = (y_pred_probs > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_sequences, y_pred_thresh).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_metrics.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    df_thresholds = pd.DataFrame(threshold_metrics)
    
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD ANALYSIS")
    logger.info("="*60)
    print(df_thresholds.to_string(index=False))
    
    best_f1_idx = df_thresholds['f1'].idxmax()
    best_threshold = df_thresholds.loc[best_f1_idx, 'threshold']
    logger.info(f"\nBest F1 Score: {df_thresholds.loc[best_f1_idx, 'f1']:.4f}")
    logger.info(f"Optimal Threshold: {best_threshold:.2f}")
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation completed successfully!")
    logger.info("="*60)


if __name__ == '__main__':
    evaluate_model()
