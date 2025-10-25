import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import sys

from drowsiness_model import DrowsinessLSTM, extract_feature_vector


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TrainLSTM')


def load_training_data(data_file='data/training_data.csv'):
    if not Path(data_file).exists():
        logger.error(f"Training data not found: {data_file}")
        return None, None
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} samples from {data_file}")
    
    feature_columns = [
        'left_ear', 'right_ear', 'avg_ear', 'mar', 'blink_rate',
        'head_yaw', 'head_pitch', 'head_roll', 'ear_consecutive', 'mar_consecutive'
    ]
    
    X = df[feature_columns].values
    y = df['is_drowsy'].values
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    logger.info(f"Drowsy samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    logger.info(f"Alert samples: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    
    return X, y


def create_synthetic_data(n_samples=5000):
    logger.warning("Creating synthetic training data for demonstration")
    logger.info("In production, use real labeled data from actual driving sessions")
    
    features_list = []
    labels = []
    
    for i in range(n_samples):
        if np.random.random() < 0.3:
            label = 1
            left_ear = np.random.uniform(0.10, 0.20)
            right_ear = np.random.uniform(0.10, 0.20)
            avg_ear = (left_ear + right_ear) / 2
            mar = np.random.uniform(0.5, 0.8)
            blink_rate = np.random.uniform(25, 40)
            head_yaw = np.random.uniform(-30, 30)
            head_pitch = np.random.uniform(-25, -10)
            head_roll = np.random.uniform(-20, 20)
            ear_consecutive = np.random.randint(3, 15)
            mar_consecutive = np.random.randint(2, 10)
        else:
            label = 0
            left_ear = np.random.uniform(0.22, 0.35)
            right_ear = np.random.uniform(0.22, 0.35)
            avg_ear = (left_ear + right_ear) / 2
            mar = np.random.uniform(0.3, 0.55)
            blink_rate = np.random.uniform(12, 22)
            head_yaw = np.random.uniform(-15, 15)
            head_pitch = np.random.uniform(-10, 10)
            head_roll = np.random.uniform(-10, 10)
            ear_consecutive = np.random.randint(0, 3)
            mar_consecutive = np.random.randint(0, 2)
        
        features = [left_ear, right_ear, avg_ear, mar, blink_rate,
                   head_yaw, head_pitch, head_roll, ear_consecutive, mar_consecutive]
        features_list.append(features)
        labels.append(label)
    
    return np.array(features_list), np.array(labels)


def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    logger.info("Training history plot saved to models/training_history.png")
    plt.show()


def main():
    logger.info("="*60)
    logger.info("LSTM Drowsiness Detection Model Training")
    logger.info("="*60)
    
    sequence_length = 30
    n_features = 10
    
    X, y = load_training_data()
    
    if X is None:
        logger.info("No training data found, creating synthetic data...")
        X, y = create_synthetic_data(n_samples=5000)
    
    model = DrowsinessLSTM(sequence_length=sequence_length, n_features=n_features)
    
    X_sequences, y_sequences = model.prepare_sequences(X.tolist(), y.tolist())
    
    logger.info(f"Created {len(X_sequences)} sequences of length {sequence_length}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_sequences, 
        test_size=0.2, 
        random_state=42,
        stratify=y_sequences
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    logger.info(f"Training set: {len(X_train)} sequences")
    logger.info(f"Validation set: {len(X_val)} sequences")
    logger.info(f"Test set: {len(X_test)} sequences")
    
    logger.info("\nBuilding and training model...")
    model.build_model()
    model.get_model_summary()
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    logger.info("\nEvaluating on test set...")
    X_test_scaled = np.zeros_like(X_test)
    for i in range(len(X_test)):
        X_test_scaled[i] = model.scaler.transform(
            X_test[i].reshape(sequence_length, n_features)
        ).reshape(sequence_length, n_features)
    
    test_loss, test_acc, test_precision, test_recall = model.model.evaluate(
        X_test_scaled, y_test, verbose=0
    )
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    logger.info(f"Test F1 Score: {f1_score:.4f}")
    
    model.save_model()
    
    plot_training_history(history)
    
    logger.info("="*60)
    logger.info("Training completed successfully!")
    logger.info("Model saved to: models/drowsiness_lstm.h5")
    logger.info("="*60)


if __name__ == '__main__':
    main()
