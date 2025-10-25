import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path


class DrowsinessLSTM:
    
    def __init__(self, sequence_length=30, n_features=10):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger('DriverWellness.LSTM')
        
    def build_model(self):
        model = Sequential([
            LSTM(64, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features),
                 name='lstm_1'),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(64, return_sequences=False, name='lstm_2'),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(32, activation='relu', name='dense_1'),
            Dropout(0.2),
            
            Dense(16, activation='relu', name='dense_2'),
            
            Dense(1, activation='sigmoid', name='output')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        self.logger.info(f"LSTM model built: {self.sequence_length} steps, {self.n_features} features")
        return model
    
    def prepare_sequences(self, features_list, labels=None):
        sequences = []
        sequence_labels = []
        
        for i in range(len(features_list) - self.sequence_length + 1):
            sequence = features_list[i:i + self.sequence_length]
            sequences.append(sequence)
            
            if labels is not None:
                sequence_labels.append(labels[i + self.sequence_length - 1])
        
        X = np.array(sequences)
        
        if labels is not None:
            y = np.array(sequence_labels)
            return X, y
        return X
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        if self.model is None:
            self.build_model()
        
        n_samples = X_train.shape[0]
        X_train_reshaped = X_train.reshape(n_samples, self.sequence_length, self.n_features)
        
        X_train_scaled = np.zeros_like(X_train_reshaped)
        for i in range(n_samples):
            X_train_scaled[i] = self.scaler.fit_transform(X_train_reshaped[i])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('models/best_drowsiness_model.h5', 
                          monitor='val_loss', 
                          save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            n_val_samples = X_val.shape[0]
            X_val_reshaped = X_val.reshape(n_val_samples, self.sequence_length, self.n_features)
            X_val_scaled = np.zeros_like(X_val_reshaped)
            for i in range(n_val_samples):
                X_val_scaled[i] = self.scaler.transform(X_val_reshaped[i])
            validation_data = (X_val_scaled, y_val)
        
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("Training completed")
        return history
    
    def predict(self, feature_sequence):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if len(feature_sequence) < self.sequence_length:
            return 0.0
        
        sequence = feature_sequence[-self.sequence_length:]
        X = np.array(sequence).reshape(1, self.sequence_length, self.n_features)
        
        X_scaled = self.scaler.transform(X.reshape(self.sequence_length, self.n_features))
        X_scaled = X_scaled.reshape(1, self.sequence_length, self.n_features)
        
        prediction = self.model.predict(X_scaled, verbose=0)
        return float(prediction[0][0])
    
    def save_model(self, model_path='models/drowsiness_lstm.h5', scaler_path='models/scaler.pkl'):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='models/drowsiness_lstm.h5', scaler_path='models/scaler.pkl'):
        if not Path(model_path).exists():
            self.logger.warning(f"Model file not found: {model_path}")
            return False
        
        self.model = keras.models.load_model(model_path)
        
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
        else:
            self.logger.warning(f"Scaler file not found: {scaler_path}")
        
        self.logger.info(f"Model loaded from {model_path}")
        return True
    
    def get_model_summary(self):
        if self.model is None:
            return "Model not built yet"
        return self.model.summary()


def extract_feature_vector(features_dict):
    return [
        features_dict.get('left_ear', 0.0),
        features_dict.get('right_ear', 0.0),
        features_dict.get('avg_ear', 0.0),
        features_dict.get('mar', 0.0),
        features_dict.get('blink_rate', 0.0),
        features_dict.get('head_yaw', 0.0),
        features_dict.get('head_pitch', 0.0),
        features_dict.get('head_roll', 0.0),
        float(features_dict.get('ear_consecutive', 0)),
        float(features_dict.get('mar_consecutive', 0))
    ]
