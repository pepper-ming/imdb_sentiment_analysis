"""
IMDB情感分析模型定義
包含多種深度學習模型架構
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, 
    Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Bidirectional, Input, Concatenate
)
from tensorflow.keras.optimizers import Adam
import numpy as np

class SentimentModels:
    def __init__(self, max_features=10000, max_length=500, embedding_dim=128):
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
    
    def simple_lstm(self, lstm_units=64, dropout_rate=0.5):
        """簡單LSTM模型"""
        model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_length),
            LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def bidirectional_lstm(self, lstm_units=64, dropout_rate=0.5):
        """雙向LSTM模型"""
        model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def cnn_model(self, filters=128, kernel_size=5, dropout_rate=0.5):
        """CNN模型"""
        model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_length),
            Conv1D(filters, kernel_size, activation='relu'),
            MaxPooling1D(pool_size=4),
            Conv1D(filters//2, kernel_size, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def cnn_lstm_hybrid(self, filters=64, kernel_size=3, lstm_units=32, dropout_rate=0.5):
        """CNN+LSTM混合模型"""
        model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_length),
            Conv1D(filters, kernel_size, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def attention_lstm(self, lstm_units=64, dropout_rate=0.5):
        """帶注意力機制的LSTM模型"""
        input_layer = Input(shape=(self.max_length,))
        
        embedding = Embedding(self.max_features, self.embedding_dim)(input_layer)
        
        lstm_out = LSTM(lstm_units, return_sequences=True, 
                       dropout=dropout_rate, recurrent_dropout=dropout_rate)(embedding)
        
        attention_weights = Dense(1, activation='tanh')(lstm_out)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        
        attended_output = tf.reduce_sum(lstm_out * attention_weights, axis=1)
        
        dense = Dense(32, activation='relu')(attended_output)
        dropout = Dropout(dropout_rate)(dense)
        output = Dense(1, activation='sigmoid')(dropout)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_model(self, model_type='simple_lstm', **kwargs):
        """根據類型獲取模型"""
        model_functions = {
            'simple_lstm': self.simple_lstm,
            'bidirectional_lstm': self.bidirectional_lstm,
            'cnn': self.cnn_model,
            'cnn_lstm': self.cnn_lstm_hybrid,
            'attention_lstm': self.attention_lstm
        }
        
        if model_type not in model_functions:
            raise ValueError(f"不支持的模型類型: {model_type}")
        
        return model_functions[model_type](**kwargs)

def create_callbacks(model_name, patience=3):
    """創建訓練回調函數"""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks