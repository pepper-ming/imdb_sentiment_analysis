"""
IMDB數據預處理模組
用於載入、清理和預處理IMDB電影評論數據
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import os

class IMDBDataPreprocessor:
    def __init__(self, max_features=10000, max_length=500):
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = None
        self.word_index = None
        
    def download_nltk_data(self):
        """下載必要的NLTK數據"""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"NLTK數據下載失敗: {e}")
    
    def clean_text(self, text):
        """清理文本數據"""
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_imdb_data(self):
        """載入IMDB數據集"""
        print("載入IMDB數據集...")
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=self.max_features)
        
        word_index = imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        def decode_review(encoded_review):
            return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
        
        X_train_text = [decode_review(review) for review in X_train]
        X_test_text = [decode_review(review) for review in X_test]
        
        return X_train_text, y_train, X_test_text, y_test
    
    def preprocess_texts(self, texts):
        """預處理文本列表"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        return cleaned_texts
    
    def fit_tokenizer(self, texts):
        """訓練tokenizer"""
        self.tokenizer = Tokenizer(num_words=self.max_features)
        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index
        print(f"詞彙表大小: {len(self.word_index)}")
    
    def texts_to_sequences(self, texts):
        """將文本轉換為序列"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_length)
    
    def prepare_data(self, save_path='data/processed/'):
        """完整的數據準備流程"""
        os.makedirs(save_path, exist_ok=True)
        
        self.download_nltk_data()
        
        X_train_text, y_train, X_test_text, y_test = self.load_imdb_data()
        
        X_train_clean = self.preprocess_texts(X_train_text)
        X_test_clean = self.preprocess_texts(X_test_text)
        
        self.fit_tokenizer(X_train_clean)
        
        X_train_seq = self.texts_to_sequences(X_train_clean)
        X_test_seq = self.texts_to_sequences(X_test_clean)
        
        with open(os.path.join(save_path, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        np.save(os.path.join(save_path, 'X_train.npy'), X_train_seq)
        np.save(os.path.join(save_path, 'y_train.npy'), y_train)
        np.save(os.path.join(save_path, 'X_test.npy'), X_test_seq)
        np.save(os.path.join(save_path, 'y_test.npy'), y_test)
        
        print(f"數據預處理完成！")
        print(f"訓練集大小: {X_train_seq.shape}")
        print(f"測試集大小: {X_test_seq.shape}")
        
        return X_train_seq, y_train, X_test_seq, y_test

def load_processed_data(data_path='data/processed/'):
    """載入已處理的數據"""
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    with open(os.path.join(data_path, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    
    return X_train, y_train, X_test, y_test, tokenizer