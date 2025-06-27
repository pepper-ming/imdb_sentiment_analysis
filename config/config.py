"""
IMDB情感分析專案配置文件
"""

# 數據配置
DATA_CONFIG = {
    'raw_data_path': 'data/raw/',
    'processed_data_path': 'data/processed/',
    'max_features': 10000,
    'max_length': 500,
    'test_size': 0.2,
    'random_state': 42
}

# 模型配置
MODEL_CONFIG = {
    'embedding_dim': 128,
    'lstm_units': 64,
    'dropout_rate': 0.5,
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001
}

# 路徑配置
PATHS = {
    'models': 'models/',
    'results': 'results/',
    'notebooks': 'notebooks/'
}