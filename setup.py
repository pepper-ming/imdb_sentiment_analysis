"""
IMDB情感分析專案環境設定腳本
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """檢查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("錯誤: 此專案需要Python 3.8或更高版本")
        print(f"目前版本: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python版本檢查通過: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """安裝依賴包"""
    print("正在安裝依賴包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依賴包安裝完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安裝依賴包失敗: {e}")
        return False

def setup_directories():
    """建立必要的目錄結構"""
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'results',
        'notebooks'
    ]
    
    print("建立目錄結構...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")
    
    return True

def download_nltk_data():
    """下載NLTK數據"""
    print("下載NLTK數據...")
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        print("NLTK數據下載完成！")
        return True
    except Exception as e:
        print(f"NLTK數據下載失敗: {e}")
        return False

def verify_installation():
    """驗證安裝"""
    print("驗證安裝...")
    
    required_packages = [
        'tensorflow',
        'pandas', 
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package}")
    
    if missing_packages:
        print(f"\n缺少套件: {', '.join(missing_packages)}")
        return False
    
    print("所有套件驗證通過！")
    return True

def main():
    """主要安裝流程"""
    print("=== IMDB情感分析專案環境設定 ===")
    print(f"作業系統: {platform.system()} {platform.release()}")
    
    steps = [
        ("檢查Python版本", check_python_version),
        ("建立目錄結構", setup_directories),
        ("安裝依賴包", install_requirements),
        ("下載NLTK數據", download_nltk_data),
        ("驗證安裝", verify_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        if not step_func():
            print(f"❌ {step_name} 失敗，安裝中止")
            return False
    
    print("\n🎉 環境設定完成！")
    print("\n使用說明:")
    print("1. 數據預處理: python main.py preprocess")
    print("2. 訓練模型: python main.py train --model simple_lstm --epochs 5")
    print("3. 預測文本: python predict.py --text '這部電影真的很棒！'")
    print("4. 互動模式: python predict.py --interactive")
    print("5. 探索數據: jupyter notebook notebooks/01_data_exploration.ipynb")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)