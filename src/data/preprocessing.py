"""
文本預處理模組

提供IMDB電影評論的文本清理、標準化和特徵提取功能。
包含HTML清理、否定詞處理、分詞等核心預處理步驟。
"""

import re
import string
from typing import List, Optional, Union
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy


class TextPreprocessor:
    """
    文本預處理器
    
    提供完整的文本清理和標準化功能，支援不同的預處理策略。
    """
    
    def __init__(
        self, 
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_punctuation: bool = False,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        handle_negations: bool = True,
        stemming: bool = False,
        lemmatization: bool = False,
        min_length: int = 2
    ):
        """
        初始化預處理器
        
        Args:
            remove_html: 是否移除HTML標籤
            remove_urls: 是否移除URL
            remove_punctuation: 是否移除標點符號
            lowercase: 是否轉換為小寫
            remove_stopwords: 是否移除停用詞
            handle_negations: 是否處理否定詞
            stemming: 是否進行詞幹提取
            lemmatization: 是否進行詞形還原
            min_length: 最小詞長度
        """
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.handle_negations = handle_negations
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.min_length = min_length
        
        # 初始化NLTK元件
        self._initialize_nltk()
        
        # 初始化詞幹提取器和詞形還原器
        if self.stemming:
            self.stemmer = PorterStemmer()
        if self.lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        
        # 否定詞列表
        self.negation_words = {
            "not", "no", "never", "neither", "nowhere", "nothing",
            "none", "nobody", "isn't", "aren't", "wasn't", "weren't",
            "don't", "doesn't", "didn't", "won't", "wouldn't", "can't",
            "couldn't", "shouldn't", "mustn't", "needn't", "daren't",
            "hasn't", "haven't", "hadn't"
        }
    
    def _initialize_nltk(self):
        """初始化NLTK資源"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        if self.remove_stopwords:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        if self.lemmatization:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
    
    def clean_html(self, text: str) -> str:
        """
        移除HTML標籤
        
        Args:
            text: 輸入文本
            
        Returns:
            清理後的文本
        """
        if not self.remove_html:
            return text
        
        # 使用BeautifulSoup移除HTML標籤
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    
    def clean_urls(self, text: str) -> str:
        """
        移除URL
        
        Args:
            text: 輸入文本
            
        Returns:
            清理後的文本
        """
        if not self.remove_urls:
            return text
        
        # 移除URL
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = url_pattern.sub('', text)
        
        # 移除www開頭的網址
        www_pattern = re.compile(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = www_pattern.sub('', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        標準化空白字符
        
        Args:
            text: 輸入文本
            
        Returns:
            標準化後的文本
        """
        # 將多個空白字符替換為單個空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def handle_negation_words(self, text: str) -> str:
        """
        處理否定詞
        
        Args:
            text: 輸入文本
            
        Returns:
            處理後的文本
        """
        if not self.handle_negations:
            return text
        
        words = text.split()
        processed_words = []
        
        i = 0
        while i < len(words):
            word = words[i].lower()
            
            # 檢查是否為否定詞
            if word in self.negation_words:
                processed_words.append(words[i])
                # 標記下一個詞
                if i + 1 < len(words):
                    i += 1
                    processed_words.append(f"NOT_{words[i]}")
                i += 1
            else:
                processed_words.append(words[i])
                i += 1
        
        return " ".join(processed_words)
    
    def remove_punctuation_func(self, text: str) -> str:
        """
        移除標點符號
        
        Args:
            text: 輸入文本
            
        Returns:
            移除標點符號後的文本
        """
        if not self.remove_punctuation:
            return text
        
        # 保留某些重要的標點符號用於否定處理
        translator = str.maketrans('', '', string.punctuation.replace("'", ""))
        return text.translate(translator)
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        分詞和過濾
        
        Args:
            text: 輸入文本
            
        Returns:
            處理後的詞彙列表
        """
        # 分詞
        tokens = word_tokenize(text)
        
        # 過濾條件
        filtered_tokens = []
        for token in tokens:
            # 長度過濾
            if len(token) < self.min_length:
                continue
            
            # 停用詞過濾
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            
            # 詞幹提取
            if self.stemming:
                token = self.stemmer.stem(token)
            
            # 詞形還原
            if self.lemmatization:
                token = self.lemmatizer.lemmatize(token)
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def preprocess(self, text: str) -> str:
        """
        完整的文本預處理流程
        
        Args:
            text: 輸入文本
            
        Returns:
            預處理後的文本
        """
        # 1. HTML清理
        text = self.clean_html(text)
        
        # 2. URL清理
        text = self.clean_urls(text)
        
        # 3. 標準化空白字符
        text = self.normalize_whitespace(text)
        
        # 4. 否定詞處理（在轉小寫之前）
        text = self.handle_negation_words(text)
        
        # 5. 轉小寫
        if self.lowercase:
            text = text.lower()
        
        # 6. 移除標點符號
        text = self.remove_punctuation_func(text)
        
        # 7. 分詞和過濾
        if self.remove_stopwords or self.stemming or self.lemmatization:
            tokens = self.tokenize_and_filter(text)
            text = " ".join(tokens)
        
        # 8. 最終清理
        text = self.normalize_whitespace(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        批次預處理
        
        Args:
            texts: 文本列表
            
        Returns:
            預處理後的文本列表
        """
        return [self.preprocess(text) for text in texts]


class AdvancedTextPreprocessor(TextPreprocessor):
    """
    進階文本預處理器
    
    使用spaCy進行更精確的語言處理。
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm", **kwargs):
        """
        初始化進階預處理器
        
        Args:
            spacy_model: spaCy模型名稱
            **kwargs: 父類別參數
        """
        super().__init__(**kwargs)
        
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"警告：無法載入spaCy模型 {spacy_model}，回退到基礎預處理")
            self.nlp = None
    
    def advanced_preprocess(self, text: str) -> str:
        """
        使用spaCy進行進階預處理
        
        Args:
            text: 輸入文本
            
        Returns:
            預處理後的文本
        """
        if self.nlp is None:
            return self.preprocess(text)
        
        # 基礎清理
        text = self.clean_html(text)
        text = self.clean_urls(text)
        text = self.normalize_whitespace(text)
        
        # 使用spaCy處理
        doc = self.nlp(text)
        
        processed_tokens = []
        for token in doc:
            # 跳過標點符號和空白
            if token.is_punct or token.is_space:
                continue
            
            # 長度過濾
            if len(token.text) < self.min_length:
                continue
            
            # 停用詞過濾
            if self.remove_stopwords and token.is_stop:
                continue
            
            # 選擇詞形還原或原始形式
            word = token.lemma_ if self.lemmatization else token.text
            
            # 轉小寫
            if self.lowercase:
                word = word.lower()
            
            processed_tokens.append(word)
        
        return " ".join(processed_tokens)