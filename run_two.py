import pandas as pd
import numpy as np
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
import torch
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import sqlite3
from datetime import datetime
import joblib
from torch.nn.functional import softmax
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import plotly.graph_objects as go

import plotly.express as px

from tqdm import tqdm


import pandas as pd
import numpy as np
from transformers import pipeline
import torch
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

class AdvancedSentimentAnalyzer:
    def __init__(self, use_gpu=False):
        """Initialize the advanced sentiment analyzer with multiple models"""
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.setup_models()  # Removed self.setup_database() from here
        
    def setup_models(self):
        """Initialize all required models"""
        # Main sentiment analysis models
        self.models = {
            'roberta': pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            ),
            'bert': pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=self.device
            ),
        }
        
        # Enhanced emotion detection models
        self.emotion_models = {
            'go_emotions': pipeline(
                "text-classification",
                model="SamLowe/roberta-base-go_emotions",
                top_k=None,
                device=self.device
            )
        }

    def analyze_text(self, text):
        """Perform emotion analysis on text"""
        # Split text into sentences for granular analysis
        sentences = sent_tokenize(text)
        
        # Analyze each sentence
        sentence_emotions = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 0:
                # Analyze emotions for each sentence
                emotions = self._analyze_emotions(sentence)
                sentence_emotions.append({
                    'sentence': sentence,
                    'emotions': emotions
                })
        
        # Get emotion analysis for whole text
        emotions = self._analyze_emotions(text)
        
        # Store results
        result = {
            'text': text,
            'sentence_emotions': sentence_emotions,
            'emotions': emotions,
        }
        
        return result

    def _analyze_emotions(self, text):
        """Enhanced emotion analysis using Go Emotions model"""
        try:
            # Get emotions from Go Emotions model
            go_emotions = self.emotion_models['go_emotions']([text])[0]
            
            # Filter significant emotions (score > 0.1)
            significant_emotions = {
                item['label']: item['score'] 
                for item in go_emotions
                if item['score'] > 0.1
            }
            
            return significant_emotions
            
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            return {}

def main():
    # Initialize analyzer
    analyzer = AdvancedSentimentAnalyzer()
    
    # Get input from user
    text = input("Enter your text to analyze emotions: ")
    
    # Analyze text
    result = analyzer.analyze_text(text)
    
    # Print results
    print("\nEmotion Analysis Results:")
    
    print("\nOverall Text Emotions:")
    for emotion, score in result['emotions'].items():
        print(f"{emotion}: {score:.3f}")
    
    print("\nSentence-by-Sentence Analysis:")
    for item in result['sentence_emotions']:
        print(f"\nSentence: {item['sentence']}")
        print("Emotions detected:")
        for emotion, score in item['emotions'].items():
            print(f"- {emotion}: {score:.3f}")

if __name__ == "__main__":
    main()

    

    