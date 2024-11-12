import pandas as pd
import numpy as np
from transformers import pipeline
import torch
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm


nltk.download('punkt')

class AdvancedSentimentAnalyzer:
    def __init__(self, use_gpu=False):
        """Initialize the advanced sentiment analyzer with multiple models"""
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.setup_models()
        self.diary_entries = []
        
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

    def analyze_text(self, text, date=None):
        """Perform emotion analysis on text"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
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
            'date': date,
            'text': text,
            'sentence_emotions': sentence_emotions,
            'emotions': emotions,
        }
        
        # Add to diary entries
        self.diary_entries.append(result)
        
        return result

    def analyze_multiple_entries(self, entries):
        """Analyze multiple diary entries
        entries: list of tuples (text, date)"""
        results = []
        for text, date in tqdm(entries, desc="Analyzing entries"):
            result = self.analyze_text(text, date)
            results.append(result)
        return results

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

    def plot_emotion_trends(self):
        """Plot emotion trends over time using plotly"""
        if not self.diary_entries:
            print("No diary entries to plot")
            return

        # Prepare data for plotting
        dates = []
        emotions_data = {}
        
        for entry in self.diary_entries:
            dates.append(entry['date'])
            for emotion, score in entry['emotions'].items():
                if emotion not in emotions_data:
                    emotions_data[emotion] = []
                emotions_data[emotion].append(score)

        # Create figure
        fig = go.Figure()

        # Add traces for each emotion
        for emotion, scores in emotions_data.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=scores,
                name=emotion,
                mode='lines+markers'
            ))

        # Update layout
        fig.update_layout(
            title='Emotion Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Emotion Intensity',
            hovermode='x unified',
            showlegend=True
        )

        return fig

    def plot_emotion_heatmap(self):
        """Create a heatmap of emotions over time"""
        if not self.diary_entries:
            print("No diary entries to plot")
            return

        # Prepare data for heatmap
        dates = []
        emotions_set = set()
        emotions_data = {}
        
        for entry in self.diary_entries:
            dates.append(entry['date'])
            emotions_set.update(entry['emotions'].keys())
            
        emotions_list = sorted(list(emotions_set))
        
        # Create matrix for heatmap
        matrix = []
        for emotion in emotions_list:
            row = []
            for entry in self.diary_entries:
                row.append(entry['emotions'].get(emotion, 0))
            matrix.append(row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=dates,
            y=emotions_list,
            colorscale='Viridis'
        ))

        fig.update_layout(
            title='Emotion Intensity Heatmap',
            xaxis_title='Date',
            yaxis_title='Emotion',
        )

        return fig

def main():
    # Initialize analyzer
    analyzer = AdvancedSentimentAnalyzer()
    
    # Example of multiple entries
    diary_entries = [
        ("Today was a wonderful day. I achieved all my goals and felt really productive.", "2024-01-01"),
        ("I felt a bit anxious about the upcoming presentation, but managed to stay focused.", "2024-01-02"),
        ("Had a great time with friends, though work was challenging.", "2024-01-03"),
        ("Spent the day with family. It was relaxing and much needed after a busy week.", "2024-01-04"),
        ("Worked late to finish a project. Feeling exhausted but proud of the results.", "2024-01-05"),
        ("Went for a long walk to clear my mind. The fresh air really helped me feel more positive.", "2024-01-06"),
        ("Struggled to concentrate today, kept getting distracted. Will try to be more organized tomorrow.", "2024-01-07"),
        ("Received positive feedback on my work, which lifted my mood. Feeling appreciated.", "2024-01-08"),
        ("Today was frustrating. Encountered several setbacks at work, but I'm determined to fix them.", "2024-01-09"),
        ("Relaxed with a good book in the evening. It was nice to unwind and escape into a story.", "2024-01-10"),
        ("Met up with an old friend. Catching up was so refreshing, and it brightened my day.", "2024-01-11"),
    ]
    
    # Analyze all entries
    analyzer.analyze_multiple_entries(diary_entries)
    
    # Create and show plots
    trend_fig = analyzer.plot_emotion_trends()
    trend_fig.show()
    
    heatmap_fig = analyzer.plot_emotion_heatmap()
    heatmap_fig.show()

if __name__ == "__main__":
    main()