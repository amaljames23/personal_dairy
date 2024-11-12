import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import re
import networkx as nx

class AdvancedTextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """Clean and preprocess the input text."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s.]', '', text.lower())
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            # Tokenize words
            words = word_tokenize(sentence)
            # Remove stopwords and lemmatize
            words = [self.lemmatizer.lemmatize(word) for word in words 
                    if word not in self.stop_words and len(word) > 2]
            cleaned_sentences.append(' '.join(words))
            
        return sentences, cleaned_sentences
    
    def calculate_sentence_scores(self, sentences, cleaned_sentences):
        """Calculate importance scores for sentences using multiple methods."""
        
        # Method 1: TF-IDF based scoring
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
        tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=1)
        
        # Method 2: Position based scoring
        position_scores = [1.0 / (i + 1) for i in range(len(sentences))]
        
        # Method 3: TextRank (graph-based scoring)
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        nx_graph = nx.from_numpy_array(similarity_matrix)
        textrank_scores = nx.pagerank(nx_graph)
        
        # Method 4: Length based scoring (longer sentences often contain more information)
        length_scores = [len(word_tokenize(sent)) / 100.0 for sent in sentences]
        
        # Combine all scores with weights
        final_scores = []
        for i in range(len(sentences)):
            score = (0.4 * tfidf_scores[i] + 
                    0.2 * position_scores[i] + 
                    0.3 * textrank_scores[i] + 
                    0.1 * length_scores[i])
            final_scores.append((i, score))
            
        return final_scores
    
    def generate_summary(self, text, compression_ratio=0.3):
        """Generate a summary of the input text."""
        # Preprocess text
        original_sentences, cleaned_sentences = self.preprocess_text(text)
        
        if not original_sentences:
            return ""
            
        # Calculate sentence scores
        sentence_scores = self.calculate_sentence_scores(original_sentences, cleaned_sentences)
        
        # Sort sentences by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences based on compression ratio
        num_sentences = max(1, int(len(original_sentences) * compression_ratio))
        selected_sentences = sorted(sentence_scores[:num_sentences], key=lambda x: x[0])
        
        # Reconstruct summary
        summary = ' '.join([original_sentences[idx] for idx, _ in selected_sentences])
        
        return summary

    def get_key_phrases(self, text, num_phrases=5):
        """Extract key phrases from the text."""
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        # Create word pairs
        pairs = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        
        # Count frequencies
        pair_freq = defaultdict(int)
        for pair in pairs:
            pair_freq[pair] += 1
            
        # Sort by frequency
        sorted_pairs = sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [pair for pair, _ in sorted_pairs[:num_phrases]]

def main():
    # Example usage
    text = """
    Today was a day that truly filled my heart with joy. The morning started on such a positive note as I woke up to the gentle warmth of sunlight streaming through my window, promising a beautiful day ahead. After a refreshing breakfast, I decided to take a walk in the nearby park, and I could feel every step was lighter, almost as if my feet weren't even touching the ground. The trees swayed in the light breeze, and it seemed like they were dancing just for me. I stumbled upon a group of children playing near the swings, their laughter ringing out, pure and contagious. Seeing them so carefree, I felt that same youthful energy bubbling up within me. On my way back, I ran into an old friend I hadn't seen in years, and we spent an hour catching up on life, reminiscing about our shared memories. It's moments like these, I realized, that make life feel truly rich and meaningful. By evening, I was with family, sharing stories over a hearty meal, and it reminded me of how lucky I am to have people who care for me. I felt a deep sense of gratitude that seemed to settle in my bones, making me feel grounded and content. Life felt vibrant today, each moment like a piece of a puzzle coming together in a picture of perfect happiness.
    """
    
    summarizer = AdvancedTextSummarizer()
    
    # Generate summary
    summary = summarizer.generate_summary(text, compression_ratio=0.3)
    
    # Extract key phrases
    key_phrases = summarizer.get_key_phrases(text)
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Summary and Key Phrases</title>
    </head>
    <body>
        <h1>Text Summary</h1>
        <p>{summary}</p>
        <h2>Key Phrases</h2>
        <ul>
            {''.join(f'<li>{phrase}</li>' for phrase in key_phrases)}
        </ul>
    </body>
    </html>
    """
    
    # Save the HTML to a file
    with open("text_summary_one.html", "w", encoding="utf-8") as file:
        file.write(html_content)
    
    print("HTML file generated: text_summary.html")

if __name__ == "__main__":
    main()
