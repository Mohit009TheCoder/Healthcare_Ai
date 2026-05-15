"""
Sentiment Analysis Module for Healthcare Reviews and Feedback
Uses NLP to analyze user reviews and enhance recommendations
"""

import re
import sqlite3
from textblob import TextBlob
from collections import Counter
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'healthcare.db')

class HealthcareSentimentAnalyzer:
    """Sentiment analysis for healthcare reviews and feedback"""
    
    def __init__(self):
        # Healthcare-specific positive and negative words
        self.positive_words = {
            'effective', 'helpful', 'excellent', 'amazing', 'wonderful', 'great',
            'good', 'better', 'improved', 'relief', 'cured', 'healed', 'comfortable',
            'satisfied', 'recommend', 'works', 'successful', 'beneficial', 'positive'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worse', 'ineffective',
            'useless', 'painful', 'uncomfortable', 'side effects', 'allergic',
            'reaction', 'disappointed', 'failed', 'dangerous', 'harmful', 'negative'
        }
        
        # Medical condition keywords for context
        self.medical_keywords = {
            'diabetes': ['blood sugar', 'glucose', 'insulin', 'diabetic'],
            'heart': ['chest pain', 'blood pressure', 'cardiac', 'heart rate'],
            'pain': ['headache', 'backache', 'joint pain', 'muscle pain'],
            'infection': ['fever', 'bacteria', 'virus', 'antibiotic']
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a given text"""
        if not text or not isinstance(text, str):
            return {'polarity': 0.0, 'subjectivity': 0.0, 'classification': 'neutral'}
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Use TextBlob for basic sentiment analysis
        blob = TextBlob(cleaned_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Enhance with healthcare-specific analysis
        enhanced_polarity = self._enhance_healthcare_sentiment(cleaned_text, polarity)
        
        # Classify sentiment
        if enhanced_polarity > 0.1:
            classification = 'positive'
        elif enhanced_polarity < -0.1:
            classification = 'negative'
        else:
            classification = 'neutral'
        
        return {
            'polarity': enhanced_polarity,
            'subjectivity': subjectivity,
            'classification': classification,
            'confidence': abs(enhanced_polarity),
            'keywords': self._extract_keywords(cleaned_text)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _enhance_healthcare_sentiment(self, text: str, base_polarity: float) -> float:
        """Enhance sentiment analysis with healthcare-specific context"""
        words = set(text.split())
        
        # Count positive and negative healthcare words
        positive_count = len(words.intersection(self.positive_words))
        negative_count = len(words.intersection(self.negative_words))
        
        # Calculate healthcare sentiment boost
        healthcare_boost = (positive_count - negative_count) * 0.1
        
        # Check for medical context
        medical_context_boost = 0
        for condition, keywords in self.medical_keywords.items():
            if any(keyword in text for keyword in keywords):
                medical_context_boost = 0.05  # Slight boost for medical relevance
                break
        
        # Combine sentiments
        enhanced_polarity = base_polarity + healthcare_boost + medical_context_boost
        
        # Ensure within bounds [-1, 1]
        return max(-1.0, min(1.0, enhanced_polarity))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        words = text.split()
        
        # Filter for meaningful words (length > 3, not common words)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'way', 'she', 'use', 'her', 'now', 'oil', 'sit', 'set'}
        
        keywords = [word for word in words if len(word) > 3 and word not in common_words]
        
        # Return most common keywords
        return [word for word, count in Counter(keywords).most_common(5)]
    
    def analyze_medicine_reviews(self, medicine_name: str) -> Dict:
        """Analyze all reviews for a specific medicine"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all reviews for the medicine
        cursor.execute('''
            SELECT review, rating, created_at
            FROM user_ratings
            WHERE item_id = ? AND item_type = 'medicine' AND review IS NOT NULL
        ''', (medicine_name,))
        
        reviews = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if not reviews:
            return {'overall_sentiment': 'neutral', 'sentiment_score': 0.0, 'review_count': 0}
        
        sentiments = []
        all_keywords = []
        
        for review in reviews:
            sentiment = self.analyze_sentiment(review['review'])
            sentiments.append({
                'sentiment': sentiment,
                'rating': review['rating'],
                'date': review['created_at']
            })
            all_keywords.extend(sentiment['keywords'])
        
        # Calculate overall sentiment
        avg_polarity = np.mean([s['sentiment']['polarity'] for s in sentiments])
        avg_rating = np.mean([s['rating'] for s in sentiments])
        
        # Determine overall classification
        if avg_polarity > 0.1:
            overall_sentiment = 'positive'
        elif avg_polarity < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Get most common keywords
        common_keywords = [word for word, count in Counter(all_keywords).most_common(10)]
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': avg_polarity,
            'average_rating': avg_rating,
            'review_count': len(reviews),
            'common_keywords': common_keywords,
            'sentiment_distribution': {
                'positive': len([s for s in sentiments if s['sentiment']['classification'] == 'positive']),
                'negative': len([s for s in sentiments if s['sentiment']['classification'] == 'negative']),
                'neutral': len([s for s in sentiments if s['sentiment']['classification'] == 'neutral'])
            }
        }
    
    def get_sentiment_enhanced_recommendations(self, user_id: int, recommendations: List[Dict]) -> List[Dict]:
        """Enhance recommendations with sentiment analysis"""
        enhanced_recommendations = []
        
        for rec in recommendations:
            if rec['item_type'] == 'medicine':
                sentiment_analysis = self.analyze_medicine_reviews(rec['item_id'])
                
                # Adjust recommendation score based on sentiment
                sentiment_multiplier = 1.0
                if sentiment_analysis['overall_sentiment'] == 'positive':
                    sentiment_multiplier = 1.2
                elif sentiment_analysis['overall_sentiment'] == 'negative':
                    sentiment_multiplier = 0.8
                
                rec['score'] *= sentiment_multiplier
                rec['sentiment_analysis'] = sentiment_analysis
                rec['sentiment_adjusted'] = True
            
            enhanced_recommendations.append(rec)
        
        # Re-sort by adjusted scores
        enhanced_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return enhanced_recommendations
    
    def analyze_user_feedback_trends(self, days: int = 30) -> Dict:
        """Analyze sentiment trends in user feedback over time"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT review, rating, created_at, item_type
            FROM user_ratings
            WHERE review IS NOT NULL 
            AND created_at > datetime('now', '-{} days')
            ORDER BY created_at
        '''.format(days), )
        
        reviews = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if not reviews:
            return {'trend': 'no_data', 'sentiment_over_time': []}
        
        # Analyze sentiment for each review
        sentiment_data = []
        for review in reviews:
            sentiment = self.analyze_sentiment(review['review'])
            sentiment_data.append({
                'date': review['created_at'][:10],  # Extract date part
                'sentiment_score': sentiment['polarity'],
                'classification': sentiment['classification'],
                'item_type': review['item_type'],
                'rating': review['rating']
            })
        
        # Group by date and calculate daily averages
        df = pd.DataFrame(sentiment_data)
        daily_sentiment = df.groupby('date').agg({
            'sentiment_score': 'mean',
            'rating': 'mean'
        }).reset_index()
        
        # Determine trend
        if len(daily_sentiment) > 1:
            recent_sentiment = daily_sentiment.tail(7)['sentiment_score'].mean()
            earlier_sentiment = daily_sentiment.head(7)['sentiment_score'].mean()
            
            if recent_sentiment > earlier_sentiment + 0.1:
                trend = 'improving'
            elif recent_sentiment < earlier_sentiment - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'overall_sentiment': df['sentiment_score'].mean(),
            'sentiment_over_time': daily_sentiment.to_dict('records'),
            'classification_distribution': df['classification'].value_counts().to_dict()
        }

# Initialize sentiment analyzer
sentiment_analyzer = HealthcareSentimentAnalyzer()