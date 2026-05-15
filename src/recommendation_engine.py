"""
Advanced Healthcare Recommendation Engine
Combines content-based, collaborative, and hybrid filtering with AI/ML models
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import json
import os
from typing import List, Dict, Tuple, Optional

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'healthcare.db')

class HealthcareRecommendationEngine:
    """Advanced recommendation engine for healthcare system"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        self.scaler = StandardScaler()
        self.user_profiles = {}
        self.item_features = {}
        self.interaction_matrix = None
        
    def init_recommendation_tables(self):
        """Initialize recommendation-specific database tables"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # User preferences and interests
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # User activity tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                activity_type TEXT NOT NULL,
                item_id TEXT,
                item_type TEXT,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # User ratings and feedback
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                rating REAL NOT NULL CHECK(rating >= 1 AND rating <= 5),
                review TEXT,
                sentiment_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Recommendation history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                recommendation_type TEXT NOT NULL,
                score REAL NOT NULL,
                context TEXT,
                shown_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                clicked BOOLEAN DEFAULT 0,
                clicked_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Trending items
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trending_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                trend_score REAL NOT NULL,
                category TEXT,
                time_window TEXT NOT NULL,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def track_user_activity(self, user_id: int, activity_type: str, 
                           item_id: str = None, item_type: str = None, 
                           metadata: dict = None, session_id: str = None):
        """Track user activity for behavioral analysis"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO user_activities (user_id, activity_type, item_id, item_type, metadata, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, activity_type, item_id, item_type, metadata_json, session_id))
        
        conn.commit()
        conn.close()
        
    def update_user_preferences(self, user_id: int, preferences: Dict[str, any]):
        """Update user preferences and interests"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for category, value in preferences.items():
            # Check if preference exists
            cursor.execute('''
                SELECT id FROM user_preferences 
                WHERE user_id = ? AND category = ?
            ''', (user_id, category))
            
            if cursor.fetchone():
                # Update existing preference
                cursor.execute('''
                    UPDATE user_preferences 
                    SET preference_value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND category = ?
                ''', (str(value), user_id, category))
            else:
                # Insert new preference
                cursor.execute('''
                    INSERT INTO user_preferences (user_id, category, preference_value)
                    VALUES (?, ?, ?)
                ''', (user_id, category, str(value)))
        
        conn.commit()
        conn.close()
        
    def get_user_profile(self, user_id: int) -> Dict:
        """Get comprehensive user profile for recommendations"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get basic user info
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = dict(cursor.fetchone())
        
        # Get preferences
        cursor.execute('''
            SELECT category, preference_value, weight 
            FROM user_preferences WHERE user_id = ?
        ''', (user_id,))
        preferences = {row['category']: {'value': row['preference_value'], 'weight': row['weight']} 
                      for row in cursor.fetchall()}
        
        # Get recent activities
        cursor.execute('''
            SELECT activity_type, item_id, item_type, COUNT(*) as count
            FROM user_activities 
            WHERE user_id = ? AND timestamp > datetime('now', '-30 days')
            GROUP BY activity_type, item_id, item_type
            ORDER BY count DESC
            LIMIT 50
        ''', (user_id,))
        activities = [dict(row) for row in cursor.fetchall()]
        
        # Get ratings
        cursor.execute('''
            SELECT item_id, item_type, rating, sentiment_score
            FROM user_ratings WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        ratings = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'user': user,
            'preferences': preferences,
            'activities': activities,
            'ratings': ratings
        }
    def content_based_filtering(self, user_id: int, item_type: str = 'medicine', 
                               top_k: int = 10) -> List[Dict]:
        """Content-based filtering recommendations"""
        user_profile = self.get_user_profile(user_id)
        
        if item_type == 'medicine':
            return self._recommend_medicines_content_based(user_profile, top_k)
        elif item_type == 'treatment':
            # Return empty list for treatments for now
            return []
        else:
            return []
    
    def _recommend_medicines_content_based(self, user_profile: Dict, top_k: int) -> List[Dict]:
        """Recommend medicines based on user's medical history and preferences"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get user's medical conditions from records
        user_conditions = []
        cursor.execute('''
            SELECT DISTINCT record_type, prediction_result 
            FROM medical_records 
            WHERE patient_id = ?
        ''', (user_profile['user']['id'],))
        
        for record in cursor.fetchall():
            if 'positive' in record['prediction_result'].lower() or 'diabetes' in record['prediction_result'].lower():
                user_conditions.append(record['record_type'])
        
        # Get medicines for similar conditions
        recommendations = []
        
        # Load drug data
        drug_df_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'drug_review.csv')
        if os.path.exists(drug_df_path):
            drug_df = pd.read_csv(drug_df_path)
            
            # Filter medicines based on user conditions
            relevant_medicines = drug_df[drug_df['disease'].str.lower().isin([c.lower() for c in user_conditions])]
            
            if not relevant_medicines.empty:
                # Calculate content-based scores
                top_medicines = relevant_medicines.groupby('medicine_name').agg({
                    'rating': 'mean',
                    'effectiveness': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
                    'side_effects': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
                    'disease': 'first'
                }).reset_index()
                
                top_medicines = top_medicines.sort_values('rating', ascending=False).head(top_k)
                
                for _, medicine in top_medicines.iterrows():
                    recommendations.append({
                        'item_id': medicine['medicine_name'],
                        'item_type': 'medicine',
                        'name': medicine['medicine_name'],
                        'score': float(medicine['rating']) / 5.0,  # Normalize to 0-1
                        'reason': f"Recommended for {medicine['disease']}",
                        'effectiveness': medicine['effectiveness'],
                        'side_effects': medicine['side_effects'],
                        'recommendation_type': 'content_based'
                    })
        
        conn.close()
        return recommendations
    
    def _recommend_treatments_content_based(self, user_profile: Dict, top_k: int) -> List[Dict]:
        """Recommend treatments based on user's medical history and conditions"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get user's medical conditions from records
        user_conditions = []
        cursor.execute('''
            SELECT DISTINCT record_type, prediction_result 
            FROM medical_records 
            WHERE patient_id = ?
        ''', (user_profile['user']['id'],))
        
        for record in cursor.fetchall():
            if 'positive' in record['prediction_result'].lower() or 'diabetes' in record['prediction_result'].lower():
                user_conditions.append(record['record_type'])
        
        # Generate treatment recommendations based on conditions
        recommendations = []
        
        # Map conditions to treatments
        treatment_map = {
            'diabetes': [
                {'name': 'Insulin Therapy', 'description': 'Daily insulin injections', 'effectiveness': 'High'},
                {'name': 'Oral Medications', 'description': 'Metformin and other oral agents', 'effectiveness': 'High'},
                {'name': 'Lifestyle Modification', 'description': 'Diet and exercise program', 'effectiveness': 'Medium'},
            ],
            'heart': [
                {'name': 'Cardiac Rehabilitation', 'description': 'Supervised exercise program', 'effectiveness': 'High'},
                {'name': 'Medication Management', 'description': 'ACE inhibitors and beta-blockers', 'effectiveness': 'High'},
                {'name': 'Dietary Changes', 'description': 'Low sodium, heart-healthy diet', 'effectiveness': 'Medium'},
            ],
            'symptom': [
                {'name': 'Symptomatic Treatment', 'description': 'Address specific symptoms', 'effectiveness': 'Medium'},
                {'name': 'Preventive Care', 'description': 'Vaccination and screening', 'effectiveness': 'High'},
                {'name': 'Monitoring', 'description': 'Regular health check-ups', 'effectiveness': 'Medium'},
            ]
        }
        
        # Get treatments for user's conditions
        for condition in user_conditions:
            condition_key = condition.lower().replace('_', '').replace(' ', '')
            for key, treatments in treatment_map.items():
                if key in condition_key:
                    for treatment in treatments[:top_k]:
                        recommendations.append({
                            'item_id': treatment['name'],
                            'item_type': 'treatment',
                            'name': treatment['name'],
                            'description': treatment['description'],
                            'score': 0.8 if treatment['effectiveness'] == 'High' else 0.6,
                            'reason': f"Recommended for {condition}",
                            'effectiveness': treatment['effectiveness'],
                            'recommendation_type': 'content_based'
                        })
        
        # If no specific treatments found, return general recommendations
        if not recommendations:
            recommendations = [
                {
                    'item_id': 'General Checkup',
                    'item_type': 'treatment',
                    'name': 'General Checkup',
                    'description': 'Comprehensive health assessment',
                    'score': 0.7,
                    'reason': 'Recommended for overall health monitoring',
                    'effectiveness': 'Medium',
                    'recommendation_type': 'content_based'
                }
            ]
        
        conn.close()
        return recommendations[:top_k]
    
    def collaborative_filtering(self, user_id: int, item_type: str = 'medicine', 
                               top_k: int = 10) -> List[Dict]:
        """Collaborative filtering using user-item interactions"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get user-item interaction matrix
        cursor.execute('''
            SELECT user_id, item_id, rating
            FROM user_ratings
            WHERE item_type = ?
        ''', (item_type,))
        
        ratings_data = [dict(row) for row in cursor.fetchall()]
        
        if len(ratings_data) < 10:  # Not enough data for collaborative filtering
            return []
        
        # Create user-item matrix
        ratings_df = pd.DataFrame(ratings_data)
        user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        
        if user_id not in user_item_matrix.index:
            return []
        
        # Calculate user similarity using cosine similarity
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, 
                                         index=user_item_matrix.index, 
                                         columns=user_item_matrix.index)
        
        # Find similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]  # Top 5 similar users
        
        recommendations = []
        user_rated_items = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
        
        for similar_user_id, similarity_score in similar_users.items():
            if similarity_score > 0.1:  # Minimum similarity threshold
                similar_user_items = user_item_matrix.loc[similar_user_id]
                
                for item_id, rating in similar_user_items.items():
                    if rating > 3.0 and item_id not in user_rated_items:  # Good rating and not already rated
                        score = rating * similarity_score / 5.0  # Normalize
                        
                        recommendations.append({
                            'item_id': item_id,
                            'item_type': item_type,
                            'name': item_id,
                            'score': score,
                            'reason': f"Users similar to you rated this {rating:.1f}/5",
                            'recommendation_type': 'collaborative'
                        })
        
        # Remove duplicates and sort by score
        seen = set()
        unique_recommendations = []
        for rec in sorted(recommendations, key=lambda x: x['score'], reverse=True):
            if rec['item_id'] not in seen:
                seen.add(rec['item_id'])
                unique_recommendations.append(rec)
        
        conn.close()
        return unique_recommendations[:top_k]
    
    def hybrid_recommendation(self, user_id: int, item_type: str = 'medicine', 
                             top_k: int = 10, content_weight: float = 0.6, 
                             collab_weight: float = 0.4) -> List[Dict]:
        """Hybrid recommendation combining content-based and collaborative filtering"""
        
        # Get recommendations from both methods
        content_recs = self.content_based_filtering(user_id, item_type, top_k * 2)
        collab_recs = self.collaborative_filtering(user_id, item_type, top_k * 2)
        
        # Combine and weight the recommendations
        hybrid_scores = {}
        
        # Add content-based scores
        for rec in content_recs:
            item_id = rec['item_id']
            hybrid_scores[item_id] = {
                'content_score': rec['score'] * content_weight,
                'collab_score': 0,
                'item_data': rec
            }
        
        # Add collaborative scores
        for rec in collab_recs:
            item_id = rec['item_id']
            if item_id in hybrid_scores:
                hybrid_scores[item_id]['collab_score'] = rec['score'] * collab_weight
            else:
                hybrid_scores[item_id] = {
                    'content_score': 0,
                    'collab_score': rec['score'] * collab_weight,
                    'item_data': rec
                }
        
        # Calculate final hybrid scores
        final_recommendations = []
        for item_id, scores in hybrid_scores.items():
            final_score = scores['content_score'] + scores['collab_score']
            
            rec = scores['item_data'].copy()
            rec['score'] = final_score
            rec['recommendation_type'] = 'hybrid'
            rec['content_score'] = scores['content_score']
            rec['collab_score'] = scores['collab_score']
            
            final_recommendations.append(rec)
        
        # Sort by final score and return top k
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return final_recommendations[:top_k]
    
    def context_aware_recommendations(self, user_id: int, context: Dict, 
                                    item_type: str = 'medicine', top_k: int = 10) -> List[Dict]:
        """Context-aware recommendations based on time, location, trends, etc."""
        
        # Get base hybrid recommendations
        base_recs = self.hybrid_recommendation(user_id, item_type, top_k * 2)
        
        # Apply contextual adjustments
        contextual_recs = []
        
        for rec in base_recs:
            adjusted_score = rec['score']
            
            # Time-based adjustments
            if 'time_of_day' in context:
                if context['time_of_day'] == 'morning' and 'morning' in rec.get('name', '').lower():
                    adjusted_score *= 1.2
                elif context['time_of_day'] == 'evening' and 'evening' in rec.get('name', '').lower():
                    adjusted_score *= 1.2
            
            # Urgency adjustments
            if 'urgency' in context:
                if context['urgency'] == 'high' and rec.get('effectiveness') == 'Highly Effective':
                    adjusted_score *= 1.3
            
            # Trending adjustments
            trending_boost = self._get_trending_boost(rec['item_id'], item_type)
            adjusted_score *= (1 + trending_boost)
            
            rec['score'] = adjusted_score
            rec['context_adjustments'] = {
                'trending_boost': trending_boost,
                'original_score': rec['score'] / (1 + trending_boost)
            }
            
            contextual_recs.append(rec)
        
        # Re-sort and return top k
        contextual_recs.sort(key=lambda x: x['score'], reverse=True)
        return contextual_recs[:top_k]
    
    def _get_trending_boost(self, item_id: str, item_type: str) -> float:
        """Get trending boost factor for an item"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT trend_score FROM trending_items
            WHERE item_id = ? AND item_type = ? AND time_window = 'daily'
            ORDER BY calculated_at DESC LIMIT 1
        ''', (item_id, item_type))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return min(result[0] / 100.0, 0.5)  # Max 50% boost
        return 0.0
    
    def update_trending_items(self):
        """Update trending items based on recent user activities"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Calculate daily trends
        cursor.execute('''
            SELECT item_id, item_type, COUNT(*) as activity_count,
                   COUNT(DISTINCT user_id) as unique_users
            FROM user_activities
            WHERE timestamp > datetime('now', '-1 day')
            AND item_id IS NOT NULL
            GROUP BY item_id, item_type
            HAVING activity_count >= 3
        ''')
        
        daily_trends = cursor.fetchall()
        
        # Clear old daily trends
        cursor.execute('''
            DELETE FROM trending_items WHERE time_window = 'daily'
        ''')
        
        # Insert new trends
        for item_id, item_type, activity_count, unique_users in daily_trends:
            trend_score = (activity_count * 0.7) + (unique_users * 0.3)
            
            cursor.execute('''
                INSERT INTO trending_items (item_id, item_type, trend_score, time_window)
                VALUES (?, ?, ?, 'daily')
            ''', (item_id, item_type, trend_score))
        
        conn.commit()
        conn.close()
    
    def get_personalized_recommendations(self, user_id: int, context: Dict = None, 
                                       top_k: int = 10) -> Dict[str, List[Dict]]:
        """Get comprehensive personalized recommendations"""
        
        if context is None:
            context = {'time_of_day': 'general', 'urgency': 'normal'}
        
        recommendations = {
            'medicines': self.context_aware_recommendations(user_id, context, 'medicine', top_k),
            'treatments': self.context_aware_recommendations(user_id, context, 'treatment', top_k//2),
            'specialists': self._recommend_specialists(user_id, top_k//2)
        }
        
        # Log recommendations
        self._log_recommendations(user_id, recommendations)
        
        return recommendations
    
    def _recommend_specialists(self, user_id: int, top_k: int) -> List[Dict]:
        """Recommend specialist doctors based on user's medical history"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get user's medical conditions
        cursor.execute('''
            SELECT record_type, prediction_result, confidence
            FROM medical_records
            WHERE patient_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        medical_records = [dict(row) for row in cursor.fetchall()]
        
        # Map conditions to specializations
        condition_specialization_map = {
            'diabetes': ['Endocrinologist', 'Internal Medicine'],
            'heart': ['Cardiologist', 'Internal Medicine'],
            'symptom': ['General Practitioner', 'Internal Medicine']
        }
        
        recommended_specializations = set()
        for record in medical_records:
            if record['record_type'] in condition_specialization_map:
                recommended_specializations.update(condition_specialization_map[record['record_type']])
        
        if not recommended_specializations:
            recommended_specializations = ['General Practitioner']
        
        # Get doctors with these specializations
        placeholders = ','.join(['?' for _ in recommended_specializations])
        cursor.execute(f'''
            SELECT u.id, u.full_name, u.email, u.phone,
                   d.specialization, d.years_experience, d.qualification
            FROM users u
            JOIN doctor_profiles d ON u.id = d.user_id
            WHERE u.role = 'doctor' AND u.is_active = 1
            AND d.specialization IN ({placeholders})
            ORDER BY d.years_experience DESC
        ''', list(recommended_specializations))
        
        specialists = []
        for row in cursor.fetchall():
            specialists.append({
                'item_id': str(row['id']),
                'item_type': 'specialist',
                'name': row['full_name'],
                'specialization': row['specialization'],
                'experience': row['years_experience'],
                'qualification': row['qualification'],
                'score': min(1.0, row['years_experience'] / 20.0),  # Normalize experience
                'reason': f"Specialist in {row['specialization']}",
                'recommendation_type': 'specialist'
            })
        
        conn.close()
        return specialists[:top_k]
    
    def _log_recommendations(self, user_id: int, recommendations: Dict):
        """Log recommendations for tracking and analysis"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for category, recs in recommendations.items():
            for rec in recs:
                cursor.execute('''
                    INSERT INTO recommendation_history 
                    (user_id, item_id, item_type, recommendation_type, score, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, rec['item_id'], rec['item_type'], 
                     rec['recommendation_type'], rec['score'], 
                     json.dumps({'category': category})))
        
        conn.commit()
        conn.close()

# Initialize the recommendation engine
recommendation_engine = HealthcareRecommendationEngine()
recommendation_engine.init_recommendation_tables()