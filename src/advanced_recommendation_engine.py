"""
Advanced Personalized Healthcare Recommendation Engine
Implements multiple recommendation strategies:
- Content-Based Filtering (TF-IDF + Cosine Similarity)
- Collaborative Filtering (SVD-based)
- Hybrid Filtering (Weighted combination)
- Context-Aware Recommendations (Time, Location, Trends)
- Knowledge Graph-Based Recommendations
- Multi-Armed Bandit for Exploration/Exploitation
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

import os
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'healthcare.db')


class AdvancedRecommendationEngine:
    """Advanced recommendation engine with multiple filtering strategies"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        self.scaler = StandardScaler()
        self.knn_model = None
        self.knowledge_graph = {}
        self.bandit_arms = {}
        self.init_tables()
    
    def init_tables(self):
        """Initialize advanced recommendation tables"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Knowledge graph for disease-symptom-treatment relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT NOT NULL,
                source_type TEXT NOT NULL CHECK(source_type IN ('disease', 'symptom', 'medicine', 'treatment')),
                relationship TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                target_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Multi-armed bandit tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bandit_arms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                arm_name TEXT NOT NULL,
                pulls INTEGER DEFAULT 0,
                rewards REAL DEFAULT 0.0,
                avg_reward REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, arm_name)
            )
        ''')
        
        # Explainability tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendation_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_id INTEGER NOT NULL,
                explanation_type TEXT NOT NULL,
                explanation_text TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (recommendation_id) REFERENCES recommendation_history(id)
            )
        ''')
        
        # User segments for cohort analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                segment_name TEXT NOT NULL,
                segment_type TEXT NOT NULL,
                segment_value TEXT,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # A/B testing framework
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL UNIQUE,
                variant_a TEXT NOT NULL,
                variant_b TEXT NOT NULL,
                start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_date TIMESTAMP,
                status TEXT DEFAULT 'active',
                winner TEXT
            )
        ''')
        
        # A/B test results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                variant TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES ab_tests(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. CONTENT-BASED FILTERING
    # ─────────────────────────────────────────────────────────────────────────
    
    def content_based_filtering(self, user_id: int, item_type: str = 'medicine', 
                               top_k: int = 5) -> List[Dict]:
        """
        Content-based filtering using TF-IDF and cosine similarity
        Recommends items similar to user's past interactions
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get user's interaction history
            cursor.execute('''
                SELECT DISTINCT item_id, item_type, metadata
                FROM user_activities
                WHERE user_id = ? AND item_type = ?
                ORDER BY timestamp DESC LIMIT 20
            ''', (user_id, item_type))
            
            user_items = cursor.fetchall()
            if not user_items:
                return []
            
            # Get all available items
            cursor.execute('''
                SELECT DISTINCT item_id, metadata
                FROM user_activities
                WHERE item_type = ?
                GROUP BY item_id
            ''', (item_type,))
            
            all_items = cursor.fetchall()
            
            if not all_items:
                return []
            
            # Create TF-IDF vectors
            item_descriptions = [json.loads(item['metadata']).get('description', '') 
                               for item in all_items]
            user_item_descriptions = [json.loads(item['metadata']).get('description', '') 
                                     for item in user_items]
            
            # Vectorize
            all_descriptions = item_descriptions + user_item_descriptions
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_descriptions)
            
            # Calculate similarity
            user_vector = tfidf_matrix[len(item_descriptions):].mean(axis=0)
            item_vectors = tfidf_matrix[:len(item_descriptions)]
            similarities = cosine_similarity(user_vector, item_vectors)[0]
            
            # Get top-k recommendations
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            recommendations = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    recommendations.append({
                        'item_id': all_items[idx]['item_id'],
                        'score': float(similarities[idx]),
                        'method': 'content_based',
                        'explanation': f"Similar to items you've interacted with"
                    })
            
            conn.close()
            return recommendations
            
        except Exception as e:
            print(f"Error in content_based_filtering: {str(e)}")
            return []
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. COLLABORATIVE FILTERING
    # ─────────────────────────────────────────────────────────────────────────
    
    def collaborative_filtering(self, user_id: int, item_type: str = 'medicine',
                               top_k: int = 5) -> List[Dict]:
        """
        Collaborative filtering using SVD matrix factorization
        Recommends items liked by similar users
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build user-item interaction matrix
            cursor.execute('''
                SELECT user_id, item_id, rating
                FROM user_ratings
                WHERE item_type = ?
            ''', (item_type,))
            
            ratings = cursor.fetchall()
            if len(ratings) < 10:  # Need minimum data
                return []
            
            # Create interaction matrix
            user_ids = sorted(set(r['user_id'] for r in ratings))
            item_ids = sorted(set(r['item_id'] for r in ratings))
            
            user_idx_map = {uid: idx for idx, uid in enumerate(user_ids)}
            item_idx_map = {iid: idx for idx, iid in enumerate(item_ids)}
            
            interaction_matrix = np.zeros((len(user_ids), len(item_ids)))
            for r in ratings:
                interaction_matrix[user_idx_map[r['user_id']], 
                                 item_idx_map[r['item_id']]] = r['rating']
            
            # Apply SVD
            svd = TruncatedSVD(n_components=min(20, len(user_ids)-1, len(item_ids)-1))
            user_factors = svd.fit_transform(interaction_matrix)
            item_factors = svd.components_.T
            
            # Find similar users
            user_idx = user_idx_map.get(user_id)
            if user_idx is None:
                return []
            
            user_vector = user_factors[user_idx]
            similarities = cosine_similarity([user_vector], user_factors)[0]
            similar_users = np.argsort(similarities)[::-1][1:11]  # Top 10 similar users
            
            # Get items liked by similar users
            similar_user_ids = [user_ids[idx] for idx in similar_users]
            cursor.execute(f'''
                SELECT item_id, AVG(rating) as avg_rating
                FROM user_ratings
                WHERE user_id IN ({','.join('?' * len(similar_user_ids))})
                AND item_type = ?
                GROUP BY item_id
                ORDER BY avg_rating DESC
                LIMIT ?
            ''', similar_user_ids + [item_type, top_k])
            
            recommendations = []
            for row in cursor.fetchall():
                recommendations.append({
                    'item_id': row['item_id'],
                    'score': float(row['avg_rating']) / 5.0,
                    'method': 'collaborative',
                    'explanation': f"Recommended by users with similar health profiles"
                })
            
            conn.close()
            return recommendations
            
        except Exception as e:
            print(f"Error in collaborative_filtering: {str(e)}")
            return []
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. HYBRID FILTERING
    # ─────────────────────────────────────────────────────────────────────────
    
    def hybrid_filtering(self, user_id: int, item_type: str = 'medicine',
                        content_weight: float = 0.6, collab_weight: float = 0.4,
                        top_k: int = 5) -> List[Dict]:
        """
        Hybrid filtering combining content-based and collaborative approaches
        """
        content_recs = self.content_based_filtering(user_id, item_type, top_k * 2)
        collab_recs = self.collaborative_filtering(user_id, item_type, top_k * 2)
        
        # Merge recommendations
        rec_dict = {}
        
        for rec in content_recs:
            item_id = rec['item_id']
            rec_dict[item_id] = {
                'item_id': item_id,
                'content_score': rec['score'],
                'collab_score': 0.0,
                'explanations': [rec['explanation']]
            }
        
        for rec in collab_recs:
            item_id = rec['item_id']
            if item_id in rec_dict:
                rec_dict[item_id]['collab_score'] = rec['score']
                rec_dict[item_id]['explanations'].append(rec['explanation'])
            else:
                rec_dict[item_id] = {
                    'item_id': item_id,
                    'content_score': 0.0,
                    'collab_score': rec['score'],
                    'explanations': [rec['explanation']]
                }
        
        # Calculate hybrid score
        hybrid_recs = []
        for item_id, data in rec_dict.items():
            hybrid_score = (content_weight * data['content_score'] + 
                          collab_weight * data['collab_score'])
            if hybrid_score > 0:
                hybrid_recs.append({
                    'item_id': item_id,
                    'score': hybrid_score,
                    'method': 'hybrid',
                    'explanations': data['explanations']
                })
        
        # Sort and return top-k
        hybrid_recs.sort(key=lambda x: x['score'], reverse=True)
        return hybrid_recs[:top_k]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. CONTEXT-AWARE RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def context_aware_recommendations(self, user_id: int, context: Dict,
                                     base_recommendations: List[Dict],
                                     top_k: int = 5) -> List[Dict]:
        """
        Adjust recommendations based on context:
        - Time of day (morning/afternoon/evening)
        - Urgency level (low/medium/high)
        - Trending items
        - User location
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get trending items
            cursor.execute('''
                SELECT item_id, trend_score
                FROM trending_items
                WHERE date >= date('now', '-7 days')
                ORDER BY trend_score DESC
                LIMIT 20
            ''')
            
            trending_items = {row['item_id']: row['trend_score'] for row in cursor.fetchall()}
            
            # Adjust scores based on context
            adjusted_recs = []
            for rec in base_recommendations:
                score = rec['score']
                
                # Time-based adjustment
                time_of_day = context.get('time_of_day', 'morning')
                if time_of_day == 'morning':
                    score *= 1.1  # Boost morning recommendations
                elif time_of_day == 'evening':
                    score *= 0.9  # Reduce evening recommendations
                
                # Urgency-based adjustment
                urgency = context.get('urgency', 'normal')
                if urgency == 'high':
                    score *= 1.2
                elif urgency == 'low':
                    score *= 0.8
                
                # Trending boost
                if rec['item_id'] in trending_items:
                    score *= (1.0 + trending_items[rec['item_id']] * 0.1)
                
                adjusted_recs.append({
                    **rec,
                    'score': score,
                    'context_applied': True
                })
            
            # Sort by adjusted score
            adjusted_recs.sort(key=lambda x: x['score'], reverse=True)
            
            conn.close()
            return adjusted_recs[:top_k]
            
        except Exception as e:
            print(f"Error in context_aware_recommendations: {str(e)}")
            return base_recommendations[:top_k]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. KNOWLEDGE GRAPH-BASED RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def build_knowledge_graph(self, disease_symptom_map: Dict, 
                             drug_disease_map: Dict):
        """
        Build knowledge graph: Disease → Symptoms → Treatments → Medicines
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Clear existing graph
            cursor.execute('DELETE FROM knowledge_graph')
            
            # Add disease-symptom relationships
            for disease, symptoms in disease_symptom_map.items():
                for symptom in symptoms:
                    cursor.execute('''
                        INSERT INTO knowledge_graph 
                        (source_entity, source_type, relationship, target_entity, target_type, weight, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (disease, 'disease', 'has_symptom', symptom, 'symptom', 1.0, 0.9))
            
            # Add drug-disease relationships
            for drug, diseases in drug_disease_map.items():
                for disease in diseases:
                    cursor.execute('''
                        INSERT INTO knowledge_graph 
                        (source_entity, source_type, relationship, target_entity, target_type, weight, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (drug, 'medicine', 'treats', disease, 'disease', 1.0, 0.85))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error building knowledge graph: {str(e)}")
    
    def knowledge_graph_recommendations(self, user_symptoms: List[str],
                                       top_k: int = 5) -> List[Dict]:
        """
        Recommend medicines based on knowledge graph traversal
        Symptoms → Diseases → Medicines
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Find diseases matching symptoms
            placeholders = ','.join('?' * len(user_symptoms))
            cursor.execute(f'''
                SELECT DISTINCT target_entity as disease, COUNT(*) as match_count
                FROM knowledge_graph
                WHERE source_entity IN ({placeholders})
                AND source_type = 'symptom'
                AND relationship = 'has_symptom'
                GROUP BY target_entity
                ORDER BY match_count DESC
            ''', user_symptoms)
            
            diseases = cursor.fetchall()
            if not diseases:
                return []
            
            disease_names = [d['disease'] for d in diseases]
            
            # Find medicines for these diseases
            placeholders = ','.join('?' * len(disease_names))
            cursor.execute(f'''
                SELECT DISTINCT source_entity as medicine, 
                       COUNT(*) as disease_match_count,
                       AVG(confidence) as avg_confidence
                FROM knowledge_graph
                WHERE target_entity IN ({placeholders})
                AND target_type = 'disease'
                AND relationship = 'treats'
                GROUP BY source_entity
                ORDER BY disease_match_count DESC, avg_confidence DESC
                LIMIT ?
            ''', disease_names + [top_k])
            
            recommendations = []
            for row in cursor.fetchall():
                recommendations.append({
                    'item_id': row['medicine'],
                    'score': float(row['avg_confidence']),
                    'method': 'knowledge_graph',
                    'explanation': f"Recommended for {row['disease_match_count']} matching conditions"
                })
            
            conn.close()
            return recommendations
            
        except Exception as e:
            print(f"Error in knowledge_graph_recommendations: {str(e)}")
            return []
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6. MULTI-ARMED BANDIT FOR EXPLORATION/EXPLOITATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def epsilon_greedy_selection(self, user_id: int, arms: List[str],
                                epsilon: float = 0.1) -> str:
        """
        Epsilon-greedy strategy for exploration vs exploitation
        With probability epsilon, explore random arm; otherwise exploit best arm
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get arm statistics
            cursor.execute('''
                SELECT arm_name, avg_reward, pulls
                FROM bandit_arms
                WHERE user_id = ? AND arm_name IN ({})
            '''.format(','.join('?' * len(arms))), [user_id] + arms)
            
            arm_stats = {row['arm_name']: row for row in cursor.fetchall()}
            
            # Explore with probability epsilon
            if np.random.random() < epsilon:
                selected_arm = np.random.choice(arms)
            else:
                # Exploit best arm
                best_arm = max(arms, key=lambda a: arm_stats.get(a, {}).get('avg_reward', 0))
                selected_arm = best_arm
            
            conn.close()
            return selected_arm
            
        except Exception as e:
            print(f"Error in epsilon_greedy_selection: {str(e)}")
            return arms[0]
    
    def update_bandit_arm(self, user_id: int, arm_name: str, reward: float):
        """
        Update bandit arm with new reward
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO bandit_arms (user_id, arm_name, pulls, rewards, avg_reward)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(user_id, arm_name) DO UPDATE SET
                    pulls = pulls + 1,
                    rewards = rewards + ?,
                    avg_reward = (rewards + ?) / (pulls + 1),
                    last_updated = CURRENT_TIMESTAMP
            ''', (user_id, arm_name, reward, reward, reward, reward))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error updating bandit arm: {str(e)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7. EXPLAINABILITY & INTERPRETABILITY
    # ─────────────────────────────────────────────────────────────────────────
    
    def generate_explanation(self, recommendation_id: int, rec_data: Dict) -> str:
        """
        Generate human-readable explanation for recommendation
        """
        explanations = []
        
        if rec_data.get('method') == 'content_based':
            explanations.append("Based on your medical history and similar conditions")
        elif rec_data.get('method') == 'collaborative':
            explanations.append("Recommended by patients with similar health profiles")
        elif rec_data.get('method') == 'hybrid':
            explanations.append("Personalized recommendation combining multiple factors")
        elif rec_data.get('method') == 'knowledge_graph':
            explanations.append("Clinically associated with your symptoms")
        
        if rec_data.get('context_applied'):
            explanations.append("Adjusted for current time and urgency")
        
        if rec_data.get('trending'):
            explanations.append("Currently trending among similar patients")
        
        return ". ".join(explanations)
    
    def store_explanation(self, recommendation_id: int, explanation_type: str,
                         explanation_text: str, confidence: float = 0.8):
        """
        Store explanation in database for audit trail
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO recommendation_explanations
                (recommendation_id, explanation_type, explanation_text, confidence)
                VALUES (?, ?, ?, ?)
            ''', (recommendation_id, explanation_type, explanation_text, confidence))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing explanation: {str(e)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 8. USER SEGMENTATION & COHORT ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    
    def segment_user(self, user_id: int) -> Dict[str, str]:
        """
        Segment user based on behavior and health profile
        Returns: {segment_type: segment_value}
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            segments = {}
            
            # Activity-based segmentation
            cursor.execute('''
                SELECT COUNT(*) as activity_count
                FROM user_activities
                WHERE user_id = ? AND timestamp > datetime('now', '-30 days')
            ''', (user_id,))
            
            activity_count = cursor.fetchone()['activity_count']
            if activity_count > 20:
                segments['engagement'] = 'high'
            elif activity_count > 5:
                segments['engagement'] = 'medium'
            else:
                segments['engagement'] = 'low'
            
            # Health-based segmentation
            cursor.execute('''
                SELECT COUNT(DISTINCT record_type) as condition_count
                FROM medical_records
                WHERE patient_id = ?
            ''', (user_id,))
            
            condition_count = cursor.fetchone()['condition_count']
            if condition_count >= 3:
                segments['health_status'] = 'multi_condition'
            elif condition_count == 2:
                segments['health_status'] = 'dual_condition'
            else:
                segments['health_status'] = 'single_condition'
            
            # Store segments
            for seg_type, seg_value in segments.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO user_segments
                    (user_id, segment_name, segment_type, segment_value, confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, f"{seg_type}_{seg_value}", seg_type, seg_value, 0.8))
            
            conn.commit()
            conn.close()
            
            return segments
            
        except Exception as e:
            print(f"Error segmenting user: {str(e)}")
            return {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # 9. A/B TESTING FRAMEWORK
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_ab_test(self, test_name: str, variant_a: str, variant_b: str) -> int:
        """
        Create A/B test for recommendation algorithms
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ab_tests (test_name, variant_a, variant_b)
                VALUES (?, ?, ?)
            ''', (test_name, variant_a, variant_b))
            
            conn.commit()
            test_id = cursor.lastrowid
            conn.close()
            
            return test_id
            
        except Exception as e:
            print(f"Error creating A/B test: {str(e)}")
            return -1
    
    def assign_variant(self, test_id: int, user_id: int) -> str:
        """
        Randomly assign user to variant A or B
        """
        variant = 'A' if np.random.random() < 0.5 else 'B'
        return variant
    
    def record_ab_test_result(self, test_id: int, user_id: int, variant: str,
                             metric_name: str, metric_value: float):
        """
        Record A/B test result
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ab_test_results
                (test_id, user_id, variant, metric_name, metric_value)
                VALUES (?, ?, ?, ?, ?)
            ''', (test_id, user_id, variant, metric_name, metric_value))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error recording A/B test result: {str(e)}")
    
    def analyze_ab_test(self, test_id: int) -> Dict:
        """
        Analyze A/B test results and determine winner
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get test info
            cursor.execute('SELECT * FROM ab_tests WHERE id = ?', (test_id,))
            test = cursor.fetchone()
            
            # Get results by variant
            cursor.execute('''
                SELECT variant, metric_name, AVG(metric_value) as avg_value, COUNT(*) as count
                FROM ab_test_results
                WHERE test_id = ?
                GROUP BY variant, metric_name
            ''', (test_id,))
            
            results = {}
            for row in cursor.fetchall():
                if row['variant'] not in results:
                    results[row['variant']] = {}
                results[row['variant']][row['metric_name']] = {
                    'avg_value': row['avg_value'],
                    'count': row['count']
                }
            
            conn.close()
            
            return {
                'test_name': test['test_name'],
                'variant_a': test['variant_a'],
                'variant_b': test['variant_b'],
                'results': results
            }
            
        except Exception as e:
            print(f"Error analyzing A/B test: {str(e)}")
            return {}


# Global instance
advanced_recommendation_engine = AdvancedRecommendationEngine()
