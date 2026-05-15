"""
Advanced Analytics Module
Real-time analytics, cohort analysis, retention metrics, and predictive analytics
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'healthcare.db')


class AdvancedAnalytics:
    """Advanced analytics engine for healthcare system"""
    
    def __init__(self):
        self.init_tables()
    
    def init_tables(self):
        """Initialize advanced analytics tables"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Real-time metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_metric_time (metric_name, timestamp)
            )
        ''')
        
        # Cohort data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cohorts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cohort_name TEXT NOT NULL,
                cohort_date DATE NOT NULL,
                user_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(cohort_name, cohort_date)
            )
        ''')
        
        # Cohort retention
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cohort_retention (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cohort_id INTEGER NOT NULL,
                days_since_join INTEGER NOT NULL,
                retained_users INTEGER DEFAULT 0,
                retention_rate REAL DEFAULT 0.0,
                FOREIGN KEY (cohort_id) REFERENCES cohorts(id)
            )
        ''')
        
        # Churn prediction
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS churn_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Anomaly detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anomaly_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                description TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Attribution tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attribution_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                touchpoint TEXT NOT NULL,
                touchpoint_type TEXT NOT NULL,
                conversion_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. REAL-TIME ANALYTICS
    # ─────────────────────────────────────────────────────────────────────────
    
    def track_realtime_metric(self, metric_name: str, metric_value: float):
        """
        Track real-time metric
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_metrics (metric_name, metric_value)
                VALUES (?, ?)
            ''', (metric_name, metric_value))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error tracking realtime metric: {str(e)}")
    
    def get_realtime_dashboard(self, minutes: int = 60) -> Dict:
        """
        Get real-time dashboard metrics for last N minutes
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # Active users
            cursor.execute('''
                SELECT COUNT(DISTINCT user_id) as active_users
                FROM user_activities
                WHERE timestamp > ?
            ''', (cutoff_time,))
            
            active_users = cursor.fetchone()['active_users']
            
            # Recent predictions
            cursor.execute('''
                SELECT COUNT(*) as recent_predictions
                FROM medical_records
                WHERE created_at > ?
            ''', (cutoff_time,))
            
            recent_predictions = cursor.fetchone()['recent_predictions']
            
            # Recommendation CTR
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_shown,
                    SUM(CASE WHEN clicked = 1 THEN 1 ELSE 0 END) as total_clicked
                FROM recommendation_history
                WHERE shown_at > ?
            ''', (cutoff_time,))
            
            rec_stats = cursor.fetchone()
            ctr = (rec_stats['total_clicked'] / rec_stats['total_shown'] * 100 
                   if rec_stats['total_shown'] > 0 else 0)
            
            # Top activities
            cursor.execute('''
                SELECT activity_type, COUNT(*) as count
                FROM user_activities
                WHERE timestamp > ?
                GROUP BY activity_type
                ORDER BY count DESC
                LIMIT 5
            ''', (cutoff_time,))
            
            top_activities = {row['activity_type']: row['count'] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                'active_users': active_users,
                'recent_predictions': recent_predictions,
                'recommendation_ctr': round(ctr, 2),
                'top_activities': top_activities,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting realtime dashboard: {str(e)}")
            return {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. COHORT ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_cohort(self, cohort_name: str, cohort_date: str, user_ids: List[int]):
        """
        Create cohort of users
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO cohorts (cohort_name, cohort_date, user_count)
                VALUES (?, ?, ?)
            ''', (cohort_name, cohort_date, len(user_ids)))
            
            cohort_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return cohort_id
            
        except Exception as e:
            print(f"Error creating cohort: {str(e)}")
            return -1
    
    def calculate_cohort_retention(self, cohort_id: int, cohort_date: str) -> Dict:
        """
        Calculate retention rates for cohort
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get cohort users
            cursor.execute('''
                SELECT user_id FROM user_activities
                WHERE DATE(timestamp) = ?
                GROUP BY user_id
            ''', (cohort_date,))
            
            cohort_users = [row['user_id'] for row in cursor.fetchall()]
            
            if not cohort_users:
                return {}
            
            retention_data = {}
            
            # Calculate retention for each day
            for days_since in range(0, 31, 7):  # 0, 7, 14, 21, 30 days
                check_date = datetime.strptime(cohort_date, '%Y-%m-%d') + timedelta(days=days_since)
                
                cursor.execute(f'''
                    SELECT COUNT(DISTINCT user_id) as retained
                    FROM user_activities
                    WHERE user_id IN ({','.join('?' * len(cohort_users))})
                    AND DATE(timestamp) = ?
                ''', cohort_users + [check_date.strftime('%Y-%m-%d')])
                
                retained = cursor.fetchone()['retained']
                retention_rate = (retained / len(cohort_users) * 100) if cohort_users else 0
                
                retention_data[f'day_{days_since}'] = {
                    'retained_users': retained,
                    'retention_rate': round(retention_rate, 2)
                }
            
            conn.close()
            return retention_data
            
        except Exception as e:
            print(f"Error calculating cohort retention: {str(e)}")
            return {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. CHURN PREDICTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def predict_churn(self, user_id: int, days_lookback: int = 30) -> Tuple[float, str]:
        """
        Predict churn probability for user
        Based on: activity frequency, engagement trend, last activity
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_lookback)
            
            # Get activity metrics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_activities,
                    MAX(timestamp) as last_activity,
                    COUNT(DISTINCT DATE(timestamp)) as active_days
                FROM user_activities
                WHERE user_id = ? AND timestamp > ?
            ''', (user_id, cutoff_date))
            
            metrics = cursor.fetchone()
            
            if not metrics or metrics['total_activities'] == 0:
                return 0.95, 'high'  # High churn risk if no activity
            
            # Calculate churn score
            churn_score = 0.0
            
            # Activity frequency (lower is worse)
            activity_rate = metrics['total_activities'] / days_lookback
            if activity_rate < 0.5:
                churn_score += 0.4
            elif activity_rate < 2:
                churn_score += 0.2
            
            # Days since last activity
            if metrics['last_activity']:
                days_since_last = (datetime.now() - 
                                 datetime.fromisoformat(metrics['last_activity'])).days
                if days_since_last > 14:
                    churn_score += 0.4
                elif days_since_last > 7:
                    churn_score += 0.2
            
            # Active days ratio
            active_ratio = metrics['active_days'] / days_lookback
            if active_ratio < 0.2:
                churn_score += 0.2
            
            # Determine risk level
            if churn_score > 0.7:
                risk_level = 'high'
            elif churn_score > 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Store prediction
            cursor.execute('''
                INSERT INTO churn_predictions (user_id, churn_probability, risk_level)
                VALUES (?, ?, ?)
            ''', (user_id, churn_score, risk_level))
            
            conn.commit()
            conn.close()
            
            return churn_score, risk_level
            
        except Exception as e:
            print(f"Error predicting churn: {str(e)}")
            return 0.5, 'unknown'
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. ANOMALY DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    
    def detect_anomalies(self, anomaly_type: str = 'activity') -> List[Dict]:
        """
        Detect anomalies in user behavior or system metrics
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            anomalies = []
            
            if anomaly_type == 'activity':
                # Detect unusual activity spikes
                cursor.execute('''
                    SELECT user_id, COUNT(*) as activity_count, DATE(timestamp) as activity_date
                    FROM user_activities
                    WHERE timestamp > datetime('now', '-7 days')
                    GROUP BY user_id, DATE(timestamp)
                    HAVING activity_count > 100
                ''')
                
                for row in cursor.fetchall():
                    anomalies.append({
                        'anomaly_type': 'activity_spike',
                        'entity_id': str(row['user_id']),
                        'entity_type': 'user',
                        'anomaly_score': min(row['activity_count'] / 50, 1.0),
                        'description': f"Unusual activity spike: {row['activity_count']} activities"
                    })
            
            elif anomaly_type == 'prediction':
                # Detect unusual prediction patterns
                cursor.execute('''
                    SELECT patient_id, COUNT(*) as pred_count, DATE(created_at) as pred_date
                    FROM medical_records
                    WHERE created_at > datetime('now', '-7 days')
                    GROUP BY patient_id, DATE(created_at)
                    HAVING pred_count > 20
                ''')
                
                for row in cursor.fetchall():
                    anomalies.append({
                        'anomaly_type': 'prediction_spike',
                        'entity_id': str(row['patient_id']),
                        'entity_type': 'patient',
                        'anomaly_score': min(row['pred_count'] / 10, 1.0),
                        'description': f"Unusual prediction frequency: {row['pred_count']} predictions"
                    })
            
            # Store anomalies
            for anomaly in anomalies:
                cursor.execute('''
                    INSERT INTO anomalies
                    (anomaly_type, entity_id, entity_type, anomaly_score, description)
                    VALUES (?, ?, ?, ?, ?)
                ''', (anomaly['anomaly_type'], anomaly['entity_id'], 
                      anomaly['entity_type'], anomaly['anomaly_score'], 
                      anomaly['description']))
            
            conn.commit()
            conn.close()
            
            return anomalies
            
        except Exception as e:
            print(f"Error detecting anomalies: {str(e)}")
            return []
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. ATTRIBUTION MODELING
    # ─────────────────────────────────────────────────────────────────────────
    
    def track_touchpoint(self, user_id: int, touchpoint: str, 
                        touchpoint_type: str, conversion_id: str = None):
        """
        Track user touchpoint in conversion funnel
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO attribution_events
                (user_id, touchpoint, touchpoint_type, conversion_id)
                VALUES (?, ?, ?, ?)
            ''', (user_id, touchpoint, touchpoint_type, conversion_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error tracking touchpoint: {str(e)}")
    
    def calculate_attribution(self, conversion_id: str, 
                            attribution_model: str = 'linear') -> Dict:
        """
        Calculate attribution for conversion
        Models: first_touch, last_touch, linear, time_decay
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all touchpoints for conversion
            cursor.execute('''
                SELECT user_id, touchpoint, touchpoint_type, timestamp
                FROM attribution_events
                WHERE conversion_id = ?
                ORDER BY timestamp ASC
            ''', (conversion_id,))
            
            touchpoints = cursor.fetchall()
            
            if not touchpoints:
                return {}
            
            attribution = {}
            
            if attribution_model == 'first_touch':
                # All credit to first touchpoint
                first = touchpoints[0]
                attribution[first['touchpoint']] = 1.0
            
            elif attribution_model == 'last_touch':
                # All credit to last touchpoint
                last = touchpoints[-1]
                attribution[last['touchpoint']] = 1.0
            
            elif attribution_model == 'linear':
                # Equal credit to all touchpoints
                credit = 1.0 / len(touchpoints)
                for tp in touchpoints:
                    attribution[tp['touchpoint']] = attribution.get(tp['touchpoint'], 0) + credit
            
            elif attribution_model == 'time_decay':
                # More credit to recent touchpoints
                total_weight = sum(range(1, len(touchpoints) + 1))
                for i, tp in enumerate(touchpoints):
                    weight = (i + 1) / total_weight
                    attribution[tp['touchpoint']] = attribution.get(tp['touchpoint'], 0) + weight
            
            conn.close()
            return attribution
            
        except Exception as e:
            print(f"Error calculating attribution: {str(e)}")
            return {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6. PREDICTIVE ANALYTICS
    # ─────────────────────────────────────────────────────────────────────────
    
    def forecast_trend(self, metric_name: str, days_ahead: int = 7) -> List[Dict]:
        """
        Forecast metric trend using simple exponential smoothing
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get historical data
            cursor.execute('''
                SELECT DATE(timestamp) as date, AVG(metric_value) as value
                FROM realtime_metrics
                WHERE metric_name = ?
                AND timestamp > datetime('now', '-30 days')
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
            ''', (metric_name,))
            
            data = cursor.fetchall()
            
            if len(data) < 3:
                return []
            
            # Extract values
            values = [row['value'] for row in data]
            
            # Simple exponential smoothing
            alpha = 0.3
            forecast = []
            last_value = values[-1]
            
            for i in range(days_ahead):
                forecast_value = alpha * last_value + (1 - alpha) * (values[-1] if i == 0 else forecast[-1]['value'])
                forecast_date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
                
                forecast.append({
                    'date': forecast_date,
                    'value': round(forecast_value, 2),
                    'confidence': 0.8 - (i * 0.05)  # Decreasing confidence
                })
            
            conn.close()
            return forecast
            
        except Exception as e:
            print(f"Error forecasting trend: {str(e)}")
            return []
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7. SEGMENTATION ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    
    def segment_users_kmeans(self, n_clusters: int = 3) -> Dict:
        """
        Segment users using K-means clustering
        Based on: activity frequency, engagement, health conditions
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get user features
            cursor.execute('''
                SELECT 
                    u.id,
                    COUNT(DISTINCT ua.id) as activity_count,
                    COUNT(DISTINCT mr.id) as condition_count,
                    COUNT(DISTINCT ur.id) as rating_count
                FROM users u
                LEFT JOIN user_activities ua ON u.id = ua.user_id
                LEFT JOIN medical_records mr ON u.id = mr.patient_id
                LEFT JOIN user_ratings ur ON u.id = ur.user_id
                GROUP BY u.id
            ''')
            
            rows = cursor.fetchall()
            
            if len(rows) < n_clusters:
                return {}
            
            # Prepare feature matrix
            features = np.array([
                [row['activity_count'], row['condition_count'], row['rating_count']]
                for row in rows
            ])
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Group users by cluster
            segmentation = {}
            for i, row in enumerate(rows):
                cluster_id = int(clusters[i])
                if cluster_id not in segmentation:
                    segmentation[cluster_id] = []
                segmentation[cluster_id].append({
                    'user_id': row['id'],
                    'activity_count': row['activity_count'],
                    'condition_count': row['condition_count'],
                    'rating_count': row['rating_count']
                })
            
            conn.close()
            return segmentation
            
        except Exception as e:
            print(f"Error segmenting users: {str(e)}")
            return {}


# Global instance
advanced_analytics = AdvancedAnalytics()
