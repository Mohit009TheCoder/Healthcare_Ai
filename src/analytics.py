"""
Analytics and Reporting Module
Tracks user engagement, recommendation performance, and system metrics
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'healthcare.db')

class HealthcareAnalytics:
    """Analytics engine for healthcare recommendation system"""
    
    def __init__(self):
        self.init_analytics_tables()
    
    def init_analytics_tables(self):
        """Initialize analytics tables"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # System metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Recommendation performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendation_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_type TEXT NOT NULL,
                total_shown INTEGER DEFAULT 0,
                total_clicked INTEGER DEFAULT 0,
                total_converted INTEGER DEFAULT 0,
                click_through_rate REAL DEFAULT 0,
                conversion_rate REAL DEFAULT 0,
                avg_score REAL DEFAULT 0,
                date DATE NOT NULL,
                UNIQUE(recommendation_type, date)
            )
        ''')
        
        # User engagement metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_engagement (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_duration INTEGER,
                pages_viewed INTEGER,
                actions_taken INTEGER,
                recommendations_clicked INTEGER,
                date DATE NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def track_recommendation_click(self, recommendation_id: int):
        """Track when a user clicks on a recommendation"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE recommendation_history
            SET clicked = 1, clicked_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (recommendation_id,))
        
        conn.commit()
        conn.close()
    
    def calculate_recommendation_performance(self, days: int = 7) -> Dict:
        """Calculate recommendation algorithm performance metrics"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get recommendation stats by type
        cursor.execute('''
            SELECT 
                recommendation_type,
                COUNT(*) as total_shown,
                SUM(CASE WHEN clicked = 1 THEN 1 ELSE 0 END) as total_clicked,
                AVG(score) as avg_score
            FROM recommendation_history
            WHERE shown_at > datetime('now', '-{} days')
            GROUP BY recommendation_type
        '''.format(days))
        
        performance = {}
        for row in cursor.fetchall():
            rec_type = row['recommendation_type']
            total_shown = row['total_shown']
            total_clicked = row['total_clicked']
            
            ctr = (total_clicked / total_shown * 100) if total_shown > 0 else 0
            
            performance[rec_type] = {
                'total_shown': total_shown,
                'total_clicked': total_clicked,
                'click_through_rate': round(ctr, 2),
                'avg_score': round(row['avg_score'], 3)
            }
        
        conn.close()
        return performance
    
    def get_user_engagement_stats(self, user_id: int = None, days: int = 30) -> Dict:
        """Get user engagement statistics"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if user_id:
            # Individual user stats
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT DATE(timestamp)) as active_days,
                    COUNT(*) as total_activities,
                    COUNT(DISTINCT activity_type) as activity_types
                FROM user_activities
                WHERE user_id = ? AND timestamp > datetime('now', '-{} days')
            '''.format(days), (user_id,))
            
            stats = dict(cursor.fetchone())
            
            # Get activity breakdown
            cursor.execute('''
                SELECT activity_type, COUNT(*) as count
                FROM user_activities
                WHERE user_id = ? AND timestamp > datetime('now', '-{} days')
                GROUP BY activity_type
            '''.format(days), (user_id,))
            
            stats['activity_breakdown'] = {row['activity_type']: row['count'] 
                                          for row in cursor.fetchall()}
        else:
            # Overall system stats
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT user_id) as active_users,
                    COUNT(*) as total_activities,
                    AVG(activities_per_user) as avg_activities_per_user
                FROM (
                    SELECT user_id, COUNT(*) as activities_per_user
                    FROM user_activities
                    WHERE timestamp > datetime('now', '-{} days')
                    GROUP BY user_id
                )
            '''.format(days))
            
            stats = dict(cursor.fetchone())
        
        conn.close()
        return stats
    
    def get_trending_analysis(self, time_window: str = 'daily') -> List[Dict]:
        """Get trending items analysis"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT item_id, item_type, trend_score, category
            FROM trending_items
            WHERE time_window = ?
            ORDER BY trend_score DESC
            LIMIT 20
        ''', (time_window,))
        
        trending = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return trending
    
    def get_conversion_funnel(self, days: int = 30) -> Dict:
        """Analyze conversion funnel from recommendation to action"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Stage 1: Recommendations shown
        cursor.execute('''
            SELECT COUNT(*) FROM recommendation_history
            WHERE shown_at > datetime('now', '-{} days')
        '''.format(days))
        total_shown = cursor.fetchone()[0]
        
        # Stage 2: Recommendations clicked
        cursor.execute('''
            SELECT COUNT(*) FROM recommendation_history
            WHERE shown_at > datetime('now', '-{} days') AND clicked = 1
        '''.format(days))
        total_clicked = cursor.fetchone()[0]
        
        # Stage 3: Actions taken (prescriptions, appointments)
        cursor.execute('''
            SELECT COUNT(*) FROM prescriptions
            WHERE prescribed_at > datetime('now', '-{} days')
        '''.format(days))
        total_prescriptions = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM appointments
            WHERE created_at > datetime('now', '-{} days')
        '''.format(days))
        total_appointments = cursor.fetchone()[0]
        
        total_conversions = total_prescriptions + total_appointments
        
        conn.close()
        
        return {
            'stage_1_shown': total_shown,
            'stage_2_clicked': total_clicked,
            'stage_3_converted': total_conversions,
            'click_rate': round((total_clicked / total_shown * 100) if total_shown > 0 else 0, 2),
            'conversion_rate': round((total_conversions / total_clicked * 100) if total_clicked > 0 else 0, 2),
            'overall_conversion': round((total_conversions / total_shown * 100) if total_shown > 0 else 0, 2)
        }
    
    def get_user_behavior_patterns(self, user_id: int) -> Dict:
        """Analyze user behavior patterns"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Activity patterns by time of day
        cursor.execute('''
            SELECT 
                CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                COUNT(*) as activity_count
            FROM user_activities
            WHERE user_id = ?
            GROUP BY hour
            ORDER BY hour
        ''', (user_id,))
        
        hourly_pattern = {row['hour']: row['activity_count'] for row in cursor.fetchall()}
        
        # Most common activity types
        cursor.execute('''
            SELECT activity_type, COUNT(*) as count
            FROM user_activities
            WHERE user_id = ?
            GROUP BY activity_type
            ORDER BY count DESC
            LIMIT 5
        ''', (user_id,))
        
        top_activities = [dict(row) for row in cursor.fetchall()]
        
        # Engagement frequency
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as activities
            FROM user_activities
            WHERE user_id = ? AND timestamp > datetime('now', '-30 days')
            GROUP BY date
            ORDER BY date
        ''', (user_id,))
        
        daily_engagement = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'hourly_pattern': hourly_pattern,
            'top_activities': top_activities,
            'daily_engagement': daily_engagement,
            'avg_daily_activities': np.mean([d['activities'] for d in daily_engagement]) if daily_engagement else 0
        }
    
    def get_system_health_metrics(self) -> Dict:
        """Get overall system health metrics"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total users by role
        cursor.execute('''
            SELECT role, COUNT(*) as count
            FROM users
            WHERE is_active = 1
            GROUP BY role
        ''')
        users_by_role = dict(cursor.fetchall())
        
        # Total medical records
        cursor.execute('SELECT COUNT(*) FROM medical_records')
        total_records = cursor.fetchone()[0]
        
        # Total prescriptions
        cursor.execute('SELECT COUNT(*) FROM prescriptions WHERE status = "active"')
        active_prescriptions = cursor.fetchone()[0]
        
        # Total appointments
        cursor.execute('SELECT COUNT(*) FROM appointments WHERE status = "scheduled"')
        scheduled_appointments = cursor.fetchone()[0]
        
        # Recent activity (last 24 hours)
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) FROM user_activities
            WHERE timestamp > datetime('now', '-1 day')
        ''')
        active_users_24h = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'users_by_role': users_by_role,
            'total_medical_records': total_records,
            'active_prescriptions': active_prescriptions,
            'scheduled_appointments': scheduled_appointments,
            'active_users_24h': active_users_24h,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_admin_dashboard_data(self) -> Dict:
        """Generate comprehensive data for admin dashboard"""
        return {
            'system_health': self.get_system_health_metrics(),
            'recommendation_performance': self.calculate_recommendation_performance(days=7),
            'user_engagement': self.get_user_engagement_stats(days=30),
            'trending_items': self.get_trending_analysis('daily'),
            'conversion_funnel': self.get_conversion_funnel(days=30)
        }
    
    def generate_doctor_dashboard_data(self, doctor_id: int) -> Dict:
        """Generate data for doctor dashboard"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get doctor's patients
        cursor.execute('''
            SELECT COUNT(*) FROM patient_profiles
            WHERE assigned_doctor_id = ?
        ''', (doctor_id,))
        total_patients = cursor.fetchone()[0]
        
        # Recent medical records
        cursor.execute('''
            SELECT COUNT(*) FROM medical_records
            WHERE doctor_id = ? AND created_at > datetime('now', '-7 days')
        ''', (doctor_id,))
        recent_records = cursor.fetchone()[0]
        
        # Active prescriptions
        cursor.execute('''
            SELECT COUNT(*) FROM prescriptions
            WHERE doctor_id = ? AND status = 'active'
        ''', (doctor_id,))
        active_prescriptions = cursor.fetchone()[0]
        
        # Upcoming appointments
        cursor.execute('''
            SELECT COUNT(*) FROM appointments
            WHERE doctor_id = ? AND status = 'scheduled' 
            AND appointment_date > datetime('now')
        ''', (doctor_id,))
        upcoming_appointments = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_patients': total_patients,
            'recent_records': recent_records,
            'active_prescriptions': active_prescriptions,
            'upcoming_appointments': upcoming_appointments
        }
    
    def generate_patient_dashboard_data(self, patient_id: int) -> Dict:
        """Generate data for patient dashboard"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Medical records count
        cursor.execute('''
            SELECT COUNT(*) FROM medical_records
            WHERE patient_id = ?
        ''', (patient_id,))
        total_records = cursor.fetchone()[0]
        
        # Active prescriptions
        cursor.execute('''
            SELECT COUNT(*) FROM prescriptions
            WHERE patient_id = ? AND status = 'active'
        ''', (patient_id,))
        active_prescriptions = cursor.fetchone()[0]
        
        # Upcoming appointments
        cursor.execute('''
            SELECT COUNT(*) FROM appointments
            WHERE patient_id = ? AND status = 'scheduled'
            AND appointment_date > datetime('now')
        ''', (patient_id,))
        upcoming_appointments = cursor.fetchone()[0]
        
        # Recent activities
        cursor.execute('''
            SELECT activity_type, COUNT(*) as count
            FROM user_activities
            WHERE user_id = ? AND timestamp > datetime('now', '-7 days')
            GROUP BY activity_type
        ''', (patient_id,))
        recent_activities = {row['activity_type']: row['count'] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_medical_records': total_records,
            'active_prescriptions': active_prescriptions,
            'upcoming_appointments': upcoming_appointments,
            'recent_activities': recent_activities
        }

# Initialize analytics engine
analytics_engine = HealthcareAnalytics()
