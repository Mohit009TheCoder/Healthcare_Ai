#!/usr/bin/env python3
"""
Populate Healthcare Database with Realistic Sample Data
Fills medical_records, prescriptions, appointments, and user_activities tables
"""

import sqlite3
import os
from datetime import datetime, timedelta
import random
import json

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'healthcare.db')

# Sample data
DISEASES = [
    'Hypertension', 'Type 2 Diabetes', 'Heart Disease', 'Asthma', 
    'COPD', 'Arthritis', 'Migraine', 'Anxiety Disorder', 'Depression',
    'Thyroid Disorder', 'High Cholesterol', 'Obesity', 'Sleep Apnea'
]

MEDICINES = [
    'Lisinopril', 'Metformin', 'Atorvastatin', 'Amlodipine', 'Omeprazole',
    'Albuterol', 'Sertraline', 'Ibuprofen', 'Amoxicillin', 'Levothyroxine',
    'Metoprolol', 'Aspirin', 'Vitamin D3', 'Metformin XR', 'Losartan',
    'Simvastatin', 'Fluoxetine', 'Gabapentin', 'Tramadol', 'Prednisone'
]

SYMPTOMS = [
    'Chest pain', 'Shortness of breath', 'Fatigue', 'Headache', 'Dizziness',
    'Nausea', 'Fever', 'Cough', 'Sore throat', 'Body aches', 'Anxiety',
    'Insomnia', 'Joint pain', 'Muscle weakness', 'Blurred vision'
]

ACTIVITY_TYPES = [
    'login', 'view_dashboard', 'view_medical_record', 'view_prescription',
    'schedule_appointment', 'view_recommendation', 'click_recommendation',
    'update_profile', 'view_analytics', 'download_report'
]

def get_random_date(days_back=90):
    """Get a random date within the last N days"""
    return datetime.now() - timedelta(days=random.randint(0, days_back))

def populate_medical_records():
    """Populate medical_records table with sample data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Populating medical_records...")
    
    # Get patient IDs (assuming patients are users with role='patient')
    cursor.execute('SELECT id FROM users WHERE role = "patient"')
    patients = [row[0] for row in cursor.fetchall()]
    
    # Get doctor IDs
    cursor.execute('SELECT id FROM users WHERE role = "doctor"')
    doctors = [row[0] for row in cursor.fetchall()]
    
    if not patients or not doctors:
        print("  ⚠️  No patients or doctors found. Skipping medical records.")
        conn.close()
        return
    
    records_added = 0
    record_types = ['diabetes', 'heart', 'symptom']
    
    for patient_id in patients:
        # Add 3-8 medical records per patient
        num_records = random.randint(3, 8)
        for _ in range(num_records):
            doctor_id = random.choice(doctors)
            record_type = random.choice(record_types)
            disease = random.choice(DISEASES)
            symptoms = random.sample(SYMPTOMS, random.randint(2, 4))
            
            # Simulate prediction results
            if record_type == 'diabetes':
                prediction_result = random.choice(['Positive', 'Negative'])
                confidence = round(random.uniform(0.85, 0.99), 3)
                risk_score = round(random.uniform(0.3, 0.9), 2)
            elif record_type == 'heart':
                prediction_result = random.choice(['Heart Disease', 'No Heart Disease'])
                confidence = round(random.uniform(0.90, 0.99), 3)
                risk_score = round(random.uniform(0.2, 0.8), 2)
            else:  # symptom
                prediction_result = disease
                confidence = round(random.uniform(0.80, 0.98), 3)
                risk_score = round(random.uniform(0.4, 0.9), 2)
            
            input_data = json.dumps({'symptoms': symptoms, 'disease': disease})
            recommendations = json.dumps([f"Follow up with {disease} specialist", "Monitor symptoms", "Take prescribed medications"])
            notes = f"Patient presents with {', '.join(symptoms)}. Predicted: {prediction_result}. Confidence: {confidence}"
            created_at = get_random_date(90)
            
            try:
                cursor.execute('''
                    INSERT INTO medical_records 
                    (patient_id, doctor_id, record_type, prediction_result, confidence, risk_score, input_data, recommendations, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (patient_id, doctor_id, record_type, prediction_result, confidence, risk_score, input_data, recommendations, notes, created_at))
                records_added += 1
            except Exception as e:
                print(f"  Error adding medical record: {e}")
    
    conn.commit()
    print(f"  ✅ Added {records_added} medical records")
    conn.close()

def populate_prescriptions():
    """Populate prescriptions table with sample data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Populating prescriptions...")
    
    # Get patient IDs
    cursor.execute('SELECT id FROM users WHERE role = "patient"')
    patients = [row[0] for row in cursor.fetchall()]
    
    # Get doctor IDs
    cursor.execute('SELECT id FROM users WHERE role = "doctor"')
    doctors = [row[0] for row in cursor.fetchall()]
    
    # Get medical record IDs
    cursor.execute('SELECT id FROM medical_records')
    medical_records = [row[0] for row in cursor.fetchall()]
    
    if not patients or not doctors:
        print("  ⚠️  No patients or doctors found. Skipping prescriptions.")
        conn.close()
        return
    
    prescriptions_added = 0
    for patient_id in patients:
        # Add 2-6 prescriptions per patient
        num_prescriptions = random.randint(2, 6)
        for _ in range(num_prescriptions):
            doctor_id = random.choice(doctors)
            medicine = random.choice(MEDICINES)
            dosage = random.choice(['250mg', '500mg', '1000mg', '5mg', '10mg', '20mg', '50mg'])
            frequency = random.choice(['Once daily', 'Twice daily', 'Three times daily', 'As needed'])
            duration = random.choice(['7 days', '14 days', '30 days', '60 days', '90 days'])
            instructions = f"Take {dosage} {frequency}. Do not exceed recommended dose."
            status = random.choice(['active', 'completed', 'cancelled'])
            prescribed_at = get_random_date(90)
            medical_record_id = random.choice(medical_records) if medical_records else None
            
            try:
                cursor.execute('''
                    INSERT INTO prescriptions 
                    (patient_id, doctor_id, medical_record_id, medicine_name, dosage, frequency, duration, instructions, status, prescribed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (patient_id, doctor_id, medical_record_id, medicine, dosage, frequency, duration, instructions, status, prescribed_at))
                prescriptions_added += 1
            except Exception as e:
                print(f"  Error adding prescription: {e}")
    
    conn.commit()
    print(f"  ✅ Added {prescriptions_added} prescriptions")
    conn.close()

def populate_appointments():
    """Populate appointments table with sample data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Populating appointments...")
    
    # Get patient IDs
    cursor.execute('SELECT id FROM users WHERE role = "patient"')
    patients = [row[0] for row in cursor.fetchall()]
    
    # Get doctor IDs
    cursor.execute('SELECT id FROM users WHERE role = "doctor"')
    doctors = [row[0] for row in cursor.fetchall()]
    
    if not patients or not doctors:
        print("  ⚠️  No patients or doctors found. Skipping appointments.")
        conn.close()
        return
    
    appointments_added = 0
    for patient_id in patients:
        # Add 2-5 appointments per patient
        num_appointments = random.randint(2, 5)
        for _ in range(num_appointments):
            doctor_id = random.choice(doctors)
            appointment_date = datetime.now() + timedelta(days=random.randint(-30, 60))
            reason = random.choice(['Follow-up', 'Consultation', 'Check-up', 'Lab work', 'Medication review'])
            status = random.choice(['scheduled', 'completed', 'cancelled'])
            notes = f"{reason} appointment for patient health assessment"
            created_at = get_random_date(90)
            
            try:
                cursor.execute('''
                    INSERT INTO appointments 
                    (patient_id, doctor_id, appointment_date, reason, status, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (patient_id, doctor_id, appointment_date, reason, status, notes, created_at))
                appointments_added += 1
            except Exception as e:
                print(f"  Error adding appointment: {e}")
    
    conn.commit()
    print(f"  ✅ Added {appointments_added} appointments")
    conn.close()

def populate_user_activities():
    """Populate user_activities table with realistic activity data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Populating user_activities...")
    
    # Get all active users
    cursor.execute('SELECT id FROM users WHERE is_active = 1')
    users = [row[0] for row in cursor.fetchall()]
    
    if not users:
        print("  ⚠️  No active users found. Skipping user activities.")
        conn.close()
        return
    
    activities_added = 0
    for user_id in users:
        # Add 10-30 activities per user over the last 30 days
        num_activities = random.randint(10, 30)
        for _ in range(num_activities):
            activity_type = random.choice(ACTIVITY_TYPES)
            timestamp = get_random_date(30)
            item_id = str(random.randint(1, 100))
            item_type = random.choice(['medicine', 'disease_info', 'health_tip', 'article'])
            metadata = json.dumps({
                'page': f'/dashboard/{activity_type}',
                'duration_seconds': random.randint(10, 600),
                'user_agent': 'Mozilla/5.0 (Healthcare App)'
            })
            session_id = f"session_{user_id}_{random.randint(1000, 9999)}"
            
            try:
                cursor.execute('''
                    INSERT INTO user_activities 
                    (user_id, activity_type, item_id, item_type, metadata, timestamp, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, activity_type, item_id, item_type, metadata, timestamp, session_id))
                activities_added += 1
            except Exception as e:
                print(f"  Error adding user activity: {e}")
    
    conn.commit()
    print(f"  ✅ Added {activities_added} user activities")
    conn.close()

def populate_recommendation_history():
    """Populate recommendation_history table with sample data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Populating recommendation_history...")
    
    # Get all active users
    cursor.execute('SELECT id FROM users WHERE is_active = 1')
    users = [row[0] for row in cursor.fetchall()]
    
    if not users:
        print("  ⚠️  No active users found. Skipping recommendation history.")
        conn.close()
        return
    
    recommendations_added = 0
    recommendation_types = ['content_based', 'collaborative', 'hybrid', 'context_aware']
    
    for user_id in users:
        # Add 5-15 recommendations per user
        num_recommendations = random.randint(5, 15)
        for _ in range(num_recommendations):
            recommendation_type = random.choice(recommendation_types)
            item_id = random.randint(1, 100)
            item_type = random.choice(['medicine', 'disease_info', 'health_tip', 'article'])
            score = round(random.uniform(0.5, 1.0), 3)
            clicked = random.choice([0, 1])
            shown_at = get_random_date(30)
            clicked_at = shown_at + timedelta(hours=random.randint(0, 24)) if clicked else None
            
            try:
                cursor.execute('''
                    INSERT INTO recommendation_history 
                    (user_id, recommendation_type, item_id, item_type, score, clicked, shown_at, clicked_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, recommendation_type, item_id, item_type, score, clicked, shown_at, clicked_at))
                recommendations_added += 1
            except Exception as e:
                print(f"  Error adding recommendation: {e}")
    
    conn.commit()
    print(f"  ✅ Added {recommendations_added} recommendations")
    conn.close()

def verify_data():
    """Verify that data was populated correctly"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("DATA VERIFICATION")
    print("="*60)
    
    tables = [
        'users', 'medical_records', 'prescriptions', 'appointments',
        'user_activities', 'recommendation_history'
    ]
    
    for table in tables:
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            print(f"  {table:.<40} {count:>6} records")
        except Exception as e:
            print(f"  {table:.<40} ERROR: {e}")
    
    conn.close()
    print("="*60)

def main():
    """Main function to populate all data"""
    print("\n" + "="*60)
    print("HEALTHCARE DATABASE - SAMPLE DATA POPULATION")
    print("="*60 + "\n")
    
    try:
        # Check if database exists
        if not os.path.exists(DB_PATH):
            print(f"❌ Database not found at {DB_PATH}")
            print("Please run the Flask app first to initialize the database.")
            return
        
        # Populate all tables
        populate_medical_records()
        populate_prescriptions()
        populate_appointments()
        populate_user_activities()
        populate_recommendation_history()
        
        # Verify data
        verify_data()
        
        print("\n✅ Sample data population completed successfully!")
        print("\nYou can now:")
        print("  1. Log in with demo credentials")
        print("  2. View dashboards with populated data")
        print("  3. See analytics and recommendations")
        
    except Exception as e:
        print(f"\n❌ Error during data population: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
