"""
Patient-Doctor Consultation System
Handles patient complaints, doctor recommendations, and drug prescriptions
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Tuple
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'healthcare.db')

class ConsultationSystem:
    """Manages patient-doctor consultations and prescriptions"""
    
    def __init__(self):
        self.init_consultation_tables()
    
    def init_consultation_tables(self):
        """Initialize consultation-related tables"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Patient Complaints/Issues
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                severity TEXT CHECK(severity IN ('mild', 'moderate', 'severe')),
                status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'assigned', 'resolved')),
                assigned_doctor_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES users(id),
                FOREIGN KEY (assigned_doctor_id) REFERENCES users(id)
            )
        ''')
        
        # Doctor Recommendations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctor_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                diagnosis TEXT NOT NULL,
                recommended_drugs TEXT NOT NULL,
                treatment_plan TEXT NOT NULL,
                notes TEXT,
                status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'accepted', 'rejected')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (complaint_id) REFERENCES patient_complaints(id),
                FOREIGN KEY (doctor_id) REFERENCES users(id)
            )
        ''')
        
        # Drug Information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drug_information (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_name TEXT UNIQUE NOT NULL,
                generic_name TEXT,
                dosage TEXT NOT NULL,
                frequency TEXT NOT NULL,
                duration TEXT,
                side_effects TEXT,
                precautions TEXT,
                uses TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Patient-Doctor Messages
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consultation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id INTEGER NOT NULL,
                sender_id INTEGER NOT NULL,
                sender_role TEXT NOT NULL,
                message TEXT NOT NULL,
                attachment_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (complaint_id) REFERENCES patient_complaints(id),
                FOREIGN KEY (sender_id) REFERENCES users(id)
            )
        ''')
        
        # Patient Prescriptions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_prescriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id INTEGER NOT NULL,
                recommendation_id INTEGER NOT NULL,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                drugs TEXT NOT NULL,
                instructions TEXT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE,
                status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'cancelled')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (complaint_id) REFERENCES patient_complaints(id),
                FOREIGN KEY (recommendation_id) REFERENCES doctor_recommendations(id),
                FOREIGN KEY (patient_id) REFERENCES users(id),
                FOREIGN KEY (doctor_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ─── Patient Complaint Management ─────────────────────────────────────
    
    def create_complaint(self, patient_id: int, title: str, description: str, 
                        symptoms: List[str], severity: str) -> int:
        """Create a new patient complaint"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        symptoms_json = json.dumps(symptoms)
        
        cursor.execute('''
            INSERT INTO patient_complaints 
            (patient_id, title, description, symptoms, severity)
            VALUES (?, ?, ?, ?, ?)
        ''', (patient_id, title, description, symptoms_json, severity))
        
        complaint_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return complaint_id
    
    def get_patient_complaints(self, patient_id: int) -> List[Dict]:
        """Get all complaints for a patient"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                pc.id, pc.title, pc.description, pc.symptoms, pc.severity,
                pc.status, pc.assigned_doctor_id, pc.created_at,
                u.full_name as doctor_name
            FROM patient_complaints pc
            LEFT JOIN users u ON pc.assigned_doctor_id = u.id
            WHERE pc.patient_id = ?
            ORDER BY pc.created_at DESC
        ''', (patient_id,))
        
        complaints = [dict(row) for row in cursor.fetchall()]
        
        # Parse symptoms JSON
        for complaint in complaints:
            complaint['symptoms'] = json.loads(complaint['symptoms'])
        
        conn.close()
        return complaints
    
    def get_complaint_details(self, complaint_id: int) -> Dict:
        """Get detailed information about a complaint"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                pc.id, pc.patient_id, pc.title, pc.description, pc.symptoms,
                pc.severity, pc.status, pc.assigned_doctor_id, pc.created_at,
                u1.full_name as patient_name, u1.email as patient_email,
                u2.full_name as doctor_name, u2.email as doctor_email
            FROM patient_complaints pc
            LEFT JOIN users u1 ON pc.patient_id = u1.id
            LEFT JOIN users u2 ON pc.assigned_doctor_id = u2.id
            WHERE pc.id = ?
        ''', (complaint_id,))
        
        complaint = dict(cursor.fetchone())
        complaint['symptoms'] = json.loads(complaint['symptoms'])
        
        conn.close()
        return complaint
    
    def get_pending_complaints(self) -> List[Dict]:
        """Get all pending complaints (for doctor assignment)"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                pc.id, pc.title, pc.description, pc.symptoms, pc.severity,
                pc.created_at, u.full_name as patient_name, u.email as patient_email
            FROM patient_complaints pc
            JOIN users u ON pc.patient_id = u.id
            WHERE pc.status = 'pending'
            ORDER BY pc.created_at ASC
        ''')
        
        complaints = [dict(row) for row in cursor.fetchall()]
        
        for complaint in complaints:
            complaint['symptoms'] = json.loads(complaint['symptoms'])
        
        conn.close()
        return complaints
    
    def assign_complaint_to_doctor(self, complaint_id: int, doctor_id: int) -> bool:
        """Assign a complaint to a doctor"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE patient_complaints
            SET assigned_doctor_id = ?, status = 'assigned'
            WHERE id = ?
        ''', (doctor_id, complaint_id))
        
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0
    
    def get_doctor_complaints(self, doctor_id: int) -> List[Dict]:
        """Get all complaints assigned to a doctor"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                pc.id, pc.title, pc.description, pc.symptoms, pc.severity,
                pc.status, pc.created_at, u.full_name as patient_name,
                u.email as patient_email
            FROM patient_complaints pc
            JOIN users u ON pc.patient_id = u.id
            WHERE pc.assigned_doctor_id = ?
            ORDER BY pc.created_at DESC
        ''', (doctor_id,))
        
        complaints = [dict(row) for row in cursor.fetchall()]
        
        for complaint in complaints:
            complaint['symptoms'] = json.loads(complaint['symptoms'])
        
        conn.close()
        return complaints
    
    # ─── Doctor Recommendations ──────────────────────────────────────────
    
    def create_recommendation(self, complaint_id: int, doctor_id: int,
                            diagnosis: str, recommended_drugs: List[str],
                            treatment_plan: str, notes: str = None) -> int:
        """Create a doctor recommendation for a complaint"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        drugs_json = json.dumps(recommended_drugs)
        
        cursor.execute('''
            INSERT INTO doctor_recommendations
            (complaint_id, doctor_id, diagnosis, recommended_drugs, treatment_plan, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (complaint_id, doctor_id, diagnosis, drugs_json, treatment_plan, notes))
        
        recommendation_id = cursor.lastrowid
        
        # Update complaint status
        cursor.execute('''
            UPDATE patient_complaints
            SET status = 'resolved'
            WHERE id = ?
        ''', (complaint_id,))
        
        conn.commit()
        conn.close()
        
        return recommendation_id
    
    def get_recommendation(self, recommendation_id: int) -> Dict:
        """Get a specific recommendation"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                dr.id, dr.complaint_id, dr.doctor_id, dr.diagnosis,
                dr.recommended_drugs, dr.treatment_plan, dr.notes,
                dr.status, dr.created_at,
                u.full_name as doctor_name
            FROM doctor_recommendations dr
            JOIN users u ON dr.doctor_id = u.id
            WHERE dr.id = ?
        ''', (recommendation_id,))
        
        recommendation = dict(cursor.fetchone())
        recommendation['recommended_drugs'] = json.loads(recommendation['recommended_drugs'])
        
        conn.close()
        return recommendation
    
    def get_complaint_recommendations(self, complaint_id: int) -> List[Dict]:
        """Get all recommendations for a complaint"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                dr.id, dr.diagnosis, dr.recommended_drugs, dr.treatment_plan,
                dr.notes, dr.status, dr.created_at,
                u.full_name as doctor_name
            FROM doctor_recommendations dr
            JOIN users u ON dr.doctor_id = u.id
            WHERE dr.complaint_id = ?
            ORDER BY dr.created_at DESC
        ''', (complaint_id,))
        
        recommendations = [dict(row) for row in cursor.fetchall()]
        
        for rec in recommendations:
            rec['recommended_drugs'] = json.loads(rec['recommended_drugs'])
        
        conn.close()
        return recommendations
    
    # ─── Drug Information ────────────────────────────────────────────────
    
    def add_drug_info(self, drug_name: str, generic_name: str, dosage: str,
                     frequency: str, duration: str, side_effects: str,
                     precautions: str, uses: str, description: str) -> int:
        """Add drug information to database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drug_information
            (drug_name, generic_name, dosage, frequency, duration, side_effects,
             precautions, uses, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (drug_name, generic_name, dosage, frequency, duration, side_effects,
              precautions, uses, description))
        
        drug_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return drug_id
    
    def get_drug_info(self, drug_name: str) -> Dict:
        """Get information about a drug"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM drug_information
            WHERE drug_name = ?
        ''', (drug_name,))
        
        drug = cursor.fetchone()
        conn.close()
        
        return dict(drug) if drug else None
    
    def search_drugs(self, query: str) -> List[Dict]:
        """Search for drugs by name or uses"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        search_term = f"%{query}%"
        cursor.execute('''
            SELECT id, drug_name, generic_name, dosage, uses
            FROM drug_information
            WHERE drug_name LIKE ? OR generic_name LIKE ? OR uses LIKE ?
            LIMIT 20
        ''', (search_term, search_term, search_term))
        
        drugs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return drugs
    
    # ─── Consultation Messages ──────────────────────────────────────────
    
    def send_message(self, complaint_id: int, sender_id: int, sender_role: str,
                    message: str, attachment_url: str = None) -> int:
        """Send a message in consultation"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO consultation_messages
            (complaint_id, sender_id, sender_role, message, attachment_url)
            VALUES (?, ?, ?, ?, ?)
        ''', (complaint_id, sender_id, sender_role, message, attachment_url))
        
        message_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return message_id
    
    def get_consultation_messages(self, complaint_id: int) -> List[Dict]:
        """Get all messages for a consultation"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                cm.id, cm.sender_id, cm.sender_role, cm.message,
                cm.attachment_url, cm.created_at,
                u.full_name as sender_name
            FROM consultation_messages cm
            JOIN users u ON cm.sender_id = u.id
            WHERE cm.complaint_id = ?
            ORDER BY cm.created_at ASC
        ''', (complaint_id,))
        
        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return messages
    
    # ─── Prescriptions ──────────────────────────────────────────────────
    
    def create_prescription(self, complaint_id: int, recommendation_id: int,
                           patient_id: int, doctor_id: int, drugs: List[str],
                           instructions: str, start_date: str, end_date: str = None) -> int:
        """Create a prescription from recommendation"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        drugs_json = json.dumps(drugs)
        
        cursor.execute('''
            INSERT INTO patient_prescriptions
            (complaint_id, recommendation_id, patient_id, doctor_id, drugs, instructions,
             start_date, end_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (complaint_id, recommendation_id, patient_id, doctor_id, drugs_json,
              instructions, start_date, end_date))
        
        prescription_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prescription_id
    
    def get_patient_prescriptions(self, patient_id: int) -> List[Dict]:
        """Get all prescriptions for a patient"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                pp.id, pp.complaint_id, pp.drugs, pp.instructions,
                pp.start_date, pp.end_date, pp.status, pp.created_at,
                u.full_name as doctor_name
            FROM patient_prescriptions pp
            JOIN users u ON pp.doctor_id = u.id
            WHERE pp.patient_id = ?
            ORDER BY pp.created_at DESC
        ''', (patient_id,))
        
        prescriptions = [dict(row) for row in cursor.fetchall()]
        
        for rx in prescriptions:
            rx['drugs'] = json.loads(rx['drugs'])
        
        conn.close()
        return prescriptions
    
    def get_prescription_details(self, prescription_id: int) -> Dict:
        """Get detailed prescription information"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                pp.id, pp.complaint_id, pp.drugs, pp.instructions,
                pp.start_date, pp.end_date, pp.status, pp.created_at,
                u1.full_name as doctor_name, u1.email as doctor_email,
                u2.full_name as patient_name,
                dr.diagnosis, dr.treatment_plan
            FROM patient_prescriptions pp
            JOIN users u1 ON pp.doctor_id = u1.id
            JOIN users u2 ON pp.patient_id = u2.id
            LEFT JOIN doctor_recommendations dr ON pp.recommendation_id = dr.id
            WHERE pp.id = ?
        ''', (prescription_id,))
        
        prescription = dict(cursor.fetchone())
        prescription['drugs'] = json.loads(prescription['drugs'])
        
        conn.close()
        return prescription

# Initialize consultation system
consultation_system = ConsultationSystem()
