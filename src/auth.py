"""
Authentication and Authorization Module
Handles user management, login, and role-based access control
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import session, redirect, url_for, flash, request
import os

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'healthcare.db')

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'doctor', 'patient')),
            full_name TEXT NOT NULL,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Doctor profiles
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctor_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            specialization TEXT NOT NULL,
            license_number TEXT UNIQUE NOT NULL,
            years_experience INTEGER,
            qualification TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    # Patient profiles
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            date_of_birth DATE,
            gender TEXT CHECK(gender IN ('Male', 'Female', 'Other')),
            blood_group TEXT,
            address TEXT,
            emergency_contact TEXT,
            assigned_doctor_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (assigned_doctor_id) REFERENCES users(id)
        )
    ''')
    
    # Medical records
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER,
            record_type TEXT NOT NULL CHECK(record_type IN ('diabetes', 'heart', 'symptom')),
            prediction_result TEXT NOT NULL,
            confidence REAL,
            risk_score REAL,
            input_data TEXT,
            recommendations TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users(id),
            FOREIGN KEY (doctor_id) REFERENCES users(id)
        )
    ''')
    
    # Prescriptions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            medical_record_id INTEGER,
            medicine_name TEXT NOT NULL,
            dosage TEXT NOT NULL,
            frequency TEXT NOT NULL,
            duration TEXT NOT NULL,
            instructions TEXT,
            prescribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'cancelled')),
            FOREIGN KEY (patient_id) REFERENCES users(id),
            FOREIGN KEY (doctor_id) REFERENCES users(id),
            FOREIGN KEY (medical_record_id) REFERENCES medical_records(id)
        )
    ''')
    
    # Appointments
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            appointment_date TIMESTAMP NOT NULL,
            reason TEXT,
            status TEXT DEFAULT 'scheduled' CHECK(status IN ('scheduled', 'completed', 'cancelled')),
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users(id),
            FOREIGN KEY (doctor_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    
    # Create default admin if not exists
    cursor.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    if cursor.fetchone()[0] == 0:
        admin_password = hash_password('admin123')
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role, full_name, phone)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('admin', 'admin@healthcare.com', admin_password, 'admin', 'System Administrator', '1234567890'))
        conn.commit()
        print("✅ Default admin created: username='admin', password='admin123'")
    
    conn.close()

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """Verify a password against its hash"""
    return hash_password(password) == password_hash

def create_user(username, email, password, role, full_name, phone=None):
    """Create a new user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role, full_name, phone)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, email, password_hash, role, full_name, phone))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError as e:
        return None

def authenticate_user(username, password):
    """Authenticate a user and return user data"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM users WHERE username = ? AND is_active = 1
    ''', (username,))
    
    user = cursor.fetchone()
    
    if user and verify_password(password, user['password_hash']):
        # Update last login
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
        ''', (user['id'],))
        conn.commit()
        conn.close()
        return dict(user)
    
    conn.close()
    return None

def get_user_by_id(user_id):
    """Get user by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    return dict(user) if user else None

def get_user_profile(user_id, role):
    """Get user profile based on role"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if role == 'doctor':
        cursor.execute('''
            SELECT u.*, d.specialization, d.license_number, d.years_experience, d.qualification
            FROM users u
            LEFT JOIN doctor_profiles d ON u.id = d.user_id
            WHERE u.id = ?
        ''', (user_id,))
    elif role == 'patient':
        cursor.execute('''
            SELECT u.*, p.date_of_birth, p.gender, p.blood_group, p.address, 
                   p.emergency_contact, p.assigned_doctor_id,
                   d.full_name as doctor_name
            FROM users u
            LEFT JOIN patient_profiles p ON u.id = p.user_id
            LEFT JOIN users d ON p.assigned_doctor_id = d.id
            WHERE u.id = ?
        ''', (user_id,))
    else:
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    
    profile = cursor.fetchone()
    conn.close()
    
    return dict(profile) if profile else None

def create_doctor_profile(user_id, specialization, license_number, years_experience=0, qualification=''):
    """Create doctor profile"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO doctor_profiles (user_id, specialization, license_number, years_experience, qualification)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, specialization, license_number, years_experience, qualification))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def create_patient_profile(user_id, date_of_birth=None, gender=None, blood_group=None, 
                          address=None, emergency_contact=None, assigned_doctor_id=None):
    """Create patient profile"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO patient_profiles (user_id, date_of_birth, gender, blood_group, 
                                         address, emergency_contact, assigned_doctor_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, date_of_birth, gender, blood_group, address, emergency_contact, assigned_doctor_id))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def get_all_doctors():
    """Get all active doctors"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT u.id, u.full_name, u.email, u.phone, 
               d.specialization, d.license_number, d.years_experience
        FROM users u
        JOIN doctor_profiles d ON u.id = d.user_id
        WHERE u.role = 'doctor' AND u.is_active = 1
        ORDER BY u.full_name
    ''')
    
    doctors = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return doctors

def get_all_patients(doctor_id=None):
    """Get all patients, optionally filtered by doctor"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if doctor_id:
        cursor.execute('''
            SELECT u.id, u.full_name, u.email, u.phone,
                   p.date_of_birth, p.gender, p.blood_group
            FROM users u
            JOIN patient_profiles p ON u.id = p.user_id
            WHERE u.role = 'patient' AND u.is_active = 1 AND p.assigned_doctor_id = ?
            ORDER BY u.full_name
        ''', (doctor_id,))
    else:
        cursor.execute('''
            SELECT u.id, u.full_name, u.email, u.phone,
                   p.date_of_birth, p.gender, p.blood_group
            FROM users u
            JOIN patient_profiles p ON u.id = p.user_id
            WHERE u.role = 'patient' AND u.is_active = 1
            ORDER BY u.full_name
        ''')
    
    patients = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return patients

# Decorators for route protection
def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def role_required(*roles):
    """Decorator to require specific role(s)"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash('Please log in to access this page.', 'warning')
                return redirect(url_for('login'))
            
            if session.get('role') not in roles:
                flash('You do not have permission to access this page.', 'danger')
                return redirect(url_for('dashboard_home'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Initialize database on module import
init_db()
