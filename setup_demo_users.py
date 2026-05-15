#!/usr/bin/env python3
"""
Setup Demo Users for Healthcare System
Creates admin, doctor, and patient accounts for testing
"""

from src.auth import create_user, create_doctor_profile, create_patient_profile, init_db
import sys

def setup_demo_users():
    """Create demo users for testing"""
    
    print("=" * 60)
    print("Setting up Demo Users for Healthcare System")
    print("=" * 60)
    print()
    
    # Initialize database
    print("1. Initializing database...")
    init_db()
    print("   ✅ Database initialized")
    print()
    
    # Create demo doctor
    print("2. Creating demo doctor...")
    doctor_id = create_user(
        username='doctor1',
        email='doctor1@healthcare.com',
        password='doctor123',
        role='doctor',
        full_name='Dr. Sarah Johnson',
        phone='555-0101'
    )
    
    if doctor_id:
        create_doctor_profile(
            user_id=doctor_id,
            specialization='Cardiologist',
            license_number='LIC001234',
            years_experience=10,
            qualification='MD, FACC'
        )
        print("   ✅ Doctor created: doctor1 / doctor123")
    else:
        print("   ⚠️  Doctor already exists")
    print()
    
    # Create demo patient
    print("3. Creating demo patient...")
    patient_id = create_user(
        username='patient1',
        email='patient1@healthcare.com',
        password='patient123',
        role='patient',
        full_name='John Smith',
        phone='555-0102'
    )
    
    if patient_id:
        create_patient_profile(
            user_id=patient_id,
            date_of_birth='1985-06-15',
            gender='Male',
            blood_group='O+',
            address='123 Main St, City, State 12345',
            emergency_contact='555-0199',
            assigned_doctor_id=doctor_id if doctor_id else None
        )
        print("   ✅ Patient created: patient1 / patient123")
    else:
        print("   ⚠️  Patient already exists")
    print()
    
    # Create additional demo users
    print("4. Creating additional demo users...")
    
    # Doctor 2
    doctor2_id = create_user(
        username='doctor2',
        email='doctor2@healthcare.com',
        password='doctor123',
        role='doctor',
        full_name='Dr. Michael Chen',
        phone='555-0103'
    )
    if doctor2_id:
        create_doctor_profile(
            user_id=doctor2_id,
            specialization='Endocrinologist',
            license_number='LIC001235',
            years_experience=8,
            qualification='MD, FACE'
        )
        print("   ✅ Doctor 2 created: doctor2 / doctor123")
    
    # Patient 2
    patient2_id = create_user(
        username='patient2',
        email='patient2@healthcare.com',
        password='patient123',
        role='patient',
        full_name='Emily Davis',
        phone='555-0104'
    )
    if patient2_id:
        create_patient_profile(
            user_id=patient2_id,
            date_of_birth='1990-03-22',
            gender='Female',
            blood_group='A+',
            address='456 Oak Ave, City, State 12345',
            emergency_contact='555-0198',
            assigned_doctor_id=doctor2_id if doctor2_id else None
        )
        print("   ✅ Patient 2 created: patient2 / patient123")
    
    print()
    print("=" * 60)
    print("Demo Users Setup Complete!")
    print("=" * 60)
    print()
    print("📋 Demo Accounts:")
    print()
    print("👨‍💼 Admin:")
    print("   Username: admin")
    print("   Password: admin123")
    print()
    print("👨‍⚕️ Doctors:")
    print("   Username: doctor1 / doctor2")
    print("   Password: doctor123")
    print()
    print("🧑‍🦱 Patients:")
    print("   Username: patient1 / patient2")
    print("   Password: patient123")
    print()
    print("🌐 Access the system:")
    print("   http://localhost:5001/login")
    print()
    print("=" * 60)

if __name__ == '__main__':
    try:
        setup_demo_users()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
