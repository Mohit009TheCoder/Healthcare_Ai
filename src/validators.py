# validators.py
"""
Input validation module for healthcare prediction system.
Ensures all user inputs are within safe and medically reasonable ranges.
"""

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_age(age):
    """Validate age input (0-120 years)"""
    try:
        age = float(age)
        if not 0 <= age <= 120:
            raise ValidationError("Age must be between 0 and 120 years")
        return age
    except (ValueError, TypeError):
        raise ValidationError("Age must be a valid number")


def validate_glucose(glucose):
    """Validate glucose level (0-500 mg/dL)"""
    try:
        glucose = float(glucose)
        if not 0 <= glucose <= 500:
            raise ValidationError("Glucose level must be between 0 and 500 mg/dL")
        return glucose
    except (ValueError, TypeError):
        raise ValidationError("Glucose level must be a valid number")


def validate_bmi(bmi):
    """Validate BMI (10-60)"""
    try:
        bmi = float(bmi)
        if not 10 <= bmi <= 60:
            raise ValidationError("BMI must be between 10 and 60")
        return bmi
    except (ValueError, TypeError):
        raise ValidationError("BMI must be a valid number")


def validate_insulin(insulin):
    """Validate insulin level (0-1000 µU/mL)"""
    try:
        insulin = float(insulin)
        if not 0 <= insulin <= 1000:
            raise ValidationError("Insulin level must be between 0 and 1000 µU/mL")
        return insulin
    except (ValueError, TypeError):
        raise ValidationError("Insulin level must be a valid number")


def validate_blood_pressure(bp):
    """Validate blood pressure (40-250 mmHg)"""
    try:
        bp = float(bp)
        if not 40 <= bp <= 250:
            raise ValidationError("Blood pressure must be between 40 and 250 mmHg")
        return bp
    except (ValueError, TypeError):
        raise ValidationError("Blood pressure must be a valid number")


def validate_pregnancies(pregnancies):
    """Validate number of pregnancies (0-20)"""
    try:
        pregnancies = float(pregnancies)
        if not 0 <= pregnancies <= 20:
            raise ValidationError("Number of pregnancies must be between 0 and 20")
        return pregnancies
    except (ValueError, TypeError):
        raise ValidationError("Number of pregnancies must be a valid number")


def validate_pedigree(pedigree):
    """Validate diabetes pedigree function (0-3)"""
    try:
        pedigree = float(pedigree)
        if not 0 <= pedigree <= 3:
            raise ValidationError("Diabetes pedigree function must be between 0 and 3")
        return pedigree
    except (ValueError, TypeError):
        raise ValidationError("Diabetes pedigree function must be a valid number")


def validate_cholesterol(chol):
    """Validate cholesterol level (100-500 mg/dL)"""
    try:
        chol = float(chol)
        if not 100 <= chol <= 500:
            raise ValidationError("Cholesterol must be between 100 and 500 mg/dL")
        return chol
    except (ValueError, TypeError):
        raise ValidationError("Cholesterol must be a valid number")


def validate_heart_rate(hr):
    """Validate heart rate (30-220 bpm)"""
    try:
        hr = float(hr)
        if not 30 <= hr <= 220:
            raise ValidationError("Heart rate must be between 30 and 220 bpm")
        return hr
    except (ValueError, TypeError):
        raise ValidationError("Heart rate must be a valid number")


def validate_yes_no(value, field_name):
    """Validate Yes/No fields"""
    if value not in ['Yes', 'No', 'yes', 'no']:
        raise ValidationError(f"{field_name} must be 'Yes' or 'No'")
    return value


def validate_symptoms(symptoms):
    """Validate symptom list"""
    if not symptoms or len(symptoms) == 0:
        raise ValidationError("Please select at least one symptom")
    if len(symptoms) > 20:
        raise ValidationError("Please select no more than 20 symptoms")
    return symptoms
