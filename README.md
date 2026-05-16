# 🏥 MediAI: Advanced Healthcare Intelligence Platform

![Version](https://img.shields.io/badge/version-3.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)
![License](https://img.shields.io/badge/license-Educational-green.svg)

**MediAI** is a high-fidelity, professional-grade healthcare management and diagnostic platform. It leverages state-of-the-art Machine Learning models to provide accurate disease predictions, personalized drug recommendations, and a robust consultation ecosystem for patients and doctors.

---

## 🌟 Key Features

### 🔐 Multi-Tier Role-Based Access Control (RBAC)
- **Patient Portal**: Submit complaints, track medical history, receive AI-driven diagnosis suggestions, and communicate with assigned doctors.
- **Doctor Dashboard**: Manage patient consultations, review AI predictions for symptoms, prescribe medications, and track patient progress.
- **Admin Command Center**: System-wide analytics, user management, and performance monitoring of AI models.

### 🤖 Intelligence Engine
- **Diabetes Prediction**: High-precision model analyzing glucose, BMI, and family history.
- **Heart Disease Analysis**: Comprehensive risk assessment using cardiovascular metrics.
- **Symptom-Disease Mapping**: Advanced ensemble model capable of identifying 41+ diseases from complex symptom sets.
- **Smart Drug Recommendations**: Evidence-based suggestions filtered by patient risk profile, age, and gender.

### 📊 Advanced Analytics & Visualization
- **Dynamic Dashboards**: 12+ interactive Plotly charts providing deep insights into disease distributions, model confidence, and feature importance.
- **Real-time Monitoring**: Integrated health checks and statistics for system reliability.

### 🛡️ Security & Performance
- **Enterprise-Grade Security**: Password hashing (SHA-256), session management, and CSRF/XSS protection.
- **Performance Optimized**: Prediction latency under 100ms; asynchronous logging to prevent I/O blocking.
- **Rate Limiting**: Intelligent request throttling to prevent DDoS and API abuse.

---

## 🛠️ Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Backend** | Python 3.11+, Flask, SQLite3 |
| **Frontend** | HTML5, CSS3 (Modern Glassmorphism), Vanilla JavaScript |
| **Machine Learning** | Scikit-learn, Pandas, NumPy, Joblib |
| **Visualization** | Plotly.js |
| **Security** | Flask-Limiter, Flask-Session, Secure Headers |
| **Logging** | Python Logging (Custom Rotating Files) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- `pip` or `conda`

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Mohit009TheCoder/Healthcare_Ai.git
   cd flask-healthcare
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Database & Models**
   ```bash
   # Initialize tables and create default admin
   python -c "from src.auth import init_db; init_db()"
   
   # Populate with sample demo data (optional)
   python populate_sample_data.py
   ```

5. **Run the Application**
   ```bash
   python app.py
   ```

### Default Credentials
| Role | Username | Password |
| :--- | :--- | :--- |
| **Admin** | `admin` | `admin123` |
| **Demo Doctor** | `doctor1` | `doctor123` |
| **Demo Patient** | `patient1` | `patient123` |

---

## 📁 Project Structure

```text
flask-healthcare/
├── app.py                # Main application gateway & API routes
├── src/                  # Core logic modules
│   ├── auth.py           # Authentication, RBAC & DB initialization
│   ├── consultation.py   # Patient-Doctor consultation logic
│   ├── analytics.py      # Dashboard & data analysis engine
│   ├── validators.py     # Input sanitization & validation
│   └── logger_config.py  # Centralized logging system
├── models/               # Pre-trained PKL files (RF, GradientBoost, etc.)
├── data/                 # Datasets and SQLite database
├── templates/            # High-fidelity HTML templates (RBAC-aware)
├── static/               # CSS (Glassmorphism), JS, and Images
├── logs/                 # System and Error logs
├── scripts/              # Setup and maintenance scripts
└── notebooks/            # Research & Model Development (EDA)
```

---

## 🧪 Testing & Quality Assurance

- **Health Checks**: `/health` endpoint provides real-time model and DB status.
- **Automated Tests**: Run `python test_dashboard.py` to verify visualization logic.
- **Log Inspection**: Monitor `logs/errors.log` for any runtime exceptions.

---

## ⚠️ Medical Disclaimer

**MediAI is an educational demonstration tool.** The predictions and recommendations provided are generated by machine learning models and should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a certified healthcare professional for medical concerns.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the Educational License. See `LICENSE` for more information.

**Author:** [Mohit Jain](https://github.com/Mohit009TheCoder)  
**Status:** v3.1.0 (Production-Ready Demo)
