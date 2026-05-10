# 🏥 Healthcare Prediction System

A Flask-based web application for predicting diabetes, heart disease, and diagnosing diseases based on symptoms using machine learning.

## 🎯 Features

- **Diabetes Prediction** - 99.8% accuracy
- **Heart Disease Prediction** - 100% accuracy
- **Symptom-based Disease Diagnosis** - 100% accuracy (41 diseases)
- **Drug Recommendation System**
- **Interactive Dashboard** with 12 advanced visualizations
  - Disease & symptom analysis
  - Glucose and age distributions
  - Risk factor comparisons
  - Model performance metrics (Radar charts)
  - Feature importance analysis
  - Prediction confidence distributions
  - Dataset size comparisons
- **Medical Disclaimer** for legal compliance
- **Rate Limiting** for API protection
- **Comprehensive Logging** for monitoring

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Algorithm |
|-------|----------|-----------|--------|----------|-----------|
| Diabetes | 99.8% | 99.8% | 99.7% | 99.8% | Random Forest |
| Heart Disease | 100% | 100% | 100% | 100% | Gradient Boosting |
| Symptom-Disease | 100% | 100% | 99.9% | 100% | Ensemble (RF + NB + LR) |

### 📈 Dashboard Visualizations

The dashboard now includes **12 comprehensive charts**:

1. **Disease Sample Distribution** - Top 15 diseases by frequency
2. **Top 15 Most Common Symptoms** - Symptom frequency analysis
3. **Glucose Level Distribution** - Comparison by diabetes diagnosis
4. **Age Distribution** - Violin plots by diabetes status
5. **Heart Disease Risk Factors** - Grouped comparison of risk factors
6. **BMI vs Glucose Level** - Scatter plot correlation analysis
7. **Drug Rating Distribution** - Top 10 drugs by average rating
8. **ML Model Accuracy** - Comparison across all models
9. **Model Performance Metrics** - Radar chart (Precision, Recall, F1)
10. **Feature Importance** - Top 5 features for each model
11. **Prediction Confidence Distribution** - Violin plots by model
12. **Dataset Sizes** - Visual comparison of all datasets

See [DASHBOARD_IMPROVEMENTS.md](DASHBOARD_IMPROVEMENTS.md) for detailed information.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd flask-healthcare

# Install dependencies
pip install -r requirements.txt

# Train models (first time only)
python scripts/train_models.py

# Run the application
python app.py
```

### Access the Application

Open your browser and navigate to:
```
http://localhost:5001
```

## 📁 Project Structure

```
flask-healthcare/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── src/                       # Source modules
│   ├── __init__.py
│   ├── validators.py          # Input validation
│   └── logger_config.py       # Logging configuration
│
├── config/                    # Configuration files
│   ├── .env                   # Environment variables (not in git)
│   └── .env.example           # Environment template
│
├── scripts/                   # Utility scripts
│   └── train_models.py        # Model training script
│
├── data/                      # Dataset files
│   ├── diabetes_prediction.csv
│   ├── heart_disease_prediction.csv
│   ├── drug_review.csv
│   ├── DiseaseAndSymptoms.csv
│   ├── Disease_precaution.csv
│   └── world_disease.csv
│
├── models/                    # Pre-trained ML models
│   ├── diabetes_model_v1.pkl
│   ├── heart_model_v1.pkl
│   └── symptom_model_v1.pkl
│   └── ... (11 files total)
│
├── templates/                 # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── diabetes.html
│   ├── heart.html
│   ├── symptoms.html
│   └── drug_recommend.html
│
├── static/                    # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── notebooks/                 # Jupyter notebooks
│   └── Medical_Health_EDA_Complete.ipynb
│
├── logs/                      # Application logs (auto-created)
│   ├── healthcare_app.log
│   └── errors.log
│
└── tests/                     # Test files (future)
```

## 🔧 Configuration

### Environment Variables

Copy `config/.env.example` to `config/.env` and update:

```bash
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
DEBUG=False
HOST=0.0.0.0
PORT=5001
```

## 🧪 Testing

### Manual Testing

1. **Test Health Check**
```bash
curl http://localhost:5001/health
```

2. **Test Predictions**
- Navigate to http://localhost:5001
- Try each prediction form with valid/invalid inputs

3. **Test Rate Limiting**
```bash
# Make 11 requests quickly (should fail on 11th)
for i in {1..11}; do curl http://localhost:5001/health; done
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/dashboard` | GET | Analytics dashboard |
| `/predict/diabetes` | GET, POST | Diabetes prediction |
| `/predict/heart` | GET, POST | Heart disease prediction |
| `/predict/symptoms` | GET, POST | Symptom-based diagnosis |
| `/recommend/drug` | GET, POST | Drug recommendations |
| `/health` | GET | Health check |
| `/api/version` | GET | API version |
| `/api/stats` | GET | System statistics |

## 🔒 Security Features

- ✅ Input validation on all forms
- ✅ Rate limiting (200/day, 50/hour, 10/min per endpoint)
- ✅ Environment variable configuration
- ✅ Security headers (XSS, CSRF protection)
- ✅ Error handling without exposing internals
- ✅ Comprehensive logging

## ⚠️ Medical Disclaimer

**IMPORTANT:** This tool is for informational and educational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 📈 Performance

- **Startup Time:** ~2 seconds
- **Prediction Speed:** 50-60ms per request
- **Memory Usage:** ~400MB
- **Concurrent Users:** Supports multiple users with rate limiting

## 🛠️ Development

### Running in Development Mode

```bash
export FLASK_ENV=development
python app.py
```

### Running in Production Mode

```bash
export FLASK_ENV=production
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

## 📝 Logging

Logs are stored in the `logs/` directory:

- `healthcare_app.log` - All application logs
- `errors.log` - Error logs only

View logs in real-time:
```bash
tail -f logs/healthcare_app.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Your Name

## 🙏 Acknowledgments

- Dataset sources
- Flask framework
- Scikit-learn library
- Plotly for visualizations

## 📞 Support

For issues or questions:
- Check the logs: `logs/errors.log`
- Review the code documentation
- Open an issue on GitHub

---

**Version:** 1.0.0  
**Last Updated:** 2024  
**Status:** Development/Demo Ready
# Healthcare_Ai
