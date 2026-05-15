"""
Advanced API Endpoints for Healthcare Recommendation System
RESTful endpoints with proper documentation, versioning, and error handling
"""

from flask import Blueprint, request, jsonify
from functools import wraps
from datetime import datetime
import json
from typing import Dict, Tuple

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')


def handle_errors(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({'error': str(e), 'code': 'VALIDATION_ERROR'}), 400
        except KeyError as e:
            return jsonify({'error': f'Missing field: {str(e)}', 'code': 'MISSING_FIELD'}), 400
        except Exception as e:
            return jsonify({'error': str(e), 'code': 'INTERNAL_ERROR'}), 500
    return decorated_function


def require_json(f):
    """Decorator to require JSON content type"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json', 
                          'code': 'INVALID_CONTENT_TYPE'}), 400
        return f(*args, **kwargs)
    return decorated_function


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route('/recommendations/personalized', methods=['POST'])
@require_json
@handle_errors
def get_personalized_recommendations():
    """
    Get personalized recommendations for user
    
    Request body:
    {
        "user_id": int,
        "item_type": "medicine|treatment|specialist",
        "top_k": int (default: 5),
        "context": {
            "time_of_day": "morning|afternoon|evening",
            "urgency": "low|normal|high",
            "location": "string"
        }
    }
    """
    from src.advanced_recommendation_engine import advanced_recommendation_engine
    
    data = request.get_json()
    user_id = data.get('user_id')
    item_type = data.get('item_type', 'medicine')
    top_k = data.get('top_k', 5)
    context = data.get('context', {})
    
    if not user_id:
        raise ValueError('user_id is required')
    
    # Get hybrid recommendations
    base_recs = advanced_recommendation_engine.hybrid_filtering(
        user_id, item_type, top_k=top_k
    )
    
    # Apply context
    recs = advanced_recommendation_engine.context_aware_recommendations(
        user_id, context, base_recs, top_k
    )
    
    return jsonify({
        'success': True,
        'recommendations': recs,
        'count': len(recs),
        'timestamp': datetime.now().isoformat()
    }), 200


@api_bp.route('/recommendations/content-based', methods=['POST'])
@require_json
@handle_errors
def get_content_based_recommendations():
    """
    Get content-based recommendations
    
    Request body:
    {
        "user_id": int,
        "item_type": "medicine|treatment|specialist",
        "top_k": int (default: 5)
    }
    """
    from src.advanced_recommendation_engine import advanced_recommendation_engine
    
    data = request.get_json()
    user_id = data.get('user_id')
    item_type = data.get('item_type', 'medicine')
    top_k = data.get('top_k', 5)
    
    if not user_id:
        raise ValueError('user_id is required')
    
    recs = advanced_recommendation_engine.content_based_filtering(
        user_id, item_type, top_k
    )
    
    return jsonify({
        'success': True,
        'method': 'content_based',
        'recommendations': recs,
        'count': len(recs)
    }), 200


@api_bp.route('/recommendations/collaborative', methods=['POST'])
@require_json
@handle_errors
def get_collaborative_recommendations():
    """
    Get collaborative filtering recommendations
    
    Request body:
    {
        "user_id": int,
        "item_type": "medicine|treatment|specialist",
        "top_k": int (default: 5)
    }
    """
    from src.advanced_recommendation_engine import advanced_recommendation_engine
    
    data = request.get_json()
    user_id = data.get('user_id')
    item_type = data.get('item_type', 'medicine')
    top_k = data.get('top_k', 5)
    
    if not user_id:
        raise ValueError('user_id is required')
    
    recs = advanced_recommendation_engine.collaborative_filtering(
        user_id, item_type, top_k
    )
    
    return jsonify({
        'success': True,
        'method': 'collaborative',
        'recommendations': recs,
        'count': len(recs)
    }), 200


@api_bp.route('/recommendations/knowledge-graph', methods=['POST'])
@require_json
@handle_errors
def get_knowledge_graph_recommendations():
    """
    Get knowledge graph-based recommendations
    
    Request body:
    {
        "symptoms": ["symptom1", "symptom2", ...],
        "top_k": int (default: 5)
    }
    """
    from src.advanced_recommendation_engine import advanced_recommendation_engine
    
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    top_k = data.get('top_k', 5)
    
    if not symptoms:
        raise ValueError('symptoms list is required')
    
    recs = advanced_recommendation_engine.knowledge_graph_recommendations(
        symptoms, top_k
    )
    
    return jsonify({
        'success': True,
        'method': 'knowledge_graph',
        'recommendations': recs,
        'count': len(recs)
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route('/analytics/realtime', methods=['GET'])
@handle_errors
def get_realtime_analytics():
    """
    Get real-time analytics dashboard
    
    Query parameters:
    - minutes: int (default: 60) - Time window in minutes
    """
    from src.advanced_analytics import advanced_analytics
    
    minutes = request.args.get('minutes', 60, type=int)
    
    dashboard = advanced_analytics.get_realtime_dashboard(minutes)
    
    return jsonify({
        'success': True,
        'data': dashboard
    }), 200


@api_bp.route('/analytics/cohort', methods=['POST'])
@require_json
@handle_errors
def analyze_cohort():
    """
    Analyze cohort retention
    
    Request body:
    {
        "cohort_name": "string",
        "cohort_date": "YYYY-MM-DD",
        "user_ids": [int, ...]
    }
    """
    from src.advanced_analytics import advanced_analytics
    
    data = request.get_json()
    cohort_name = data.get('cohort_name')
    cohort_date = data.get('cohort_date')
    user_ids = data.get('user_ids', [])
    
    if not cohort_name or not cohort_date:
        raise ValueError('cohort_name and cohort_date are required')
    
    # Create cohort
    cohort_id = advanced_analytics.create_cohort(cohort_name, cohort_date, user_ids)
    
    # Calculate retention
    retention = advanced_analytics.calculate_cohort_retention(cohort_id, cohort_date)
    
    return jsonify({
        'success': True,
        'cohort_id': cohort_id,
        'cohort_name': cohort_name,
        'retention': retention
    }), 200


@api_bp.route('/analytics/churn-prediction/<int:user_id>', methods=['GET'])
@handle_errors
def predict_user_churn(user_id: int):
    """
    Predict churn probability for user
    
    Path parameters:
    - user_id: int
    
    Query parameters:
    - days_lookback: int (default: 30)
    """
    from src.advanced_analytics import advanced_analytics
    
    days_lookback = request.args.get('days_lookback', 30, type=int)
    
    churn_prob, risk_level = advanced_analytics.predict_churn(user_id, days_lookback)
    
    return jsonify({
        'success': True,
        'user_id': user_id,
        'churn_probability': round(churn_prob, 3),
        'risk_level': risk_level
    }), 200


@api_bp.route('/analytics/anomalies', methods=['GET'])
@handle_errors
def detect_anomalies():
    """
    Detect anomalies in system
    
    Query parameters:
    - type: "activity|prediction" (default: "activity")
    """
    from src.advanced_analytics import advanced_analytics
    
    anomaly_type = request.args.get('type', 'activity')
    
    anomalies = advanced_analytics.detect_anomalies(anomaly_type)
    
    return jsonify({
        'success': True,
        'anomaly_type': anomaly_type,
        'anomalies': anomalies,
        'count': len(anomalies)
    }), 200


@api_bp.route('/analytics/forecast', methods=['GET'])
@handle_errors
def forecast_metric():
    """
    Forecast metric trend
    
    Query parameters:
    - metric_name: string (required)
    - days_ahead: int (default: 7)
    """
    from src.advanced_analytics import advanced_analytics
    
    metric_name = request.args.get('metric_name')
    days_ahead = request.args.get('days_ahead', 7, type=int)
    
    if not metric_name:
        raise ValueError('metric_name is required')
    
    forecast = advanced_analytics.forecast_trend(metric_name, days_ahead)
    
    return jsonify({
        'success': True,
        'metric_name': metric_name,
        'forecast': forecast,
        'days_ahead': days_ahead
    }), 200


@api_bp.route('/analytics/segmentation', methods=['GET'])
@handle_errors
def get_user_segmentation():
    """
    Get user segmentation analysis
    
    Query parameters:
    - n_clusters: int (default: 3)
    """
    from src.advanced_analytics import advanced_analytics
    
    n_clusters = request.args.get('n_clusters', 3, type=int)
    
    segmentation = advanced_analytics.segment_users_kmeans(n_clusters)
    
    return jsonify({
        'success': True,
        'n_clusters': n_clusters,
        'segmentation': segmentation
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route('/attribution/track', methods=['POST'])
@require_json
@handle_errors
def track_touchpoint():
    """
    Track user touchpoint
    
    Request body:
    {
        "user_id": int,
        "touchpoint": "string",
        "touchpoint_type": "recommendation|prediction|search",
        "conversion_id": "string" (optional)
    }
    """
    from src.advanced_analytics import advanced_analytics
    
    data = request.get_json()
    user_id = data.get('user_id')
    touchpoint = data.get('touchpoint')
    touchpoint_type = data.get('touchpoint_type')
    conversion_id = data.get('conversion_id')
    
    if not all([user_id, touchpoint, touchpoint_type]):
        raise ValueError('user_id, touchpoint, and touchpoint_type are required')
    
    advanced_analytics.track_touchpoint(user_id, touchpoint, touchpoint_type, conversion_id)
    
    return jsonify({
        'success': True,
        'message': 'Touchpoint tracked'
    }), 201


@api_bp.route('/attribution/analyze/<conversion_id>', methods=['GET'])
@handle_errors
def analyze_attribution(conversion_id: str):
    """
    Analyze attribution for conversion
    
    Path parameters:
    - conversion_id: string
    
    Query parameters:
    - model: "first_touch|last_touch|linear|time_decay" (default: "linear")
    """
    from src.advanced_analytics import advanced_analytics
    
    model = request.args.get('model', 'linear')
    
    attribution = advanced_analytics.calculate_attribution(conversion_id, model)
    
    return jsonify({
        'success': True,
        'conversion_id': conversion_id,
        'attribution_model': model,
        'attribution': attribution
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# A/B TESTING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route('/ab-tests', methods=['POST'])
@require_json
@handle_errors
def create_ab_test():
    """
    Create A/B test
    
    Request body:
    {
        "test_name": "string",
        "variant_a": "string",
        "variant_b": "string"
    }
    """
    from src.advanced_recommendation_engine import advanced_recommendation_engine
    
    data = request.get_json()
    test_name = data.get('test_name')
    variant_a = data.get('variant_a')
    variant_b = data.get('variant_b')
    
    if not all([test_name, variant_a, variant_b]):
        raise ValueError('test_name, variant_a, and variant_b are required')
    
    test_id = advanced_recommendation_engine.create_ab_test(test_name, variant_a, variant_b)
    
    return jsonify({
        'success': True,
        'test_id': test_id,
        'test_name': test_name
    }), 201


@api_bp.route('/ab-tests/<int:test_id>/assign', methods=['POST'])
@require_json
@handle_errors
def assign_ab_variant(test_id: int):
    """
    Assign user to A/B test variant
    
    Path parameters:
    - test_id: int
    
    Request body:
    {
        "user_id": int
    }
    """
    from src.advanced_recommendation_engine import advanced_recommendation_engine
    
    data = request.get_json()
    user_id = data.get('user_id')
    
    if not user_id:
        raise ValueError('user_id is required')
    
    variant = advanced_recommendation_engine.assign_variant(test_id, user_id)
    
    return jsonify({
        'success': True,
        'test_id': test_id,
        'user_id': user_id,
        'variant': variant
    }), 200


@api_bp.route('/ab-tests/<int:test_id>/result', methods=['POST'])
@require_json
@handle_errors
def record_ab_result(test_id: int):
    """
    Record A/B test result
    
    Path parameters:
    - test_id: int
    
    Request body:
    {
        "user_id": int,
        "variant": "A|B",
        "metric_name": "string",
        "metric_value": float
    }
    """
    from src.advanced_recommendation_engine import advanced_recommendation_engine
    
    data = request.get_json()
    user_id = data.get('user_id')
    variant = data.get('variant')
    metric_name = data.get('metric_name')
    metric_value = data.get('metric_value')
    
    if not all([user_id, variant, metric_name, metric_value is not None]):
        raise ValueError('user_id, variant, metric_name, and metric_value are required')
    
    advanced_recommendation_engine.record_ab_test_result(
        test_id, user_id, variant, metric_name, metric_value
    )
    
    return jsonify({
        'success': True,
        'message': 'Result recorded'
    }), 201


@api_bp.route('/ab-tests/<int:test_id>/analyze', methods=['GET'])
@handle_errors
def analyze_ab_test(test_id: int):
    """
    Analyze A/B test results
    
    Path parameters:
    - test_id: int
    """
    from src.advanced_recommendation_engine import advanced_recommendation_engine
    
    results = advanced_recommendation_engine.analyze_ab_test(test_id)
    
    return jsonify({
        'success': True,
        'results': results
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH & STATUS ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200


@api_bp.route('/version', methods=['GET'])
def get_version():
    """
    Get API version
    """
    return jsonify({
        'api_version': '1.0.0',
        'app_version': '2.0.0',
        'timestamp': datetime.now().isoformat()
    }), 200
