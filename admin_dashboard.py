from flask import Blueprint, render_template, jsonify, request
import database
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/')
def index():
    """Admin dashboard home page"""
    return render_template('admin/dashboard.html')

@admin_bp.route('/analytics')
def analytics():
    """Get feedback analytics data"""
    days = request.args.get('days', default=30, type=int)
    feedback_data = database.get_all_feedback(days)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(feedback_data)
    
    # Basic statistics
    if len(df) > 0:
        avg_rating = df['rating'].mean()
        rating_counts = df['rating'].value_counts().sort_index().to_dict()
        total_feedback = len(df)
        
        # Generate rating distribution chart
        plt.figure(figsize=(10, 6))
        plt.bar(rating_counts.keys(), rating_counts.values())
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.75)
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            'avg_rating': avg_rating,
            'rating_distribution': rating_counts,
            'total_feedback': total_feedback,
            'chart': img_str
        })
    else:
        return jsonify({
            'avg_rating': 0,
            'rating_distribution': {},
            'total_feedback': 0,
            'chart': None
        })

@admin_bp.route('/feedback')
def feedback_list():
    """List all feedback with filtering options"""
    days = request.args.get('days', default=None, type=int)
    feedback_data = database.get_all_feedback(days)
    return jsonify(feedback_data)

@admin_bp.route('/district-collector')
def district_collector():
    """District Collector's view with critical issues"""
    feedback_data = database.get_all_feedback(7)  # Last 7 days
    
    # Filter for critical feedback (rating <= 2)
    critical_feedback = [f for f in feedback_data if f['rating'] <= 2]
    
    return jsonify(critical_feedback)