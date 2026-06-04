"""
IME - Integrated Maritime Exchange
Email Intelligence Platform
============================
Flask web application - Enterprise Edition
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from parser import parse_email
from database import (init_db, save_parsed_email, get_stats, get_all_tonnage,
                      get_all_cargo_vc, get_all_cargo_tc, get_recent_emails, clear_all)
from seed_data import SAMPLE_EMAILS
from ml_classifier import train_classifier, predict_with_model, get_model_metrics
from matching import get_vessel_cargo_matches
import json

app = Flask(__name__)
app.secret_key = 'ime-enterprise-2026'

# Jinja2 enumerate filter
app.jinja_env.globals['enumerate'] = enumerate


# ─────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    stats = get_stats()
    recent = get_recent_emails(10)
    return render_template('index.html', stats=stats, recent=recent)

@app.route('/email-intelligence')
def email_intelligence():
    stats = get_stats()
    recent = get_recent_emails(20)
    return render_template('email_intelligence.html', stats=stats, recent=recent)

@app.route('/email-processing')
def email_processing():
    return render_template('email_processing.html')

@app.route('/tonnage')
def tonnage():
    search = request.args.get('q', '')
    rows = get_all_tonnage(search)
    return render_template('tonnage.html', rows=rows, search=search, count=len(rows))

@app.route('/cargo-vc')
def cargo_vc():
    search = request.args.get('q', '')
    rows = get_all_cargo_vc(search)
    return render_template('cargo_vc.html', rows=rows, search=search, count=len(rows))

@app.route('/cargo-tc')
def cargo_tc():
    search = request.args.get('q', '')
    rows = get_all_cargo_tc(search)
    return render_template('cargo_tc.html', rows=rows, search=search, count=len(rows))

@app.route('/match-engine')
def match_engine():
    matches = get_vessel_cargo_matches()
    return render_template('match_engine.html', matches=matches)

@app.route('/analytics')
def analytics():
    stats = get_stats()
    recent = get_recent_emails(100)
    return render_template('analytics.html', stats=stats, recent=recent)

@app.route('/ml-classifier')
def ml_classifier():
    metrics = get_model_metrics()
    return render_template('ml_classifier.html', metrics=metrics)


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route('/parse', methods=['POST'])
def parse():
    data = request.get_json()
    text = data.get('text', '')
    sender = data.get('sender', '')
    subject = data.get('subject', '')
    if not text.strip():
        return jsonify({'error': 'No email text provided'}), 400
    result = parse_email(text, sender, subject)
    email_id = save_parsed_email(result, text)
    result['email_id'] = email_id
    return jsonify(result)

@app.route('/api/ml-predict', methods=['POST'])
def ml_predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400
    prediction = predict_with_model(text)
    return jsonify(prediction)

@app.route('/api/ml-train', methods=['POST'])
def ml_train():
    result = train_classifier()
    return jsonify(result)

@app.route('/api/ml-metrics')
def ml_metrics():
    """Read-only — returns cached metrics without retraining."""
    return jsonify(get_model_metrics())

@app.route('/api/matches')
def api_matches():
    matches = get_vessel_cargo_matches()
    return jsonify(matches)

@app.route('/api/stats')
def api_stats():
    return jsonify(get_stats())

@app.route('/api/tonnage')
def api_tonnage():
    search = request.args.get('q', '')
    return jsonify(get_all_tonnage(search))

@app.route('/api/cargo-vc')
def api_cargo_vc():
    search = request.args.get('q', '')
    return jsonify(get_all_cargo_vc(search))

@app.route('/api/cargo-tc')
def api_cargo_tc():
    search = request.args.get('q', '')
    return jsonify(get_all_cargo_tc(search))

@app.route('/api/recent')
def api_recent():
    limit = int(request.args.get('limit', 20))
    return jsonify(get_recent_emails(limit))

@app.route('/seed', methods=['POST'])
def seed():
    count = 0
    for email in SAMPLE_EMAILS:
        result = parse_email(email['text'], email.get('sender', ''), email.get('subject', ''))
        save_parsed_email(result, email['text'])
        count += 1
    train_classifier()
    return jsonify({'loaded': count, 'message': f'Loaded {count} sample emails and trained ML model'})

@app.route('/clear', methods=['POST'])
def clear():
    clear_all()
    return jsonify({'message': 'All data cleared'})


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5050, host='0.0.0.0', use_reloader=True, extra_files=[])
