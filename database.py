"""
IME Database Layer
==================
SQLite-based storage for parsed email records.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ime.db')


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            category_label TEXT,
            sender TEXT,
            subject TEXT,
            snippet TEXT,
            confidence INTEGER,
            record_count INTEGER,
            raw_text TEXT,
            parsed_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS tonnage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id INTEGER,
            vessel_name TEXT,
            account_name TEXT,
            open_port TEXT,
            open_date TEXT,
            vessel_type TEXT,
            vessel_size_dwt TEXT,
            built_year TEXT,
            flag TEXT,
            classification TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (email_id) REFERENCES emails(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS cargo_vc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id INTEGER,
            account_name TEXT,
            cargo_name TEXT,
            quantity TEXT,
            loading_port TEXT,
            discharge_port TEXT,
            laycan TEXT,
            cargo_type TEXT,
            commission_pct TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (email_id) REFERENCES emails(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS cargo_tc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id INTEGER,
            account_name TEXT,
            cargo_name TEXT,
            delivery_port TEXT,
            redelivery_port TEXT,
            duration TEXT,
            laycan TEXT,
            cargo_type TEXT,
            vessel_size_pref TEXT,
            commission_pct TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (email_id) REFERENCES emails(id)
        )
    ''')

    conn.commit()
    conn.close()


def save_parsed_email(result: dict, raw_text: str) -> int:
    conn = get_conn()
    c = conn.cursor()

    c.execute('''
        INSERT INTO emails (category, category_label, sender, subject, snippet,
                            confidence, record_count, raw_text, parsed_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result['category'], result['category_label'],
        result.get('sender', ''), result.get('subject', ''),
        result.get('snippet', ''), result.get('confidence', 0),
        result.get('record_count', 0), raw_text,
        json.dumps(result)
    ))
    email_id = c.lastrowid

    for rec in result.get('records', []):
        if result['category'] == 'tonnage':
            c.execute('''
                INSERT INTO tonnage (email_id, vessel_name, account_name, open_port,
                                     open_date, vessel_type, vessel_size_dwt, built_year, flag, classification)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                email_id,
                rec.get('vessel_name'), rec.get('account_name'),
                rec.get('open_port'), rec.get('open_date'),
                rec.get('vessel_type', 'Bulk Carrier'), rec.get('vessel_size_dwt'),
                rec.get('built_year'), rec.get('flag'), rec.get('classification')
            ))

        elif result['category'] == 'cargo_vc':
            c.execute('''
                INSERT INTO cargo_vc (email_id, account_name, cargo_name, quantity,
                                      loading_port, discharge_port, laycan, cargo_type, commission_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                email_id,
                rec.get('account_name'), rec.get('cargo_name'), rec.get('quantity'),
                rec.get('loading_port'), rec.get('discharge_port'),
                rec.get('laycan'), rec.get('cargo_type'), rec.get('commission_pct')
            ))

        elif result['category'] == 'cargo_tc':
            c.execute('''
                INSERT INTO cargo_tc (email_id, account_name, cargo_name, delivery_port,
                                      redelivery_port, duration, laycan, cargo_type,
                                      vessel_size_pref, commission_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                email_id,
                rec.get('account_name'), rec.get('cargo_name'),
                rec.get('delivery_port'), rec.get('redelivery_port'),
                rec.get('duration'), rec.get('laycan'),
                rec.get('cargo_type'), rec.get('vessel_size_pref'), rec.get('commission_pct')
            ))

    conn.commit()
    conn.close()
    return email_id


def get_stats():
    conn = get_conn()
    c = conn.cursor()
    stats = {}
    for tbl in ['emails', 'tonnage', 'cargo_vc', 'cargo_tc']:
        c.execute(f'SELECT COUNT(*) FROM {tbl}')
        stats[tbl] = c.fetchone()[0]
    conn.close()
    return stats


def get_all_tonnage(search=''):
    conn = get_conn()
    c = conn.cursor()
    q = '''SELECT t.*, e.sender, e.created_at as email_date, e.confidence
           FROM tonnage t JOIN emails e ON t.email_id = e.id'''
    params = []
    if search:
        q += ' WHERE t.vessel_name LIKE ? OR t.open_port LIKE ? OR t.account_name LIKE ?'
        params = [f'%{search}%'] * 3
    q += ' ORDER BY t.id DESC'
    c.execute(q, params)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_all_cargo_vc(search=''):
    conn = get_conn()
    c = conn.cursor()
    q = '''SELECT cv.*, e.sender, e.created_at as email_date, e.confidence
           FROM cargo_vc cv JOIN emails e ON cv.email_id = e.id'''
    params = []
    if search:
        q += ' WHERE cv.cargo_name LIKE ? OR cv.loading_port LIKE ? OR cv.discharge_port LIKE ?'
        params = [f'%{search}%'] * 3
    q += ' ORDER BY cv.id DESC'
    c.execute(q, params)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_all_cargo_tc(search=''):
    conn = get_conn()
    c = conn.cursor()
    q = '''SELECT ct.*, e.sender, e.created_at as email_date, e.confidence
           FROM cargo_tc ct JOIN emails e ON ct.email_id = e.id'''
    params = []
    if search:
        q += ' WHERE ct.cargo_name LIKE ? OR ct.delivery_port LIKE ? OR ct.redelivery_port LIKE ?'
        params = [f'%{search}%'] * 3
    q += ' ORDER BY ct.id DESC'
    c.execute(q, params)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_recent_emails(limit=20):
    conn = get_conn()
    c = conn.cursor()
    c.execute('''SELECT id, category, category_label, sender, subject, snippet,
                        confidence, record_count, created_at
                 FROM emails ORDER BY id DESC LIMIT ?''', (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def clear_all():
    conn = get_conn()
    c = conn.cursor()
    for tbl in ['tonnage', 'cargo_vc', 'cargo_tc', 'emails']:
        c.execute(f'DELETE FROM {tbl}')
    conn.commit()
    conn.close()
