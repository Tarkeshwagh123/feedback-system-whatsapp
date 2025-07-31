import sqlite3
from datetime import datetime

def init_db():
    """Initialize the database tables"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    
    # Create reference IDs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reference_ids (
        id INTEGER PRIMARY KEY,
        ref_id TEXT UNIQUE NOT NULL,
        citizen_contact TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY,
        ref_id TEXT NOT NULL,
        citizen_contact TEXT NOT NULL,
        rating INTEGER NOT NULL,
        comment TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ref_id) REFERENCES reference_ids (ref_id)
    )
    ''')
    
    # Create user state table for tracking conversation state
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_states (
        citizen_contact TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        current_ref_id TEXT,
        current_rating INTEGER,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def add_reference_id(ref_id, citizen_contact):
    """Add a new reference ID linked to a citizen"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO reference_ids (ref_id, citizen_contact) VALUES (?, ?)',
        (ref_id, citizen_contact)
    )
    conn.commit()
    conn.close()

def validate_reference_id(ref_id, citizen_contact):
    """Validate if the reference ID exists and belongs to the citizen"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT 1 FROM reference_ids WHERE ref_id = ? AND citizen_contact = ?',
        (ref_id, citizen_contact)
    )
    result = cursor.fetchone()
    conn.close()
    return result is not None

def get_user_state(citizen_contact):
    """Get the current state of a user's conversation"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT state FROM user_states WHERE citizen_contact = ?',
        (citizen_contact,)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def set_user_state(citizen_contact, state):
    """Set the current state of a user's conversation"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO user_states (citizen_contact, state, updated_at) 
        VALUES (?, ?, CURRENT_TIMESTAMP) 
        ON CONFLICT(citizen_contact) 
        DO UPDATE SET state = ?, updated_at = CURRENT_TIMESTAMP''',
        (citizen_contact, state, state)
    )
    conn.commit()
    conn.close()

def set_current_ref_id(citizen_contact, ref_id):
    """Set the current reference ID being processed"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE user_states SET current_ref_id = ? WHERE citizen_contact = ?',
        (ref_id, citizen_contact)
    )
    conn.commit()
    conn.close()

def get_current_ref_id(citizen_contact):
    """Get the current reference ID being processed"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT current_ref_id FROM user_states WHERE citizen_contact = ?',
        (citizen_contact,)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def set_current_rating(citizen_contact, rating):
    """Set the current rating being processed"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE user_states SET current_rating = ? WHERE citizen_contact = ?',
        (rating, citizen_contact)
    )
    conn.commit()
    conn.close()

def get_current_rating(citizen_contact):
    """Get the current rating being processed"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT current_rating FROM user_states WHERE citizen_contact = ?',
        (citizen_contact,)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def save_feedback(citizen_contact, ref_id, rating, comment):
    """Save feedback to the database"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO feedback (ref_id, citizen_contact, rating, comment) VALUES (?, ?, ?, ?)',
        (ref_id, citizen_contact, rating, comment)
    )
    conn.commit()
    conn.close()

def get_all_feedback(days=None):
    """Get all feedback, optionally filtered by days"""
    conn = sqlite3.connect('feedback.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = 'SELECT * FROM feedback'
    params = ()
    
    if days:
        query += ' WHERE created_at >= datetime("now", ?)'
        params = (f'-{days} day',)
        
    query += ' ORDER BY created_at DESC'
    
    cursor.execute(query, params)
    result = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return result