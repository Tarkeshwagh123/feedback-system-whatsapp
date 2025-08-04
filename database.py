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
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS multilingual_messages (
        id INTEGER PRIMARY KEY,
        marathi_content TEXT,
        english_content TEXT,
        sender TEXT,
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
        center_number TEXT,
        document_url TEXT,
        document_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ref_id) REFERENCES reference_ids (ref_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS simple_messages (
        id INTEGER PRIMARY KEY,
        sender_name TEXT,
        phone_number TEXT NOT NULL,
        message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create user state table for tracking conversation state
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_states (
        citizen_contact TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        current_ref_id TEXT,
        current_rating INTEGER,
        current_comment TEXT,
        current_center_number TEXT,
        current_document_url TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    
def set_current_comment(citizen_contact, comment):
    """Set the current comment being processed"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE user_states SET current_comment = ? WHERE citizen_contact = ?',
        (comment, citizen_contact)
    )
    conn.commit()
    conn.close()
    
def save_simple_message(sender_name, phone_number, message):
    """Save a simple message to the database"""
    with sqlite3.connect('feedback.db') as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO simple_messages (sender_name, phone_number, message) VALUES (?, ?, ?)',
            (sender_name, phone_number, message)
        )
        conn.commit()
        
def save_multilingual_message(marathi_content, english_content, sender):
    """Save a multilingual message to the database"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO multilingual_messages (marathi_content, english_content, sender) VALUES (?, ?, ?)',
        (marathi_content, english_content, sender)
    )
    conn.commit()
    conn.close()

def get_multilingual_messages():
    """Get all multilingual messages"""
    conn = sqlite3.connect('feedback.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM multilingual_messages ORDER BY created_at DESC')
    result = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return result

def get_multilingual_message_by_english(english_content):
    """Get a multilingual message by its English content"""
    conn = sqlite3.connect('feedback.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM multilingual_messages WHERE english_content = ? LIMIT 1',
        (english_content,)
    )
    result = cursor.fetchone()
    conn.close()
    return dict(result) if result else None

def get_current_comment(citizen_contact):
    """Get the current comment being processed"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT current_comment FROM user_states WHERE citizen_contact = ?',
        (citizen_contact,)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def set_current_center_number(citizen_contact, center_number):
    """Set the current center number being processed"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE user_states SET current_center_number = ? WHERE citizen_contact = ?',
        (center_number, citizen_contact)
    )
    conn.commit()
    conn.close()

def get_current_center_number(citizen_contact):
    """Get the current center number being processed"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT current_center_number FROM user_states WHERE citizen_contact = ?',
        (citizen_contact,)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def set_current_document_url(citizen_contact, document_url):
    """Set the current document URL being processed"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE user_states SET current_document_url = ? WHERE citizen_contact = ?',
        (document_url, citizen_contact)
    )
    conn.commit()
    conn.close()

def get_current_document_url(citizen_contact):
    """Get the current document URL being processed"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT current_document_url FROM user_states WHERE citizen_contact = ?',
        (citizen_contact,)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def save_feedback_with_document(citizen_contact, ref_id, rating, comment, center_number, document_url, document_data=None):
    """Save feedback with document information to the database"""
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO feedback (ref_id, citizen_contact, rating, comment, center_number, document_url, document_data) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (ref_id, citizen_contact, rating, comment, center_number, document_url, document_data)
    )
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

def clear_user_states():
    """Remove all entries from user_states table"""
    with sqlite3.connect('feedback.db') as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM user_states')
        conn.commit()

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