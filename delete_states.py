import sqlite3

# with sqlite3.connect('D:/projects/feedback.db') as conn:
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM user_states WHERE state = 'AWAITING_DOCUMENT'")
#     print(f"Deleted {cursor.rowcount} rows")
#     conn.commit()

# with sqlite3.connect('D:/projects/feedback.db') as conn:
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM user_states WHERE state = 'AWAITING_RATING'")
#     print(f"Deleted {cursor.rowcount} rows")
#     conn.commit()

# with sqlite3.connect('D:/projects/feedback.db') as conn:
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM user_states WHERE state = 'AWAITING_LANGUAGE'")
#     print(f"Deleted {cursor.rowcount} rows")
#     conn.commit()

import database

# Add language preference column
#database.add_language_preference_column()
    
with sqlite3.connect('D:/projects/feedback.db') as conn:
    cursor = conn.cursor()
    # cursor.execute("ALTER TABLE feedback ADD COLUMN sentiment TEXT")
    # cursor.execute("ALTER TABLE feedback ADD COLUMN intent TEXT")
    # cursor.execute("ALTER TABLE feedback ADD COLUMN toxicity_score REAL")
    cursor.execute('ALTER TABLE feedback ADD COLUMN language TEXT')
    cursor.execute('ALTER TABLE feedback ADD COLUMN entities TEXT')
    #cursor.execute("ALTER TABLE feedback ADD COLUMN embedding TEXT")
    print("Added new columns to feedback table")
    #print(f"Deleted {cursor.rowcount} rows")
    conn.commit()
    
# with sqlite3.connect('D:/projects/feedback.db') as conn:
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM multilingual_messages")
#     print(f"Deleted {cursor.rowcount} rows")
#     conn.commit()
    
# with sqlite3.connect('D:/projects/feedback.db') as conn:
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM simple_messages")
#     print(f"Deleted {cursor.rowcount} rows")
#     conn.commit()
    
# with sqlite3.connect('D:/projects/feedback.db') as conn:
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM reference_ids")
#     print(f"Deleted {cursor.rowcount} rows")
#     conn.commit()
    
# with sqlite3.connect('D:/projects/feedback.db') as conn:
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM feedback")
#     print(f"Deleted {cursor.rowcount} rows")
#     conn.commit()
    
# with sqlite3.connect('D:/projects/feedback.db') as conn:
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM user_states WHERE state = 'AWAITING_CENTER'")
#     print(f"Deleted {cursor.rowcount} rows")
#     conn.commit()

# with sqlite3.connect('D:/projects/feedback.db') as conn:
#     cursor = conn.cursor()
#     cursor.execute("DELETE FROM user_states WHERE state = 'AWAITING_COMMENT'")
#     print(f"Deleted {cursor.rowcount} rows")
#     conn.commit()
    
    
# conn = sqlite3.connect('feedback.db')
# cursor = conn.cursor()
# cursor.execute('ALTER TABLE user_states ADD COLUMN current_document_url TEXT')
# conn.commit()
# conn.close()

# def add_missing_columns():
#     conn = sqlite3.connect('feedback.db')
#     cursor = conn.cursor()
    
#     # Check if columns exist before adding them
#     try:
#         cursor.execute('ALTER TABLE user_states ADD COLUMN current_center_number TEXT')
#         print("Added current_center_number column")
#     except:
#         print("current_center_number column already exists")
        
#     try:
#         cursor.execute('ALTER TABLE user_states ADD COLUMN current_document_url TEXT')
#         print("Added current_document_url column")
#     except:
#         print("current_document_url column already exists")
    
#     conn.commit()
#     conn.close()

# add_missing_columns()

# def add_missing_columns():
#     conn = sqlite3.connect('D:/projects/feedback.db')
#     cursor = conn.cursor()
    
#     # Add all potentially missing columns
#     columns_to_add = [
#         "current_comment", 
#         "current_center_number",
#         "current_document_url",
#         "current_rating",
#         "current_ref_id"
#     ]
    
#     for column in columns_to_add:
#         try:
#             cursor.execute(f'ALTER TABLE user_states ADD COLUMN {column} TEXT')
#             print(f"Added {column} column")
#         except sqlite3.OperationalError as e:
#             if "duplicate column name" in str(e):
#                 print(f"Column {column} already exists")
#             else:
#                 print(f"Error adding {column}: {e}")
    
#     conn.commit()
#     conn.close()

# add_missing_columns()


# def add_missing_feedback_columns():
#     conn = sqlite3.connect('D:/projects/feedback.db')
#     cursor = conn.cursor()
    
#     # Add missing columns to feedback table
#     columns_to_add = [
#         "center_number",
#         "document_url",
#         "document_data"
#     ]
    
#     for column in columns_to_add:
#         try:
#             cursor.execute(f'ALTER TABLE feedback ADD COLUMN {column} TEXT')
#             print(f"Added {column} column to feedback table")
#         except sqlite3.OperationalError as e:
#             if "duplicate column name" in str(e):
#                 print(f"Column {column} already exists in feedback table")
#             else:
#                 print(f"Error adding {column}: {e}")
    
#     conn.commit()
#     conn.close()
#     print("Database update completed")

# add_missing_feedback_columns()