"""
Database Service
Production-ready SQLite database for user memory persistence
"""

import sqlite3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Production-ready database service for user memory management
    """
    
    def __init__(self, db_path: str = "guppshupp.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # User memories table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        memory_data TEXT NOT NULL,
                        message_count INTEGER DEFAULT 0,
                        confidence_score REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Conversation history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        messages TEXT NOT NULL,
                        personality_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Personality responses cache
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS personality_responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        user_message TEXT NOT NULL,
                        base_response TEXT NOT NULL,
                        personality_type TEXT NOT NULL,
                        personality_response TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Analytics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        event_type TEXT NOT NULL,
                        event_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    def create_user(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new user record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO users (id, metadata, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (user_id, json.dumps(metadata or {})))
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error creating user: {str(e)}")
            return False
    
    def save_user_memory(self, user_id: str, memory_data: Dict[str, Any], 
                        message_count: int = 0, confidence_score: float = 0.0) -> bool:
        """Save or update user memory"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if memory exists for user
                cursor.execute("SELECT id FROM user_memories WHERE user_id = ?", (user_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing memory
                    cursor.execute("""
                        UPDATE user_memories 
                        SET memory_data = ?, message_count = ?, confidence_score = ?, 
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                    """, (json.dumps(memory_data), message_count, confidence_score, user_id))
                else:
                    # Insert new memory
                    cursor.execute("""
                        INSERT INTO user_memories 
                        (user_id, memory_data, message_count, confidence_score)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, json.dumps(memory_data), message_count, confidence_score))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error saving user memory: {str(e)}")
            return False
    
    def get_user_memory(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user memory"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT memory_data, message_count, confidence_score, updated_at
                    FROM user_memories WHERE user_id = ?
                    ORDER BY updated_at DESC LIMIT 1
                """, (user_id,))
                
                result = cursor.fetchone()
                if result:
                    memory_data = json.loads(result[0])
                    memory_data['message_count'] = result[1]
                    memory_data['confidence_score'] = result[2]
                    memory_data['updated_at'] = result[3]
                    return memory_data
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving user memory: {str(e)}")
            return None
    
    def save_conversation(self, user_id: str, session_id: str, 
                         messages: List[Dict[str, str]], personality_type: str) -> bool:
        """Save conversation history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO conversations 
                    (user_id, session_id, messages, personality_type)
                    VALUES (?, ?, ?, ?)
                """, (user_id, session_id, json.dumps(messages), personality_type))
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error saving conversation: {str(e)}")
            return False
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, messages, personality_type, created_at
                    FROM conversations 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'session_id': row[0],
                        'messages': json.loads(row[1]),
                        'personality_type': row[2],
                        'created_at': row[3]
                    })
                return results
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    def save_personality_response(self, user_id: str, user_message: str, 
                                 base_response: str, personality_type: str, 
                                 personality_response: str) -> bool:
        """Save personality response for comparison"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO personality_responses 
                    (user_id, user_message, base_response, personality_type, personality_response)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, user_message, base_response, personality_type, personality_response))
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error saving personality response: {str(e)}")
            return False
    
    def get_personality_comparisons(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get personality response comparisons"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_message, base_response, personality_type, 
                           personality_response, created_at
                    FROM personality_responses 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'user_message': row[0],
                        'base_response': row[1],
                        'personality_type': row[2],
                        'personality_response': row[3],
                        'created_at': row[4]
                    })
                return results
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving personality comparisons: {str(e)}")
            return []
    
    def log_analytics_event(self, user_id: str, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Log analytics event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analytics (user_id, event_type, event_data)
                    VALUES (?, ?, ?)
                """, (user_id, event_type, json.dumps(event_data)))
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error logging analytics event: {str(e)}")
            return False
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Memory stats
                cursor.execute("""
                    SELECT COUNT(*), AVG(confidence_score), SUM(message_count)
                    FROM user_memories WHERE user_id = ?
                """, (user_id,))
                memory_stats = cursor.fetchone()
                
                # Conversation stats
                cursor.execute("""
                    SELECT COUNT(*), COUNT(DISTINCT personality_type)
                    FROM conversations WHERE user_id = ?
                """, (user_id,))
                conv_stats = cursor.fetchone()
                
                # Personality response stats
                cursor.execute("""
                    SELECT COUNT(*), COUNT(DISTINCT personality_type)
                    FROM personality_responses WHERE user_id = ?
                """, (user_id,))
                personality_stats = cursor.fetchone()
                
                return {
                    'memory_extractions': memory_stats[0] or 0,
                    'avg_confidence': round(memory_stats[1] or 0, 2),
                    'total_messages_analyzed': memory_stats[2] or 0,
                    'total_conversations': conv_stats[0] or 0,
                    'personalities_tried': conv_stats[1] or 0,
                    'personality_comparisons': personality_stats[0] or 0,
                    'unique_personalities_tested': personality_stats[1] or 0
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error getting user stats: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days: int = 30) -> bool:
        """Clean up old data to manage database size"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now().replace(microsecond=0) - timedelta(days=days)
                
                # Clean old conversations
                cursor.execute("""
                    DELETE FROM conversations 
                    WHERE created_at < ?
                """, (cutoff_date,))
                
                # Clean old personality responses
                cursor.execute("""
                    DELETE FROM personality_responses 
                    WHERE created_at < ?
                """, (cutoff_date,))
                
                # Clean old analytics
                cursor.execute("""
                    DELETE FROM analytics 
                    WHERE created_at < ?
                """, (cutoff_date,))
                
                conn.commit()
                logger.info(f"Cleaned up data older than {days} days")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error backing up database: {str(e)}")
            return False


class DatabaseError(Exception):
    """Custom exception for database errors"""
    pass
