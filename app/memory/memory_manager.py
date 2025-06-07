"""
Core Memory Management System

Handles both short-term and long-term memory storage, retrieval,
and management for the AI 3D Generator pipeline.
"""

import sqlite3
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import hashlib

@dataclass
class MemoryEntry:
    """Represents a single memory entry in the system."""
    id: str
    session_id: str
    user_id: str
    timestamp: float
    memory_type: str  # 'generation', 'conversation', 'preference'
    original_prompt: str
    enhanced_prompt: str
    image_path: Optional[str] = None
    model_path: Optional[str] = None
    video_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    tags: List[str] = None
    quality_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []

@dataclass
class SessionContext:
    """Represents session-specific context and memory."""
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    context_buffer: List[Dict[str, Any]]
    total_generations: int = 0
    session_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.session_metadata is None:
            self.session_metadata = {}

class MemoryManager:
    """
    Core memory management system for the AI 3D Generator.
    
    Provides both short-term session memory and long-term persistent storage
    with support for natural language queries and semantic search.
    """

    def __init__(self, db_path: str = "memory/ai_memory.db", max_context_size: int = 20):
        """
        Initialize the memory manager.
        
        Args:
            db_path: Path to SQLite database file
            max_context_size: Maximum number of items in session context buffer
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_context_size = max_context_size
        self.active_sessions: Dict[str, SessionContext] = {}
        
        # Initialize database
        self._init_database()
        
        # Load active sessions from database
        self._load_active_sessions()
        
        logging.info(f"🧠 Memory Manager initialized with database: {self.db_path}")

    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    memory_type TEXT NOT NULL,
                    original_prompt TEXT NOT NULL,
                    enhanced_prompt TEXT NOT NULL,
                    image_path TEXT,
                    model_path TEXT,
                    video_path TEXT,
                    metadata TEXT,
                    tags TEXT,
                    quality_score REAL DEFAULT 0.0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_activity REAL NOT NULL,
                    context_buffer TEXT,
                    total_generations INTEGER DEFAULT 0,
                    session_metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS memory_associations (
                    id TEXT PRIMARY KEY,
                    memory_id_1 TEXT NOT NULL,
                    memory_id_2 TEXT NOT NULL,
                    association_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (memory_id_1) REFERENCES memory_entries (id),
                    FOREIGN KEY (memory_id_2) REFERENCES memory_entries (id)
                );

                CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_entries (timestamp);
                CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_entries (session_id);
                CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_entries (user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries (memory_type);
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_activity ON sessions (last_activity);
            """)
            conn.commit()

    def _load_active_sessions(self):
        """Load recent active sessions from database."""
        cutoff_time = time.time() - (24 * 60 * 60)  # Last 24 hours
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT session_id, user_id, created_at, last_activity, 
                       context_buffer, total_generations, session_metadata
                FROM sessions 
                WHERE last_activity > ?
                ORDER BY last_activity DESC
                LIMIT 10
            """, (cutoff_time,))
            
            for row in cursor.fetchall():
                session_id, user_id, created_at, last_activity, context_buffer, total_generations, session_metadata = row
                
                context_data = json.loads(context_buffer) if context_buffer else []
                metadata = json.loads(session_metadata) if session_metadata else {}
                
                session = SessionContext(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=created_at,
                    last_activity=last_activity,
                    context_buffer=context_data,
                    total_generations=total_generations,
                    session_metadata=metadata
                )
                
                self.active_sessions[session_id] = session

    def create_session(self, user_id: str = "default") -> str:
        """
        Create a new session for the user.
        
        Args:
            user_id: Identifier for the user
            
        Returns:
            str: New session ID
        """
        session_id = str(uuid.uuid4())
        current_time = time.time()
        
        session = SessionContext(
            session_id=session_id,
            user_id=user_id,
            created_at=current_time,
            last_activity=current_time,
            context_buffer=[],
            total_generations=0
        )
        
        self.active_sessions[session_id] = session
        self._save_session(session)
        
        logging.info(f"🆕 Created new session: {session_id} for user: {user_id}")
        return session_id

    def get_or_create_session(self, user_id: str = "default", session_id: Optional[str] = None) -> str:
        """
        Get existing session or create new one.
        
        Args:
            user_id: User identifier
            session_id: Optional existing session ID
            
        Returns:
            str: Session ID (existing or new)
        """
        if session_id and session_id in self.active_sessions:
            # Update last activity
            self.active_sessions[session_id].last_activity = time.time()
            return session_id
        
        # Create new session
        return self.create_session(user_id)

    def store_generation(self, session_id: str, user_id: str, original_prompt: str, 
                        enhanced_prompt: str, image_path: Optional[str] = None,
                        model_path: Optional[str] = None, video_path: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a generation result in memory.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            original_prompt: Original user prompt
            enhanced_prompt: LLM-enhanced prompt
            image_path: Path to generated image
            model_path: Path to generated 3D model
            video_path: Path to generated video
            metadata: Additional metadata
            
        Returns:
            str: Memory entry ID
        """
        memory_id = str(uuid.uuid4())
        current_time = time.time()
        
        # Extract tags from prompts
        tags = self._extract_tags(original_prompt, enhanced_prompt)
        
        memory_entry = MemoryEntry(
            id=memory_id,
            session_id=session_id,
            user_id=user_id,
            timestamp=current_time,
            memory_type="generation",
            original_prompt=original_prompt,
            enhanced_prompt=enhanced_prompt,
            image_path=image_path,
            model_path=model_path,
            video_path=video_path,
            metadata=metadata or {},
            tags=tags,
            quality_score=1.0  # Initial quality score
        )
        
        # Store in database
        self._save_memory_entry(memory_entry)
        
        # Update session context
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.context_buffer.append({
                "type": "generation",
                "memory_id": memory_id,
                "prompt": original_prompt,
                "timestamp": current_time
            })
            
            # Maintain context buffer size
            if len(session.context_buffer) > self.max_context_size:
                session.context_buffer = session.context_buffer[-self.max_context_size:]
            
            session.total_generations += 1
            session.last_activity = current_time
            self._save_session(session)
        
        logging.info(f"💾 Stored generation memory: {memory_id}")
        return memory_id

    def search_memories(self, query: str, user_id: str = "default", 
                       limit: int = 10, memory_type: Optional[str] = None) -> List[MemoryEntry]:
        """
        Search memories using text-based similarity.
        
        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum number of results
            memory_type: Optional filter by memory type
            
        Returns:
            List[MemoryEntry]: Matching memory entries
        """
        query_lower = query.lower()
        
        with sqlite3.connect(self.db_path) as conn:
            # Build SQL query
            sql = """
                SELECT * FROM memory_entries 
                WHERE user_id = ? AND (
                    LOWER(original_prompt) LIKE ? OR 
                    LOWER(enhanced_prompt) LIKE ? OR
                    LOWER(tags) LIKE ?
                )
            """
            params = [user_id, f"%{query_lower}%", f"%{query_lower}%", f"%{query_lower}%"]
            
            if memory_type:
                sql += " AND memory_type = ?"
                params.append(memory_type)
            
            sql += " ORDER BY quality_score DESC, timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(sql, params)
            results = []
            
            for row in cursor.fetchall():
                memory_entry = self._row_to_memory_entry(row)
                results.append(memory_entry)
                
                # Update access count
                self._update_access_count(memory_entry.id)
        
        logging.info(f"🔍 Found {len(results)} memories for query: '{query}'")
        return results

    def get_recent_memories(self, user_id: str = "default", limit: int = 10, 
                           hours: int = 24) -> List[MemoryEntry]:
        """
        Get recent memories within specified time range.
        
        Args:
            user_id: User identifier
            limit: Maximum number of results
            hours: Time range in hours
            
        Returns:
            List[MemoryEntry]: Recent memory entries
        """
        cutoff_time = time.time() - (hours * 60 * 60)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM memory_entries 
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (user_id, cutoff_time, limit))
            
            results = []
            for row in cursor.fetchall():
                memory_entry = self._row_to_memory_entry(row)
                results.append(memory_entry)
        
        return results

    def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """
        Get session context for the given session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Optional[SessionContext]: Session context if found
        """
        return self.active_sessions.get(session_id)

    def cleanup_old_sessions(self, days: int = 7):
        """
        Clean up old inactive sessions.
        
        Args:
            days: Number of days to keep sessions
        """
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # Remove from active sessions
        to_remove = []
        for session_id, session in self.active_sessions.items():
            if session.last_activity < cutoff_time:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.active_sessions[session_id]
        
        # Clean up database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE last_activity < ?", (cutoff_time,))
            conn.commit()
        
        logging.info(f"🧹 Cleaned up {len(to_remove)} old sessions")

    def _extract_tags(self, original_prompt: str, enhanced_prompt: str) -> List[str]:
        """Extract relevant tags from prompts."""
        combined_text = f"{original_prompt} {enhanced_prompt}".lower()
        
        # Common 3D model categories and descriptors
        tag_keywords = {
            'robot', 'dragon', 'castle', 'house', 'car', 'vehicle', 'animal', 'character',
            'futuristic', 'medieval', 'steampunk', 'cyberpunk', 'fantasy', 'sci-fi',
            'blue', 'red', 'green', 'gold', 'silver', 'metallic', 'glowing',
            'mechanical', 'organic', 'architectural', 'creature', 'weapon', 'tool'
        }
        
        tags = []
        for keyword in tag_keywords:
            if keyword in combined_text:
                tags.append(keyword)
        
        return tags[:10]  # Limit to 10 tags

    def _save_memory_entry(self, entry: MemoryEntry):
        """Save memory entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memory_entries (
                    id, session_id, user_id, timestamp, memory_type,
                    original_prompt, enhanced_prompt, image_path, model_path, video_path,
                    metadata, tags, quality_score, access_count, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, entry.session_id, entry.user_id, entry.timestamp, entry.memory_type,
                entry.original_prompt, entry.enhanced_prompt, entry.image_path, 
                entry.model_path, entry.video_path,
                json.dumps(entry.metadata), json.dumps(entry.tags),
                entry.quality_score, entry.access_count, entry.last_accessed
            ))
            conn.commit()

    def _save_session(self, session: SessionContext):
        """Save session to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions (
                    session_id, user_id, created_at, last_activity,
                    context_buffer, total_generations, session_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id, session.user_id, session.created_at, session.last_activity,
                json.dumps(session.context_buffer), session.total_generations,
                json.dumps(session.session_metadata)
            ))
            conn.commit()

    def _row_to_memory_entry(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry object."""
        (id, session_id, user_id, timestamp, memory_type, original_prompt, enhanced_prompt,
         image_path, model_path, video_path, metadata, tags, quality_score, access_count, last_accessed) = row
        
        return MemoryEntry(
            id=id,
            session_id=session_id,
            user_id=user_id,
            timestamp=timestamp,
            memory_type=memory_type,
            original_prompt=original_prompt,
            enhanced_prompt=enhanced_prompt,
            image_path=image_path,
            model_path=model_path,
            video_path=video_path,
            metadata=json.loads(metadata) if metadata else {},
            tags=json.loads(tags) if tags else [],
            quality_score=quality_score,
            access_count=access_count,
            last_accessed=last_accessed
        )

    def _update_access_count(self, memory_id: str):
        """Update access count for a memory entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE memory_entries 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """, (time.time(), memory_id))
            conn.commit()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)