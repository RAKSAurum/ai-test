"""
Core Memory Management System

Handles both short-term and long-term memory storage, retrieval,
and management for the AI 3D Generator pipeline.
"""

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os


@dataclass
class MemoryEntry:
    """
    Represents a single memory entry in the system.
    
    This class encapsulates all information related to a memory entry including
    generation data, file paths, metadata, and access tracking information.
    
    Attributes:
        id (str): Unique identifier for the memory entry.
        session_id (str): Session identifier this memory belongs to.
        user_id (str): User identifier who created this memory.
        timestamp (float): Unix timestamp when memory was created.
        memory_type (str): Type of memory ('generation', 'conversation', 'preference').
        original_prompt (str): Original user prompt text.
        enhanced_prompt (str): LLM-enhanced version of the prompt.
        image_path (Optional[str]): Path to generated image file.
        model_path (Optional[str]): Path to generated 3D model file.
        video_path (Optional[str]): Path to generated video file.
        metadata (Dict[str, Any]): Additional metadata dictionary.
        tags (List[str]): List of extracted tags for categorization.
        quality_score (float): Quality score for ranking (0.0-1.0).
        access_count (int): Number of times this memory has been accessed.
        last_accessed (Optional[float]): Unix timestamp of last access.
    """
    id: str
    session_id: str
    user_id: str
    timestamp: float
    memory_type: str
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

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []

    def __repr__(self):
        return f"<MemoryEntry id={self.id} prompt={self.original_prompt[:30]}...>"



@dataclass
class SessionContext:
    """
    Represents session-specific context and memory.
    
    This class manages the context and state for individual user sessions,
    including conversation history and generation statistics.
    
    Attributes:
        session_id (str): Unique session identifier.
        user_id (str): User identifier for this session.
        created_at (float): Unix timestamp when session was created.
        last_activity (float): Unix timestamp of last session activity.
        context_buffer (List[Dict[str, Any]]): Recent context items for this session.
        total_generations (int): Total number of generations in this session.
        session_metadata (Dict[str, Any]): Additional session-specific metadata.
    """
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    context_buffer: List[Dict[str, Any]]
    total_generations: int = 0
    session_metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.session_metadata is None:
            self.session_metadata = {}


class MemoryManager:
    """
    Core memory management system for the AI 3D Generator.
    
    Provides comprehensive memory management including both short-term session memory
    and long-term persistent storage with support for natural language queries,
    semantic search, and memory associations.
    
    This system is designed to work seamlessly with AI image generation workflows,
    particularly for 3D model generation and text-to-image applications. It maintains
    session context, tracks user preferences, and enables intelligent memory retrieval.
    
    Attributes:
        db_path (Path): Path to the SQLite database file.
        max_context_size (int): Maximum items in session context buffer.
        active_sessions (Dict[str, SessionContext]): Currently active user sessions.
    
    Example:
        >>> manager = MemoryManager("memory/ai_memory.db")
        >>> session_id = manager.create_session("user123")
        >>> memory_id = manager.store_generation(session_id, "user123", "robot", "futuristic robot")
        >>> memories = manager.search_memories("robot", "user123")
    """

    def __init__(self, db_path: str = "memory/ai_memory.db", max_context_size: int = 20) -> None:
        """
        Initialize the memory manager with database and session management.
        
        Sets up the SQLite database, creates necessary tables, and loads recent
        active sessions for immediate availability.
        
        Args:
            db_path (str): Path to SQLite database file. Defaults to "memory/ai_memory.db".
            max_context_size (int): Maximum number of items in session context buffer.
                Defaults to 20.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_context_size = max_context_size
        self.active_sessions: Dict[str, SessionContext] = {}
        
        # Initialize database schema
        self._init_database()
        
        # Load recent active sessions from database
        self._load_active_sessions()
        
        logging.info(f"🧠 Memory Manager initialized with database: {self.db_path}")

    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables and indexes."""
        with sqlite3.connect(self.db_path) as conn:
            # Add version tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                )
            """)
            
            # Check current version
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            current_version = cursor.fetchone()[0] or 0
            
            if current_version < 1:
                # Apply initial schema
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
                conn.execute("INSERT INTO schema_version VALUES (1, ?)", (time.time(),))
                conn.commit()

    def _load_active_sessions(self) -> None:
        """
        Load recent active sessions from database into memory.
        
        Loads sessions that have been active within the last 24 hours to
        maintain session context and enable quick access to recent user activity.
        """
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
        
        Generates a new session with unique identifier and initializes
        the session context for tracking user activity and generations.
        
        Args:
            user_id (str): Identifier for the user. Defaults to "default".
            
        Returns:
            str: New session ID for tracking user activity.
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
        Get existing session or create new one if not found.
        
        Provides session continuity by retrieving existing sessions or
        creating new ones as needed. Updates last activity timestamp.
        
        Args:
            user_id (str): User identifier. Defaults to "default".
            session_id (Optional[str]): Optional existing session ID to retrieve.
            
        Returns:
            str: Session ID (existing or newly created).
        """
        if session_id and session_id in self.active_sessions:
            # Update last activity timestamp
            self.active_sessions[session_id].last_activity = time.time()
            return session_id
        
        # Create new session if not found
        return self.create_session(user_id)

    def store_generation(self, session_id: str, user_id: str, original_prompt: str, 
                        enhanced_prompt: str, image_path: Optional[str] = None,
                        model_path: Optional[str] = None, video_path: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a generation result in memory with comprehensive metadata.
        
        Creates a new memory entry for generated content including prompts,
        file paths, extracted tags, and session context updates.
        
        Args:
            session_id (str): Session identifier for this generation.
            user_id (str): User identifier who created this generation.
            original_prompt (str): Original user prompt text.
            enhanced_prompt (str): LLM-enhanced version of the prompt.
            image_path (Optional[str]): Path to generated image file.
            model_path (Optional[str]): Path to generated 3D model file.
            video_path (Optional[str]): Path to generated video file.
            metadata (Optional[Dict[str, Any]]): Additional metadata dictionary.
            
        Returns:
            str: Unique memory entry ID for future reference.
        """
        memory_id = str(uuid.uuid4())
        current_time = time.time()
        
        # Extract relevant tags from prompts for categorization
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
        
        # Store in persistent database
        self._save_memory_entry(memory_entry)
        
        # Update session context with new generation
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.context_buffer.append({
                "type": "generation",
                "memory_id": memory_id,
                "prompt": original_prompt,
                "timestamp": current_time
            })
            
            # Maintain context buffer size limit
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
        Search memories using text-based similarity matching.
        
        Performs comprehensive text search across original prompts, enhanced prompts,
        and tags to find relevant memories. Results are ranked by quality score
        and recency.
        
        Args:
            query (str): Search query text to match against memory content.
            user_id (str): User identifier to scope search. Defaults to "default".
            limit (int): Maximum number of results to return. Defaults to 10.
            memory_type (Optional[str]): Optional filter by memory type
                ('generation', 'conversation', 'preference').
            
        Returns:
            List[MemoryEntry]: List of matching memory entries sorted by relevance
                and recency, with access counts updated.
        """
        query_lower = query.lower()
        
        with sqlite3.connect(self.db_path) as conn:
            # Build dynamic SQL query with optional filters
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
                
                # Update access count for usage tracking
                self._update_access_count(memory_entry.id)
        
        logging.info(f"🔍 Found {len(results)} memories for query: '{query}'")
        return results

    def get_recent_memories(self, user_id: str = "default", limit: int = 10, 
                           hours: int = 24) -> List[MemoryEntry]:
        """
        Get recent memories within specified time range.
        
        Retrieves memories created within the specified time window,
        useful for showing recent activity or continuing conversations.
        
        Args:
            user_id (str): User identifier to scope search. Defaults to "default".
            limit (int): Maximum number of results to return. Defaults to 10.
            hours (int): Time range in hours to look back. Defaults to 24.
            
        Returns:
            List[MemoryEntry]: List of recent memory entries sorted by timestamp
                (most recent first).
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
        
        Retrieves complete session information including context buffer,
        generation statistics, and metadata for maintaining conversational context.
        
        Args:
            session_id (str): Session identifier to retrieve.
            
        Returns:
            Optional[SessionContext]: Session context if found, None otherwise.
        """
        return self.active_sessions.get(session_id)

    def cleanup_old_sessions(self, days: int = 7) -> None:
        """
        Clean up old inactive sessions to maintain performance.
        
        Removes sessions that haven't been active within the specified
        time period from both memory and database to prevent resource bloat.
        
        Args:
            days (int): Number of days to keep sessions. Defaults to 7.
        """
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # Remove from active sessions in memory
        to_remove = []
        for session_id, session in self.active_sessions.items():
            if session.last_activity < cutoff_time:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.active_sessions[session_id]
        
        # Clean up database records
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE last_activity < ?", (cutoff_time,))
            conn.commit()
        
        logging.info(f"🧹 Cleaned up {len(to_remove)} old sessions")

    def cleanup_old_memories(self, days: int = 30) -> None:
        """Clean up memories older than specified days."""
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get files to delete
                cursor = conn.execute("""
                    SELECT image_path, model_path, video_path 
                    FROM memory_entries 
                    WHERE timestamp < ?
                """, (cutoff_time,))
                
                # Delete physical files
                for row in cursor.fetchall():
                    for path in row:
                        if path and os.path.exists(path):
                            try:
                                os.remove(path)
                                logging.info(f"Removed old file: {path}")
                            except OSError as e:
                                logging.error(f"Failed to remove file {path}: {e}")
                
                # Delete database entries
                conn.execute("DELETE FROM memory_entries WHERE timestamp < ?", (cutoff_time,))
                conn.commit()
                
        except Exception as e:
            logging.error(f"Memory cleanup failed: {e}")

    def _extract_tags(self, original_prompt: str, enhanced_prompt: str) -> List[str]:
        """
        Extract relevant tags from prompts for categorization and search.
        
        Analyzes prompt text to identify common 3D model categories, styles,
        colors, and descriptors for improved searchability and organization.
        
        Args:
            original_prompt (str): Original user prompt text.
            enhanced_prompt (str): Enhanced prompt text.
            
        Returns:
            List[str]: List of extracted tags (limited to 10 for efficiency).
        """
        combined_text = f"{original_prompt} {enhanced_prompt}".lower()
        
        # Comprehensive tag keywords for 3D model categorization
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
        
        return tags[:10]  # Limit to 10 tags for performance

    def _save_memory_entry(self, entry: MemoryEntry) -> None:
        """
        Save memory entry to persistent database storage.
        
        Args:
            entry (MemoryEntry): Memory entry object to save.
        """
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

    def _save_session(self, session: SessionContext) -> None:
        """
        Save session context to persistent database storage.
        
        Args:
            session (SessionContext): Session context object to save.
        """
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
        """
        Convert database row to MemoryEntry object.
        
        Args:
            row: SQLite row tuple from memory_entries table.
            
        Returns:
            MemoryEntry: Constructed memory entry object with parsed JSON fields.
        """
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

    def _update_access_count(self, memory_id: str) -> None:
        """
        Update access count and timestamp for a memory entry.
        
        Tracks memory usage for analytics and quality scoring.
        
        Args:
            memory_id (str): Memory entry ID to update.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE memory_entries 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """, (time.time(), memory_id))
            conn.commit()

    def _get_connection(self):
        """
        Get database connection for advanced operations.
        
        Returns:
            sqlite3.Connection: Database connection object.
        """
        return sqlite3.connect(self.db_path)