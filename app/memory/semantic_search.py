"""
Semantic Search and Natural Language Processing

Provides advanced search capabilities using vector embeddings and
natural language understanding for memory queries.
"""

import numpy as np
import sqlite3
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import re
from pathlib import Path

# Global flag for embeddings availability - define at module level
EMBEDDINGS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
    logging.info("✅ Sentence transformers and FAISS available")
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False
    logging.warning(f"⚠️ Sentence transformers not available: {e}. Using fallback search.")

@dataclass
class SearchResult:
    """Represents a search result with relevance scoring."""
    memory_id: str
    relevance_score: float
    match_type: str  # 'semantic', 'keyword', 'temporal'
    matched_content: str

class SemanticSearch:
    """
    Advanced semantic search system for memory queries.
    
    Provides vector-based similarity search with fallback to keyword matching
    when embedding models are not available.
    """

    def __init__(self, db_path: str = "memory/ai_memory.db"):
        """
        Initialize semantic search system.
        
        Args:
            db_path: Path to the memory database
        """
        self.db_path = Path(db_path)
        self.model = None
        self.index = None
        self.memory_vectors = {}
        
        # Initialize embedding model if available
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
                self._initialize_vector_index()
                logging.info("🔍 Semantic search initialized with embeddings")
            except Exception as e:
                logging.warning(f"⚠️ Failed to load embedding model: {e}")
                # Don't modify global variable, just set instance flag
                self.embeddings_available = False
        else:
            self.embeddings_available = False
        
        if not EMBEDDINGS_AVAILABLE:
            logging.info("🔍 Semantic search initialized with keyword fallback")

    def _initialize_vector_index(self):
        """Initialize FAISS vector index for similarity search."""
        try:
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            
            # Load existing embeddings from database
            self._load_existing_embeddings()
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize vector index: {e}")
            self.index = None

    def _load_existing_embeddings(self):
        """Load existing memory embeddings from database."""
        if not self.db_path.exists():
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if embeddings table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='memory_embeddings'
                """)
                
                if not cursor.fetchone():
                    # Create embeddings table
                    conn.execute("""
                        CREATE TABLE memory_embeddings (
                            memory_id TEXT PRIMARY KEY,
                            embedding BLOB NOT NULL,
                            created_at REAL NOT NULL,
                            FOREIGN KEY (memory_id) REFERENCES memory_entries (id)
                        )
                    """)
                    conn.commit()
                    return
                
                # Load existing embeddings
                cursor = conn.execute("SELECT memory_id, embedding FROM memory_embeddings")
                vectors = []
                memory_ids = []
                
                for memory_id, embedding_blob in cursor.fetchall():
                    try:
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        vectors.append(embedding)
                        memory_ids.append(memory_id)
                        self.memory_vectors[memory_id] = len(vectors) - 1
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to load embedding for {memory_id}: {e}")
                
                if vectors:
                    vectors_array = np.array(vectors).astype('float32')
                    # Normalize for cosine similarity
                    faiss.normalize_L2(vectors_array)
                    self.index.add(vectors_array)
                    logging.info(f"📚 Loaded {len(vectors)} existing embeddings")
                
        except Exception as e:
            logging.error(f"❌ Failed to load existing embeddings: {e}")

    def add_memory_embedding(self, memory_id: str, text_content: str):
        """
        Add embedding for a new memory entry.
        
        Args:
            memory_id: Unique memory identifier
            text_content: Text content to embed
        """
        if not EMBEDDINGS_AVAILABLE or not self.model:
            return
        
        try:
            # Generate embedding
            embedding = self.model.encode([text_content])[0].astype('float32')
            
            # Normalize for cosine similarity
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            # Add to FAISS index
            if self.index is not None:
                self.index.add(np.array([embedding_norm]))
                self.memory_vectors[memory_id] = self.index.ntotal - 1
            
            # Store in database
            self._store_embedding(memory_id, embedding_norm)
            
            logging.debug(f"📝 Added embedding for memory: {memory_id}")
            
        except Exception as e:
            logging.error(f"❌ Failed to add embedding for {memory_id}: {e}")

    def _store_embedding(self, memory_id: str, embedding: np.ndarray):
        """Store embedding in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding, created_at)
                    VALUES (?, ?, ?)
                """, (memory_id, embedding.tobytes(), time.time()))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to store embedding: {e}")

    def semantic_search(self, query: str, user_id: str = "default", 
                       limit: int = 10, threshold: float = 0.3) -> List[SearchResult]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query: Search query text
            user_id: User identifier for filtering
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List[SearchResult]: Ranked search results
        """
        if not EMBEDDINGS_AVAILABLE or not self.model or not self.index:
            return self._fallback_keyword_search(query, user_id, limit)
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0].astype('float32')
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(np.array([query_embedding]), limit * 2)
            
            # Get memory IDs and filter by user
            results = []
            memory_id_to_index = {v: k for k, v in self.memory_vectors.items()}
            
            with sqlite3.connect(self.db_path) as conn:
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1 or score < threshold:
                        continue
                    
                    memory_id = memory_id_to_index.get(idx)
                    if not memory_id:
                        continue
                    
                    # Check if memory belongs to user
                    cursor = conn.execute("""
                        SELECT original_prompt, enhanced_prompt FROM memory_entries 
                        WHERE id = ? AND user_id = ?
                    """, (memory_id, user_id))
                    
                    row = cursor.fetchone()
                    if row:
                        original_prompt, enhanced_prompt = row
                        results.append(SearchResult(
                            memory_id=memory_id,
                            relevance_score=float(score),
                            match_type='semantic',
                            matched_content=original_prompt
                        ))
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logging.error(f"❌ Semantic search failed: {e}")
            return self._fallback_keyword_search(query, user_id, limit)

    def _fallback_keyword_search(self, query: str, user_id: str = "default", 
                                limit: int = 10) -> List[SearchResult]:
        """
        Fallback keyword-based search when embeddings are unavailable.
        
        Args:
            query: Search query text
            user_id: User identifier
            limit: Maximum number of results
            
        Returns:
            List[SearchResult]: Keyword-based search results
        """
        try:
            query_terms = self._extract_keywords(query)
            if not query_terms:
                return []
            
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                # Build search query
                search_conditions = []
                params = [user_id]
                
                for term in query_terms:
                    search_conditions.append("""
                        (LOWER(original_prompt) LIKE ? OR 
                         LOWER(enhanced_prompt) LIKE ? OR 
                         LOWER(tags) LIKE ?)
                    """)
                    term_pattern = f"%{term.lower()}%"
                    params.extend([term_pattern, term_pattern, term_pattern])
                
                sql = f"""
                    SELECT id, original_prompt, enhanced_prompt, tags,
                           (access_count + 1) * quality_score as relevance
                    FROM memory_entries 
                    WHERE user_id = ? AND ({' OR '.join(search_conditions)})
                    ORDER BY relevance DESC, timestamp DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                
                for row in cursor.fetchall():
                    memory_id, original_prompt, enhanced_prompt, tags, relevance = row
                    
                    # Calculate keyword match score
                    match_score = self._calculate_keyword_score(query_terms, original_prompt, enhanced_prompt, tags)
                    
                    results.append(SearchResult(
                        memory_id=memory_id,
                        relevance_score=match_score,
                        match_type='keyword',
                        matched_content=original_prompt
                    ))
            
            return results
            
        except Exception as e:
            logging.error(f"❌ Keyword search failed: {e}")
            return []

    def temporal_search(self, user_id: str, time_range: Tuple[float, float], 
                       limit: int = 10) -> List[SearchResult]:
        """
        Search memories within a specific time range.
        
        Args:
            user_id: User identifier
            time_range: Tuple of (start_time, end_time) timestamps
            limit: Maximum number of results
            
        Returns:
            List[SearchResult]: Time-filtered results
        """
        try:
            start_time, end_time = time_range
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, original_prompt, timestamp, quality_score
                    FROM memory_entries 
                    WHERE user_id = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, start_time, end_time, limit))
                
                for row in cursor.fetchall():
                    memory_id, original_prompt, timestamp, quality_score = row
                    
                    # Score based on recency and quality
                    recency_score = 1.0 - ((time.time() - timestamp) / (24 * 60 * 60))  # Decay over 24 hours
                    combined_score = (quality_score + max(0, recency_score)) / 2
                    
                    results.append(SearchResult(
                        memory_id=memory_id,
                        relevance_score=combined_score,
                        match_type='temporal',
                        matched_content=original_prompt
                    ))
            
            return results
            
        except Exception as e:
            logging.error(f"❌ Temporal search failed: {e}")
            return []

    def hybrid_search(self, query: str, user_id: str = "default", 
                     time_range: Optional[Tuple[float, float]] = None,
                     entity_filters: Optional[Dict[str, List[str]]] = None,
                     limit: int = 10) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic, temporal, and entity filtering.
        
        Args:
            query: Search query text
            user_id: User identifier
            time_range: Optional time range filter
            entity_filters: Optional entity-based filters
            limit: Maximum number of results
            
        Returns:
            List[SearchResult]: Combined search results
        """
        all_results = []
        
        # Semantic/keyword search
        semantic_results = self.semantic_search(query, user_id, limit * 2)
        all_results.extend(semantic_results)
        
        # Temporal search if time range specified
        if time_range:
            temporal_results = self.temporal_search(user_id, time_range, limit)
            all_results.extend(temporal_results)
        
        # Remove duplicates and combine scores
        unique_results = {}
        for result in all_results:
            if result.memory_id in unique_results:
                # Combine scores from different search methods
                existing = unique_results[result.memory_id]
                combined_score = (existing.relevance_score + result.relevance_score) / 2
                existing.relevance_score = combined_score
                existing.match_type = f"{existing.match_type}+{result.match_type}"
            else:
                unique_results[result.memory_id] = result
        
        # Apply entity filters if specified
        if entity_filters:
            filtered_results = self._apply_entity_filters(
                list(unique_results.values()), entity_filters, user_id
            )
        else:
            filtered_results = list(unique_results.values())
        
        # Sort by relevance and return top results
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return filtered_results[:limit]

    def _apply_entity_filters(self, results: List[SearchResult], 
                             entity_filters: Dict[str, List[str]], 
                             user_id: str) -> List[SearchResult]:
        """Apply entity-based filtering to search results."""
        if not entity_filters:
            return results
        
        filtered_results = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for result in results:
                    cursor = conn.execute("""
                        SELECT original_prompt, enhanced_prompt, tags
                        FROM memory_entries 
                        WHERE id = ? AND user_id = ?
                    """, (result.memory_id, user_id))
                    
                    row = cursor.fetchone()
                    if not row:
                        continue
                    
                    original_prompt, enhanced_prompt, tags_json = row
                    combined_text = f"{original_prompt} {enhanced_prompt}".lower()
                    
                    # Parse tags
                    try:
                        tags = json.loads(tags_json) if tags_json else []
                    except:
                        tags = []
                    
                    # Check if any entity filter matches
                    matches_filter = False
                    for entity_type, entity_values in entity_filters.items():
                        for entity_value in entity_values:
                            if (entity_value.lower() in combined_text or 
                                entity_value.lower() in [tag.lower() for tag in tags]):
                                matches_filter = True
                                break
                        if matches_filter:
                            break
                    
                    if matches_filter:
                        filtered_results.append(result)
        
        except Exception as e:
            logging.error(f"❌ Entity filtering failed: {e}")
            return results
        
        return filtered_results

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'i', 'me', 'my',
            'you', 'your', 'he', 'she', 'it', 'we', 'they', 'them', 'this', 'that',
            'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Limit to 10 keywords

    def _calculate_keyword_score(self, query_terms: List[str], original_prompt: str, 
                                enhanced_prompt: str, tags_json: str) -> float:
        """Calculate relevance score for keyword matching."""
        try:
            combined_text = f"{original_prompt} {enhanced_prompt}".lower()
            
            # Parse tags
            try:
                tags = json.loads(tags_json) if tags_json else []
                tags_text = ' '.join(tags).lower()
            except:
                tags_text = ''
            
            score = 0.0
            total_terms = len(query_terms)
            
            for term in query_terms:
                term_lower = term.lower()
                
                # Score based on where the term appears
                if term_lower in original_prompt.lower():
                    score += 1.0  # Highest weight for original prompt
                elif term_lower in enhanced_prompt.lower():
                    score += 0.7  # Medium weight for enhanced prompt
                elif term_lower in tags_text:
                    score += 0.5  # Lower weight for tags
            
            # Normalize score
            return score / total_terms if total_terms > 0 else 0.0
            
        except Exception as e:
            logging.error(f"❌ Score calculation failed: {e}")
            return 0.0