"""
Semantic Search and Natural Language Processing

Provides advanced search capabilities using vector embeddings and
natural language understanding for memory queries in AI applications.
"""

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Global flag for embeddings availability - define at module level
EMBEDDINGS_AVAILABLE = False

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    
    EMBEDDINGS_AVAILABLE = True
    logging.info("✅ Sentence transformers and FAISS available")
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False
    logging.warning(f"⚠️ Sentence transformers not available: {e}. Using fallback search.")


@dataclass
class SearchResult:
    """
    Represents a search result with relevance scoring and metadata.
    
    This class encapsulates the information returned from semantic or keyword
    searches, including relevance metrics and match type information for
    result ranking and display purposes.
    
    Attributes:
        memory_id (str): Unique identifier for the memory entry.
        relevance_score (float): Calculated relevance score (0.0-1.0).
        match_type (str): Type of match ('semantic', 'keyword', 'temporal').
        matched_content (str): The content that matched the search query.
    """
    memory_id: str
    relevance_score: float
    match_type: str  # 'semantic', 'keyword', 'temporal'
    matched_content: str


class SemanticSearch:
    """
    Advanced semantic search system for AI memory queries.
    
    Provides comprehensive search capabilities using vector-based similarity search
    with intelligent fallback to keyword matching when embedding models are not available.
    This system is designed to work seamlessly with AI 3D generation workflows and
    memory management systems.
    
    The search system supports multiple search modes:
    - Semantic search using sentence transformers and FAISS indexing
    - Keyword-based fallback search with relevance scoring
    - Temporal search for time-based queries
    - Hybrid search combining multiple approaches
    - Entity filtering for refined results
    
    Attributes:
        db_path (Path): Path to the memory database file.
        model (Optional[SentenceTransformer]): Loaded sentence transformer model.
        index (Optional[faiss.Index]): FAISS vector index for similarity search.
        memory_vectors (Dict[str, int]): Mapping of memory IDs to vector indices.
        embedding_dim (int): Dimensionality of embedding vectors.
        embeddings_available (bool): Flag indicating if embeddings are available.
    
    Example:
        >>> search = SemanticSearch("memory/ai_memory.db")
        >>> results = search.semantic_search("robot with wings", "user123")
        >>> hybrid_results = search.hybrid_search("futuristic car", "user123", limit=5)
    """

    def __init__(self, db_path: str = "memory/ai_memory.db") -> None:
        """
        Initialize semantic search system with embedding models and vector index.
        
        Sets up the search system with sentence transformers for embeddings,
        FAISS for vector similarity search, and database connections for
        persistent storage and retrieval.
        
        Args:
            db_path (str): Path to the memory database. Defaults to "memory/ai_memory.db".
        """
        self.db_path = Path(db_path)
        self.model = None
        self.index = None
        self.memory_vectors = {}
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.embeddings_available = EMBEDDINGS_AVAILABLE
        
        # Initialize embedding model if available
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
                self._initialize_vector_index()
                logging.info("🔍 Semantic search initialized with embeddings")
            except Exception as e:
                logging.warning(f"⚠️ Failed to load embedding model: {e}")
                self.embeddings_available = False
        
        if not self.embeddings_available:
            logging.info("🔍 Semantic search initialized with keyword fallback")

    def _initialize_vector_index(self) -> None:
        """
        Initialize FAISS vector index for efficient similarity search.
        
        Creates a FAISS index optimized for cosine similarity and loads
        existing embeddings from the database to maintain search continuity.
        """
        try:
            # Create FAISS index for inner product (cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Load existing embeddings from database
            self._load_existing_embeddings()
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize vector index: {e}")
            self.index = None

    def _load_existing_embeddings(self) -> None:
        """
        Load existing memory embeddings from database into FAISS index.
        
        Retrieves previously computed embeddings from the database and
        rebuilds the FAISS index for immediate search availability.
        """
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

    def add_memory_embedding(self, memory_id: str, text_content: str) -> None:
        """
        Add embedding for a new memory entry to enable semantic search.
        
        Generates vector embeddings for new memory content and adds them
        to both the FAISS index and persistent database storage.
        
        Args:
            memory_id (str): Unique memory identifier.
            text_content (str): Text content to embed and index.
        """
        if not self.embeddings_available or not self.model:
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

    def _store_embedding(self, memory_id: str, embedding: np.ndarray) -> None:
        """
        Store embedding in persistent database storage.
        
        Args:
            memory_id (str): Memory identifier for the embedding.
            embedding (np.ndarray): Normalized embedding vector to store.
        """
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
        Perform semantic search using vector embeddings and similarity matching.
        
        Converts the query to an embedding vector and searches for the most
        similar memory entries using cosine similarity in the FAISS index.
        Falls back to keyword search if embeddings are unavailable.
        
        Args:
            query (str): Search query text to find similar memories.
            user_id (str): User identifier for filtering results. Defaults to "default".
            limit (int): Maximum number of results to return. Defaults to 10.
            threshold (float): Minimum similarity threshold (0.0-1.0). Defaults to 0.3.
            
        Returns:
            List[SearchResult]: Ranked search results sorted by relevance score.
        """
        if not self.embeddings_available or not self.model or not self.index:
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
        
        Performs text-based matching using SQL LIKE queries and calculates
        relevance scores based on keyword matches and memory quality metrics.
        
        Args:
            query (str): Search query text for keyword matching.
            user_id (str): User identifier for filtering. Defaults to "default".
            limit (int): Maximum number of results to return. Defaults to 10.
            
        Returns:
            List[SearchResult]: Keyword-based search results with relevance scores.
        """
        try:
            query_terms = self._extract_keywords(query)
            if not query_terms:
                return []
            
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                # Build search query with dynamic conditions
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
                    match_score = self._calculate_keyword_score(
                        query_terms, original_prompt, enhanced_prompt, tags
                    )
                    
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
        Search memories within a specific time range with recency scoring.
        
        Retrieves memories created within the specified time window and
        calculates relevance scores based on recency and quality metrics.
        
        Args:
            user_id (str): User identifier for filtering results.
            time_range (Tuple[float, float]): Tuple of (start_time, end_time) timestamps.
            limit (int): Maximum number of results to return. Defaults to 10.
            
        Returns:
            List[SearchResult]: Time-filtered results sorted by combined relevance score.
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
        
        Combines multiple search approaches to provide comprehensive results
        that match both semantic similarity and specified constraints.
        
        Args:
            query (str): Search query text for semantic/keyword matching.
            user_id (str): User identifier for filtering. Defaults to "default".
            time_range (Optional[Tuple[float, float]]): Optional time range filter.
            entity_filters (Optional[Dict[str, List[str]]]): Optional entity-based filters.
            limit (int): Maximum number of results to return. Defaults to 10.
            
        Returns:
            List[SearchResult]: Combined search results with merged relevance scores.
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
        """
        Apply entity-based filtering to search results.
        
        Filters search results based on entity matches in prompts and tags
        to provide more targeted results for specific categories or attributes.
        
        Args:
            results (List[SearchResult]): Search results to filter.
            entity_filters (Dict[str, List[str]]): Entity filters to apply.
            user_id (str): User identifier for database queries.
            
        Returns:
            List[SearchResult]: Filtered results matching entity criteria.
        """
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
        """
        Extract meaningful keywords from text by removing stop words.
        
        Processes input text to identify significant terms for keyword-based
        search while filtering out common stop words and short terms.
        
        Args:
            text (str): Input text to process for keywords.
            
        Returns:
            List[str]: List of meaningful search terms (limited to 10).
        """
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Comprehensive stop words list for better search term extraction
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
        return keywords[:10]  # Limit to 10 keywords for performance

    def _calculate_keyword_score(self, query_terms: List[str], original_prompt: str, 
                                enhanced_prompt: str, tags_json: str) -> float:
        """
        Calculate relevance score for keyword matching with weighted scoring.
        
        Computes relevance based on where keywords appear (original prompt
        weighted highest, enhanced prompt medium, tags lowest) and normalizes
        the score based on total query terms.
        
        Args:
            query_terms (List[str]): List of search terms to match.
            original_prompt (str): Original user prompt text.
            enhanced_prompt (str): Enhanced prompt text.
            tags_json (str): JSON string of tags.
            
        Returns:
            float: Normalized relevance score (0.0-1.0).
        """
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
                
                # Score based on where the term appears (weighted scoring)
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