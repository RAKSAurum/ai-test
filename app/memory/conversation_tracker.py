"""
Conversation Tracking and Context Management

Handles conversation flow, context maintenance, and natural language
understanding for memory queries.
"""

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    timestamp: float
    user_input: str
    system_response: str
    context: Dict[str, Any]
    memory_references: List[str] = None

    def __post_init__(self):
        if self.memory_references is None:
            self.memory_references = []

class ConversationTracker:
    """
    Tracks conversations and manages context for natural language memory queries.
    
    Provides functionality to understand temporal references, extract intent,
    and maintain conversational context for memory operations.
    """

    def __init__(self):
        """Initialize the conversation tracker."""
        self.temporal_patterns = self._init_temporal_patterns()
        self.intent_patterns = self._init_intent_patterns()
        self.entity_patterns = self._init_entity_patterns()
        
        logging.info("💬 Conversation Tracker initialized")

    def _init_temporal_patterns(self) -> Dict[str, List[str]]:
        """Initialize temporal pattern recognition."""
        return {
            'yesterday': [r'\byesterday\b', r'\blast day\b'],
            'today': [r'\btoday\b', r'\bearlier today\b', r'\bthis morning\b', r'\bthis afternoon\b'],
            'last_week': [r'\blast week\b', r'\ba week ago\b'],
            'this_week': [r'\bthis week\b', r'\bearlier this week\b'],
            'last_month': [r'\blast month\b', r'\ba month ago\b'],
            'last_thursday': [r'\blast thursday\b'],
            'last_friday': [r'\blast friday\b'],
            'last_weekend': [r'\blast weekend\b', r'\bover the weekend\b'],
            'recent': [r'\brecently\b', r'\ba while ago\b', r'\bearlier\b'],
            'specific_time': [r'\b(\d{1,2}):(\d{2})\b', r'\b(\d{1,2})\s*(am|pm)\b']
        }

    def _init_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialize intent recognition patterns."""
        return {
            'recall': [
                r'\bshow me\b', r'\bfind\b', r'\brecall\b', r'\bremember\b',
                r'\bget\b', r'\blook for\b', r'\bsearch for\b'
            ],
            'create_similar': [
                r'\blike\b', r'\bsimilar to\b', r'\bbased on\b', r'\binspired by\b',
                r'\bvariation of\b', r'\bmodify\b', r'\bchange\b'
            ],
            'create_new': [
                r'\bcreate\b', r'\bgenerate\b', r'\bmake\b', r'\bbuild\b',
                r'\bdesign\b', r'\bdraw\b', r'\brender\b'
            ],
            'list': [
                r'\blist\b', r'\bshow all\b', r'\bwhat did i\b', r'\bmy creations\b'
            ]
        }

    def _init_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize entity extraction patterns."""
        return {
            'objects': [
                r'\brobot\b', r'\bdragon\b', r'\bcastle\b', r'\bhouse\b', r'\bcar\b',
                r'\bvehicle\b', r'\banimal\b', r'\bcharacter\b', r'\bcreature\b',
                r'\bweapon\b', r'\btool\b', r'\bbuilding\b', r'\bstructure\b'
            ],
            'styles': [
                r'\bfuturistic\b', r'\bmedieval\b', r'\bsteampunk\b', r'\bcyberpunk\b',
                r'\bfantasy\b', r'\bsci-fi\b', r'\bmodern\b', r'\bclassic\b',
                r'\bminimalist\b', r'\bdetailed\b'
            ],
            'colors': [
                r'\bblue\b', r'\bred\b', r'\bgreen\b', r'\byellow\b', r'\bpurple\b',
                r'\borange\b', r'\bblack\b', r'\bwhite\b', r'\bgold\b', r'\bsilver\b',
                r'\bmetallic\b', r'\bglowing\b'
            ],
            'modifications': [
                r'\bwith wings\b', r'\bwith wheels\b', r'\bwith lights\b',
                r'\bwith armor\b', r'\bwith weapons\b', r'\bbigger\b', r'\bsmaller\b',
                r'\btaller\b', r'\bwider\b'
            ]
        }

    def parse_memory_query(self, user_input: str) -> Dict[str, Any]:
        """
        Parse user input to extract memory query information.
        
        Args:
            user_input: Raw user input text
            
        Returns:
            Dict containing parsed query information
        """
        query_info = {
            'intent': self._extract_intent(user_input),
            'temporal': self._extract_temporal_info(user_input),
            'entities': self._extract_entities(user_input),
            'search_terms': self._extract_search_terms(user_input),
            'is_memory_query': self._is_memory_query(user_input)
        }
        
        logging.info(f"🔍 Parsed query: {query_info}")
        return query_info

    def _extract_intent(self, text: str) -> str:
        """Extract the primary intent from user input."""
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        
        # Default intent based on context
        if any(word in text_lower for word in ['show', 'find', 'get', 'recall']):
            return 'recall'
        elif any(word in text_lower for word in ['create', 'make', 'generate']):
            return 'create_new'
        
        return 'unknown'

    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Extract temporal information from user input."""
        text_lower = text.lower()
        temporal_info = {
            'type': None,
            'specific_date': None,
            'relative_days': None,
            'time_range': None
        }
        
        current_time = datetime.now()
        
        for time_type, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    temporal_info['type'] = time_type
                    
                    # Calculate specific time ranges
                    if time_type == 'yesterday':
                        yesterday = current_time - timedelta(days=1)
                        temporal_info['time_range'] = (
                            yesterday.replace(hour=0, minute=0, second=0).timestamp(),
                            yesterday.replace(hour=23, minute=59, second=59).timestamp()
                        )
                    elif time_type == 'today':
                        today = current_time
                        temporal_info['time_range'] = (
                            today.replace(hour=0, minute=0, second=0).timestamp(),
                            today.replace(hour=23, minute=59, second=59).timestamp()
                        )
                    elif time_type == 'last_week':
                        week_ago = current_time - timedelta(days=7)
                        temporal_info['time_range'] = (
                            week_ago.timestamp(),
                            current_time.timestamp()
                        )
                    elif time_type == 'last_thursday':
                        # Find last Thursday
                        days_since_thursday = (current_time.weekday() - 3) % 7
                        if days_since_thursday == 0:  # Today is Thursday
                            days_since_thursday = 7
                        last_thursday = current_time - timedelta(days=days_since_thursday)
                        temporal_info['time_range'] = (
                            last_thursday.replace(hour=0, minute=0, second=0).timestamp(),
                            last_thursday.replace(hour=23, minute=59, second=59).timestamp()
                        )
                    
                    break
            
            if temporal_info['type']:
                break
        
        return temporal_info

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities (objects, styles, colors, etc.) from text."""
        text_lower = text.lower()
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            found_entities = []
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    if isinstance(matches[0], tuple):
                        found_entities.extend([match[0] for match in matches])
                    else:
                        found_entities.extend(matches)
            
            if found_entities:
                entities[entity_type] = list(set(found_entities))  # Remove duplicates
        
        return entities

    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract key search terms from user input."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'i', 'me', 'my',
            'you', 'your', 'he', 'she', 'it', 'we', 'they', 'them', 'this', 'that',
            'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'show', 'find', 'get', 'like'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        search_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return search_terms

    def _is_memory_query(self, text: str) -> bool:
        """Determine if the input is a memory-related query."""
        memory_indicators = [
            'show me', 'find', 'recall', 'remember', 'get', 'look for',
            'yesterday', 'last', 'earlier', 'before', 'previous',
            'like the', 'similar to', 'based on', 'that', 'which'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in memory_indicators)

    def build_search_query(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a structured search query from parsed information.
        
        Args:
            parsed_query: Parsed query information
            
        Returns:
            Dict containing search parameters
        """
        search_query = {
            'text_search': ' '.join(parsed_query.get('search_terms', [])),
            'time_range': None,
            'entity_filters': {},
            'intent': parsed_query.get('intent', 'unknown')
        }
        
        # Add temporal constraints
        temporal_info = parsed_query.get('temporal', {})
        if temporal_info.get('time_range'):
            search_query['time_range'] = temporal_info['time_range']
        
        # Add entity filters
        entities = parsed_query.get('entities', {})
        for entity_type, entity_list in entities.items():
            if entity_list:
                search_query['entity_filters'][entity_type] = entity_list
        
        return search_query

    def format_memory_response(self, memories: List[Any], query_intent: str) -> str:
        """
        Format memory search results into a natural language response.
        
        Args:
            memories: List of memory entries
            query_intent: The intent of the original query
            
        Returns:
            str: Formatted response text
        """
        if not memories:
            return "I couldn't find any memories matching your request. Try describing what you're looking for differently."
        
        response_parts = []
        
        if query_intent == 'recall':
            response_parts.append(f"I found {len(memories)} memory(ies) matching your request:")
        elif query_intent == 'list':
            response_parts.append(f"Here are your {len(memories)} recent creation(s):")
        else:
            response_parts.append(f"Found {len(memories)} relevant memory(ies):")
        
        for i, memory in enumerate(memories[:5], 1):  # Limit to 5 results
            timestamp = datetime.fromtimestamp(memory.timestamp)
            time_str = timestamp.strftime("%Y-%m-%d %H:%M")
            
            response_parts.append(
                f"\n{i}. **{memory.original_prompt}** (created {time_str})"
            )
            
            if memory.tags:
                response_parts.append(f"   Tags: {', '.join(memory.tags[:5])}")
        
        if len(memories) > 5:
            response_parts.append(f"\n... and {len(memories) - 5} more results")
        
        return ''.join(response_parts)