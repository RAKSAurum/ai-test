"""
Memory System for AI 3D Generator

This module provides comprehensive memory functionality for AI applications including:
- Short-term session memory management
- Long-term persistent storage solutions
- Natural language query processing capabilities
- Semantic search and retrieval functionality
- Conversation tracking and context management

This package is designed to integrate seamlessly with AI image generation workflows,
particularly for 3D model generation and text-to-image applications.
"""

from .conversation_tracker import ConversationTracker
from .memory_manager import MemoryManager, MemoryEntry  # Add MemoryEntry
from .semantic_search import SemanticSearch

# Update public API exports
__all__ = ['MemoryManager', 'ConversationTracker', 'SemanticSearch', 'MemoryEntry']

# Package metadata
__version__ = "1.0.0"
__author__ = "RAKSAurum"
__description__ = "Comprehensive memory system for AI 3D generation applications"