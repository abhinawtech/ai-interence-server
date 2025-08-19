#!/usr/bin/env python3
"""
Conversation Database Cleanup Tool
Removes low-quality conversations while preserving high-quality ones.
"""

import requests
import json
from typing import List, Dict, Any

QDRANT_URL = "http://localhost:6334"
COLLECTION_NAME = "conversations"

def assess_conversation_quality(conversation: str) -> float:
    """
    Assess conversation quality using the same logic as Rust implementation.
    """
    if not conversation.strip():
        return 0.0
    
    words = conversation.split()
    if len(words) < 3:
        return 0.1
    
    # Check for excessive repetition
    unique_words = set(words)
    uniqueness_ratio = len(unique_words) / len(words)
    
    # Check for proper structure
    has_proper_dialogue = "User:" in conversation and "Assistant:" in conversation
    reasonable_length = 5 <= len(words) <= 200
    not_corrupted = not all(c in 'ai' for c in conversation.replace(' ', '').lower()[:20])
    
    if uniqueness_ratio > 0.6 and has_proper_dialogue and reasonable_length and not_corrupted:
        return 0.9
    elif uniqueness_ratio > 0.4 and has_proper_dialogue and reasonable_length and not_corrupted:
        return 0.7
    elif uniqueness_ratio > 0.5 and reasonable_length and not_corrupted:
        return 0.6
    else:
        return 0.2

def get_all_conversations() -> List[Dict[str, Any]]:
    """Retrieve all conversations from Qdrant."""
    response = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll",
        json={
            "limit": 1000,
            "with_payload": True,
            "with_vector": False
        }
    )
    response.raise_for_status()
    return response.json()["result"]["points"]

def delete_conversation(point_id: str):
    """Delete a conversation by ID."""
    response = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/delete",
        json={
            "points": [point_id]
        }
    )
    response.raise_for_status()

def cleanup_conversations():
    """Main cleanup function."""
    print("🔍 Fetching all conversations...")
    conversations = get_all_conversations()
    print(f"📊 Found {len(conversations)} conversations")
    
    low_quality_count = 0
    high_quality_count = 0
    
    for conv in conversations:
        if "conversation" in conv["payload"]:
            conversation_text = conv["payload"]["conversation"]
            quality_score = assess_conversation_quality(conversation_text)
            
            print(f"📝 ID: {conv['id']} | Quality: {quality_score:.2f} | Text: {conversation_text[:50]}...")
            
            if quality_score < 0.5:  # Remove low-quality conversations
                print(f"🗑️  Deleting low-quality conversation (score: {quality_score:.2f})")
                delete_conversation(conv["id"])
                low_quality_count += 1
            else:
                print(f"✅ Keeping high-quality conversation (score: {quality_score:.2f})")
                high_quality_count += 1
    
    print(f"\n🎯 Cleanup Complete:")
    print(f"   ✅ Kept: {high_quality_count} high-quality conversations")
    print(f"   🗑️  Removed: {low_quality_count} low-quality conversations")

if __name__ == "__main__":
    try:
        cleanup_conversations()
    except Exception as e:
        print(f"❌ Error: {e}")