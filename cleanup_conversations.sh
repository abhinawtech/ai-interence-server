#!/bin/bash

# Conversation Database Cleanup Script
# Removes low-quality conversations while preserving high-quality ones

QDRANT_URL="http://localhost:6333"
COLLECTION_NAME="conversations"

echo "🔍 Fetching all conversations from Qdrant..."

# Get all conversations
response=$(curl -s -X POST "${QDRANT_URL}/collections/${COLLECTION_NAME}/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 1000,
    "with_payload": true,
    "with_vector": false
  }')

# Check if curl succeeded
if [ $? -ne 0 ]; then
    echo "❌ Failed to connect to Qdrant. Make sure it's running on port 6334."
    exit 1
fi

# Extract conversations and analyze them
echo "$response" | jq -r '.result.points[] | "\(.id)|\(.payload.conversation // "NO_CONVERSATION")"' | while IFS='|' read -r id conversation; do
    if [ "$conversation" = "NO_CONVERSATION" ]; then
        echo "⚠️  Skipping point $id - no conversation data"
        continue
    fi
    
    # Simple quality assessment (bash version)
    word_count=$(echo "$conversation" | wc -w)
    unique_words=$(echo "$conversation" | tr ' ' '\n' | sort -u | wc -l)
    
    # Check for basic quality indicators
    has_user=$(echo "$conversation" | grep -c "User:")
    has_assistant=$(echo "$conversation" | grep -c "Assistant:")
    has_repetition=$(echo "$conversation" | grep -c "Alex, Alex\|AI's, Alex\|Sarah, Sarah")
    
    # Calculate basic quality score
    quality_score=0
    if [ $word_count -gt 5 ] && [ $word_count -lt 200 ]; then
        quality_score=$((quality_score + 30))
    fi
    if [ $has_user -gt 0 ] && [ $has_assistant -gt 0 ]; then
        quality_score=$((quality_score + 40))
    fi
    if [ $has_repetition -eq 0 ]; then
        quality_score=$((quality_score + 30))
    fi
    
    # Display conversation info
    conversation_preview=$(echo "$conversation" | head -c 80)
    echo "📝 ID: $id | Score: $quality_score | Words: $word_count | Preview: $conversation_preview..."
    
    # Delete low-quality conversations (score < 50)
    if [ $quality_score -lt 50 ]; then
        echo "🗑️  Deleting low-quality conversation (score: $quality_score)"
        curl -s -X POST "${QDRANT_URL}/collections/${COLLECTION_NAME}/points/delete" \
          -H "Content-Type: application/json" \
          -d "{\"points\": [\"$id\"]}" > /dev/null
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully deleted conversation $id"
        else
            echo "❌ Failed to delete conversation $id"
        fi
    else
        echo "✅ Keeping high-quality conversation (score: $quality_score)"
    fi
    echo "---"
done

echo "🎯 Cleanup completed! Check the logs above for details."
echo "💡 You may want to restart your AI server to see the improvements."