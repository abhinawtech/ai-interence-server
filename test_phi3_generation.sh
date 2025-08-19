#!/bin/bash

# Simple test script for phi3 generation API
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SERVER_URL="http://localhost:3000"

echo -e "${BLUE}🧪 Testing Phi-3 Generation API${NC}"
echo "============================================"

# Check if server is running
echo -e "${BLUE}1. Checking server health...${NC}"
if curl -s "$SERVER_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Server is running${NC}"
else
    echo -e "${RED}❌ Server is not running at $SERVER_URL${NC}"
    echo "Please start the server first: cargo run"
    exit 1
fi

# Test basic generation
echo -e "\n${BLUE}2. Testing basic text generation...${NC}"
response=$(curl -s -X POST "$SERVER_URL/api/v1/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "What is artificial intelligence?",
        "max_tokens": 50,
        "use_memory": false
    }')

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Generation request successful${NC}"
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
else
    echo -e "${RED}❌ Generation request failed${NC}"
    exit 1
fi

# Test with memory enabled
echo -e "\n${BLUE}3. Testing generation with memory...${NC}"
response=$(curl -s -X POST "$SERVER_URL/api/v1/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Hello, my name is Alice",
        "max_tokens": 30,
        "use_memory": true
    }')

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Memory-enabled generation successful${NC}"
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
else
    echo -e "${RED}❌ Memory-enabled generation failed${NC}"
fi

# Test follow-up question
echo -e "\n${BLUE}4. Testing follow-up question...${NC}"
response=$(curl -s -X POST "$SERVER_URL/api/v1/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "What is my name?",
        "max_tokens": 20,
        "use_memory": true
    }')

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Follow-up question successful${NC}"
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
else
    echo -e "${RED}❌ Follow-up question failed${NC}"
fi

echo -e "\n${GREEN}🎉 Phi-3 API testing completed!${NC}"