#!/usr/bin/env python3
"""
Test script to verify file upload functionality works
"""
import requests
import tempfile
import os

def test_file_upload():
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document for upload testing.\nIt contains multiple lines to test document processing.")
        test_file_path = f.name
    
    try:
        # Prepare the upload request
        url = "http://localhost:3000/api/v1/generate/upload"
        
        files = {
            'file': ('test_document.txt', open(test_file_path, 'rb'), 'text/plain')
        }
        
        data = {
            'prompt': 'What is this document about?',
            'max_tokens': '50',
            'auto_process_document': 'true',
            'use_document_context': 'true'
        }
        
        print("🚀 Testing file upload endpoint...")
        print(f"📄 Uploading file: {test_file_path}")
        
        # Make the request with a timeout
        response = requests.post(url, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ File upload successful!")
            print(f"📝 Response: {result.get('text', 'No text returned')}")
            print(f"📊 Document processed: {result.get('document_processed', False)}")
            print(f"🧩 Chunks created: {result.get('document_chunks_created', 0)}")
            return True
        else:
            print(f"❌ File upload failed with status: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - this indicates the hanging issue still exists")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server - make sure it's running on port 3000")
        return False
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False
    finally:
        # Clean up
        os.unlink(test_file_path)

if __name__ == "__main__":
    success = test_file_upload()
    exit(0 if success else 1)