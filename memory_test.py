#!/usr/bin/env python3
"""
Memory Leak Test Script for AI Inference Server

This script simulates various usage patterns that could cause memory leaks
and monitors the server's memory usage to verify the fixes are working.

Usage:
    python memory_test.py

Requirements:
    pip install requests psutil
"""

import requests
import time
import json
import random
import string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_URL = "http://localhost:3000"

def random_text(length=100):
    """Generate random text for testing"""
    return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))

def test_memory_monitoring():
    """Test the memory monitoring endpoints"""
    print("🧠 Testing memory monitoring endpoints...")
    
    try:
        # Get initial memory stats
        response = requests.get(f"{SERVER_URL}/api/v1/memory/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"Initial memory usage: {stats['process_memory_mb']}MB")
            return stats
        else:
            print(f"Failed to get memory stats: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error testing memory endpoints: {e}")
        return None

def test_embedding_cache_stress():
    """Test embedding cache with many unique requests"""
    print("📊 Testing embedding cache stress...")
    
    def make_unique_embedding_request():
        unique_text = random_text(random.randint(50, 200))
        try:
            response = requests.post(f"{SERVER_URL}/api/v1/embed", 
                                   json={"text": unique_text})
            return response.status_code == 200
        except:
            return False
    
    # Generate 1000 unique embedding requests
    successful = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_unique_embedding_request) for _ in range(1000)]
        for future in as_completed(futures):
            if future.result():
                successful += 1
    
    print(f"Embedding cache stress test: {successful}/1000 successful requests")
    return successful

def test_search_session_cleanup():
    """Test search session creation and cleanup"""
    print("🔍 Testing search session cleanup...")
    
    def create_search_session():
        session_id = f"session_{random.randint(1000, 9999)}"
        try:
            response = requests.post(f"{SERVER_URL}/api/v1/search/semantic", 
                                   json={
                                       "query": random_text(50),
                                       "session_id": session_id
                                   })
            return response.status_code == 200
        except:
            return False
    
    # Create 500 different search sessions
    successful = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(create_search_session) for _ in range(500)]
        for future in as_completed(futures):
            if future.result():
                successful += 1
    
    print(f"Search session test: {successful}/500 successful requests")
    return successful

def test_batch_processor_stress():
    """Test batch processor with high load"""
    print("⚡ Testing batch processor stress...")
    
    def make_generation_request():
        try:
            response = requests.post(f"{SERVER_URL}/api/v1/generate", 
                                   json={
                                       "prompt": random_text(30),
                                       "max_tokens": 50
                                   })
            return response.status_code == 200
        except:
            return False
    
    # Generate 200 concurrent requests
    successful = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(make_generation_request) for _ in range(200)]
        for future in as_completed(futures):
            if future.result():
                successful += 1
    
    print(f"Batch processor stress test: {successful}/200 successful requests")
    return successful

def monitor_memory_over_time(duration_minutes=5):
    """Monitor memory usage over time"""
    print(f"📈 Monitoring memory usage for {duration_minutes} minutes...")
    
    memory_readings = []
    start_time = time.time()
    
    while time.time() - start_time < duration_minutes * 60:
        try:
            response = requests.get(f"{SERVER_URL}/api/v1/memory/stats")
            if response.status_code == 200:
                stats = response.json()
                memory_readings.append({
                    'timestamp': time.time(),
                    'memory_mb': stats.get('process_memory_mb', 0),
                    'health_score': stats.get('memory_health_score', 0)
                })
                print(f"Memory: {stats.get('process_memory_mb', 0)}MB, "
                      f"Health: {stats.get('memory_health_score', 0):.2f}")
        except Exception as e:
            print(f"Failed to get memory stats: {e}")
        
        time.sleep(30)  # Check every 30 seconds
    
    return memory_readings

def trigger_cleanup():
    """Trigger manual cleanup"""
    print("🧹 Triggering manual memory cleanup...")
    
    try:
        response = requests.get(f"{SERVER_URL}/api/v1/memory/cleanup")
        if response.status_code == 200:
            result = response.json()
            print(f"Cleanup freed: {result.get('freed_memory_mb', 0)}MB")
            return result.get('freed_memory_mb', 0)
        else:
            print(f"Cleanup failed: {response.status_code}")
            return 0
    except Exception as e:
        print(f"Error triggering cleanup: {e}")
        return 0

def main():
    """Run comprehensive memory leak tests"""
    print("🚀 Starting AI Inference Server Memory Leak Tests")
    print("=" * 60)
    
    # Test server connectivity
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not responding. Start the server first.")
            return
        print("✅ Server is running")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Get initial memory stats
    initial_stats = test_memory_monitoring()
    if not initial_stats:
        print("❌ Memory monitoring not available")
        return
    
    initial_memory = initial_stats.get('process_memory_mb', 0)
    
    print(f"\n📊 Initial Memory Usage: {initial_memory}MB")
    print("=" * 60)
    
    # Run stress tests
    test_embedding_cache_stress()
    test_search_session_cleanup() 
    test_batch_processor_stress()
    
    # Monitor memory for 2 minutes
    readings = monitor_memory_over_time(2)
    
    # Get final memory stats
    final_stats = test_memory_monitoring()
    final_memory = final_stats.get('process_memory_mb', 0) if final_stats else 0
    
    # Trigger cleanup
    freed_mb = trigger_cleanup()
    
    # Final analysis
    print("\n" + "=" * 60)
    print("📊 MEMORY LEAK TEST RESULTS")
    print("=" * 60)
    print(f"Initial Memory: {initial_memory}MB")
    print(f"Final Memory: {final_memory}MB")
    print(f"Memory Growth: {final_memory - initial_memory}MB")
    print(f"Cleanup Freed: {freed_mb}MB")
    
    if len(readings) > 0:
        max_memory = max(r['memory_mb'] for r in readings)
        min_memory = min(r['memory_mb'] for r in readings)
        print(f"Peak Memory: {max_memory}MB")
        print(f"Memory Range: {min_memory}MB - {max_memory}MB")
    
    # Health assessment
    if final_memory - initial_memory < 100:
        print("✅ PASS: Memory growth is within acceptable limits")
    elif final_memory - initial_memory < 200:
        print("⚠️  WARNING: Moderate memory growth detected")
    else:
        print("❌ FAIL: Significant memory growth - potential leak")
    
    print("\n💡 Recommendations:")
    if final_stats and final_stats.get('recommendations'):
        for rec in final_stats['recommendations']:
            print(f"  • {rec}")
    else:
        print("  • Monitor memory usage in production")
        print("  • Consider implementing additional cleanup strategies")

if __name__ == "__main__":
    main()