// ================================================================================================
// SIMPLE IN-MEMORY VECTOR STORAGE
// ================================================================================================
//
// A basic in-memory vector store to get started. This will be replaced with Qdrant later.
//
// ================================================================================================

use super::{VectorPoint, VectorResult};
use std::collections::HashMap;
use uuid::Uuid;

/// Simple in-memory vector storage
#[derive(Debug, Default)]
pub struct VectorStorage {
    vectors: HashMap<Uuid, VectorPoint>,
}

impl VectorStorage {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }
    
    /// Insert a vector point
    pub fn insert(&mut self, point: VectorPoint) -> VectorResult<Uuid> {
        let id = point.id;
        self.vectors.insert(id, point);
        Ok(id)
    }
    
    /// Get a vector point by ID
    pub fn get(&self, id: &Uuid) -> Option<&VectorPoint> {
        self.vectors.get(id)
    }
    
    /// Delete a vector point
    pub fn delete(&mut self, id: &Uuid) -> VectorResult<bool> {
        Ok(self.vectors.remove(id).is_some())
    }
    
    /// Get all vector points
    pub fn list_all(&self) -> Vec<&VectorPoint> {
        self.vectors.values().collect()
    }
    
    /// Simple similarity search using cosine similarity
    pub fn search_similar(&self, query_vector: &[f32], limit: usize) -> VectorResult<Vec<(Uuid, f32)>> {
        let mut results: Vec<(Uuid, f32)> = self.vectors
            .iter()
            .map(|(id, point)| {
                let similarity = cosine_similarity(query_vector, &point.vector);
                (*id, similarity)
            })
            .collect();
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top results
        results.truncate(limit);
        
        Ok(results)
    }
    
    /// Get storage statistics
    pub fn stats(&self) -> StorageStats {
        StorageStats {
            total_vectors: self.vectors.len(),
            memory_usage_estimate: self.vectors.len() * std::mem::size_of::<VectorPoint>(),
        }
    }
}

/// Storage statistics
#[derive(Debug)]
pub struct StorageStats {
    pub total_vectors: usize,
    pub memory_usage_estimate: usize,
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_storage_operations() {
        let mut storage = VectorStorage::new();
        
        // Test insert
        let point = VectorPoint::new(vec![1.0, 2.0, 3.0]);
        let id = storage.insert(point.clone()).unwrap();
        assert_eq!(id, point.id);
        
        // Test get
        let retrieved = storage.get(&id).unwrap();
        assert_eq!(retrieved.vector, point.vector);
        
        // Test delete
        assert!(storage.delete(&id).unwrap());
        assert!(storage.get(&id).is_none());
    }
    
    #[test]
    fn test_similarity_search() {
        let mut storage = VectorStorage::new();
        
        // Insert some test vectors
        let point1 = VectorPoint::new(vec![1.0, 0.0, 0.0]);
        let point2 = VectorPoint::new(vec![0.9, 0.1, 0.0]);
        let point3 = VectorPoint::new(vec![0.0, 1.0, 0.0]);
        
        storage.insert(point1.clone()).unwrap();
        storage.insert(point2.clone()).unwrap();
        storage.insert(point3.clone()).unwrap();
        
        // Search for vectors similar to [1.0, 0.0, 0.0]
        let query = vec![1.0, 0.0, 0.0];
        let results = storage.search_similar(&query, 2).unwrap();
        
        assert_eq!(results.len(), 2);
        // First result should be point1 (exact match)
        assert_eq!(results[0].0, point1.id);
        assert!((results[0].1 - 1.0).abs() < 0.001);
        // Second result should be point2 (high similarity)
        assert_eq!(results[1].0, point2.id);
        assert!(results[1].1 > 0.9);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
        
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }
}