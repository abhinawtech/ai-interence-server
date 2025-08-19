// ================================================================================================
// SIMPLE TEXT EMBEDDING - MINIMAL IMPLEMENTATION
// ================================================================================================
//
// Basic text-to-vector conversion for conversation memory.
// Uses simple token-based approach for now.
//
// ================================================================================================

use std::collections::HashMap;

/// Simple text embedding using character frequency and position
pub fn create_simple_embedding(text: &str, dimension: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; dimension];
    
    if text.is_empty() {
        return embedding;
    }
    
    // Normalize text
    let text = text.to_lowercase().replace(|c: char| !c.is_alphanumeric() && c != ' ', "");
    let words: Vec<&str> = text.split_whitespace().collect();
    
    if words.is_empty() {
        return embedding;
    }
    
    // Character frequency embedding
    let mut char_freq: HashMap<char, usize> = HashMap::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            *char_freq.entry(c).or_insert(0) += 1;
        }
    }
    
    // Fill embedding with character frequency features
    for (i, c) in "abcdefghijklmnopqrstuvwxyz0123456789".chars().enumerate() {
        if i < dimension / 2 {
            embedding[i] = char_freq.get(&c).unwrap_or(&0).clone() as f32 / text.len() as f32;
        }
    }
    
    // Add word-level features
    let word_count = words.len() as f32;
    for (i, word) in words.iter().enumerate() {
        let feature_idx = (dimension / 2) + (i % (dimension / 2));
        if feature_idx < dimension {
            embedding[feature_idx] += word.len() as f32 / word_count;
        }
    }
    
    // Add text length feature
    if dimension > 0 {
        embedding[dimension - 1] = (text.len() as f32).ln() / 10.0;
    }
    
    // Normalize the embedding
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }
    
    embedding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_embedding() {
        let text = "Hello world";
        let embedding = create_simple_embedding(text, 64);
        assert_eq!(embedding.len(), 64);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01 || norm == 0.0);
    }
    
    #[test]
    fn test_empty_text() {
        let embedding = create_simple_embedding("", 64);
        assert_eq!(embedding.len(), 64);
        assert!(embedding.iter().all(|&x| x == 0.0));
    }
    
    #[test]
    fn test_similar_texts() {
        let emb1 = create_simple_embedding("hello world", 64);
        let emb2 = create_simple_embedding("hello world!", 64);
        
        // Should be similar (cosine similarity)
        let dot: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        assert!(dot > 0.5);
    }
}