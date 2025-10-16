"""
Sentence Encoder for Bio Text Processing
"""

import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class SentenceEncoder:
    """
    Wrapper for sentence transformer models to encode bio text
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto"):
        """
        Initialize sentence encoder
        
        Args:
            model_name: HuggingFace model name for sentence transformer
            device: Device to run model on (auto, cpu, cuda, mps)
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        
        # Load sentence transformer model
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ Loaded {model_name} on {device} (dim: {self.embedding_dim})")
        except Exception as e:
            print(f"⚠️ Failed to load {model_name}, falling back to simple encoder: {e}")
            self.model = None
            self.embedding_dim = 64
    
    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts to encode
            convert_to_tensor: Return torch tensor instead of numpy array
            normalize_embeddings: L2 normalize embeddings
            
        Returns:
            Embeddings as numpy array or torch tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model is not None:
            # Use sentence transformer
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=normalize_embeddings,
                device=self.device
            )
        else:
            # Fallback to simple encoding
            embeddings = self._simple_encode(texts, convert_to_tensor, normalize_embeddings)
        
        return embeddings
    
    def _simple_encode(
        self,
        texts: List[str],
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Simple fallback encoding when sentence transformer is not available
        """
        embeddings = []
        
        for text in texts:
            # Simple word-based encoding
            words = text.lower().split()
            embedding = np.zeros(self.embedding_dim)
            
            if words:
                # Hash-based word encoding
                for i, word in enumerate(words[:10]):  # Max 10 words
                    word_hash = hash(word) % self.embedding_dim
                    embedding[word_hash] += 1.0 / (i + 1)  # Position weighting
                
                # Add text length feature
                embedding[0] = len(words) / 50.0
                
                # Add some randomness based on text content
                np.random.seed(hash(text) % 1000000)
                embedding += np.random.normal(0, 0.1, self.embedding_dim)
            
            # Normalize if requested
            if normalize_embeddings:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        if convert_to_tensor:
            embeddings = torch.FloatTensor(embeddings)
            if self.device != "cpu":
                embeddings = embeddings.to(self.device)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim


class BatchSentenceEncoder:
    """
    Batch processor for encoding large numbers of texts efficiently
    """
    
    def __init__(self, encoder: SentenceEncoder, batch_size: int = 32):
        """
        Initialize batch encoder
        
        Args:
            encoder: SentenceEncoder instance
            batch_size: Batch size for processing
        """
        self.encoder = encoder
        self.batch_size = batch_size
    
    def encode_batch(
        self,
        texts: List[str],
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
        show_progress: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode large list of texts in batches
        
        Args:
            texts: List of texts to encode
            convert_to_tensor: Return torch tensor
            normalize_embeddings: L2 normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            All embeddings concatenated
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            batch_embeddings = self.encoder.encode(
                batch_texts,
                convert_to_tensor=False,  # Keep as numpy for concatenation
                normalize_embeddings=normalize_embeddings
            )
            
            all_embeddings.append(batch_embeddings)
            
            if show_progress and (i // self.batch_size + 1) % 10 == 0:
                processed = min(i + self.batch_size, len(texts))
                print(f"   Processed {processed:,}/{len(texts):,} texts")
        
        # Concatenate all batches
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
        else:
            final_embeddings = np.array([])
        
        if convert_to_tensor:
            final_embeddings = torch.FloatTensor(final_embeddings)
            if self.encoder.device != "cpu":
                final_embeddings = final_embeddings.to(self.encoder.device)
        
        return final_embeddings


def create_sentence_encoder(config: dict) -> SentenceEncoder:
    """
    Create sentence encoder from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized SentenceEncoder
    """
    model_config = config['model']
    hardware_config = config['hardware']
    
    return SentenceEncoder(
        model_name=model_config.get('text_encoder', 'sentence-transformers/all-MiniLM-L6-v2'),
        device=hardware_config.get('device', 'auto')
    )
