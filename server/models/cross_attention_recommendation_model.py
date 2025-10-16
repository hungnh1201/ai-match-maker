"""
Cross-Attention Two-Tower Recommendation Model
Learns from user interactions (age, bio, distance) and recommends similar profiles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import math


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention for profile matching"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(attended), attention_weights.mean(dim=1)  # Return attention weights for interpretability


class UserInteractionTower(nn.Module):
    """
    User Tower - Learns from user's interaction patterns
    Processes: age preferences, bio patterns, location preferences, interaction history
    """
    
    def __init__(
        self,
        feature_dim: int = 8,  # age, gender, lat, long, bio_length, has_bio, interaction_count, avg_action_weight
        bio_embedding_dim: int = 384,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64],
        num_attention_heads: int = 8,
        dropout: float = 0.2
    ):
        super(UserInteractionTower, self).__init__()
        
        self.feature_dim = feature_dim
        self.bio_embedding_dim = bio_embedding_dim
        self.embedding_dim = embedding_dim
        
        # User demographic features processing
        self.demographic_layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # User bio preference processing (learned from liked profiles)
        self.bio_preference_layers = nn.Sequential(
            nn.Linear(bio_embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Interaction pattern fusion
        fusion_input_dim = hidden_dims[1] * 2  # demographics + bio preferences
        self.interaction_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Self-attention for interaction pattern refinement
        self.interaction_attention = MultiHeadCrossAttention(
            embed_dim=hidden_dims[2],
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Final user interaction embedding
        self.user_encoder = nn.Sequential(
            nn.Linear(hidden_dims[2], embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, user_features: torch.Tensor, user_bio_preferences: torch.Tensor) -> torch.Tensor:
        """
        Encode user interaction patterns
        
        Args:
            user_features: User demographic features (batch_size, feature_dim)
            user_bio_preferences: Learned bio preferences from liked profiles (batch_size, bio_embedding_dim)
            
        Returns:
            User interaction embedding (batch_size, embedding_dim)
        """
        # Process user demographics
        demographic_emb = self.demographic_layers(user_features)  # (batch, hidden_dim)
        
        # Process bio preferences learned from interactions
        bio_pref_emb = self.bio_preference_layers(user_bio_preferences)  # (batch, hidden_dim)
        
        # Fuse interaction patterns
        fused = torch.cat([demographic_emb, bio_pref_emb], dim=1)  # (batch, hidden_dim * 2)
        interaction_emb = self.interaction_fusion(fused)  # (batch, hidden_dim)
        
        # Add sequence dimension for self-attention
        interaction_seq = interaction_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Self-attention for interaction pattern refinement
        refined_interaction, _ = self.interaction_attention(
            interaction_seq, interaction_seq, interaction_seq
        )  # (batch, 1, hidden_dim)
        
        # Remove sequence dimension
        refined_interaction = refined_interaction.squeeze(1)  # (batch, hidden_dim)
        
        # Final user interaction embedding
        user_emb = self.user_encoder(refined_interaction)  # (batch, embedding_dim)
        
        return user_emb


class CandidateProfileTower(nn.Module):
    """
    Candidate Tower - Encodes candidate profiles
    Processes: age, bio, location, profile completeness
    """
    
    def __init__(
        self,
        feature_dim: int = 8,  # age, gender, lat, long, bio_length, has_bio, profile_completeness, activity_score
        bio_embedding_dim: int = 384,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2
    ):
        super(CandidateProfileTower, self).__init__()
        
        # Candidate demographic features
        self.demographic_layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Candidate bio content processing
        self.bio_content_layers = nn.Sequential(
            nn.Linear(bio_embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Profile fusion
        fusion_input_dim = hidden_dims[1] * 2
        self.profile_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, candidate_features: torch.Tensor, candidate_bio: torch.Tensor) -> torch.Tensor:
        """Encode candidate profile"""
        demographic_emb = self.demographic_layers(candidate_features)
        bio_emb = self.bio_content_layers(candidate_bio)
        
        fused = torch.cat([demographic_emb, bio_emb], dim=1)
        profile_emb = self.profile_fusion(fused)
        
        return profile_emb


class CrossAttentionRecommendationModel(nn.Module):
    """
    Cross-Attention Two-Tower Recommendation Model
    Learns from user interactions and recommends similar profiles with gender filtering
    """
    
    def __init__(
        self,
        user_feature_dim: int = 19,      # User features with learned patterns
        candidate_feature_dim: int = 10, # Candidate features
        bio_embedding_dim: int = 384,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64],
        num_attention_heads: int = 8,
        dropout: float = 0.2
    ):
        super(CrossAttentionRecommendationModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        
        # Two towers
        self.user_tower = UserInteractionTower(
            feature_dim=user_feature_dim,
            bio_embedding_dim=bio_embedding_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.candidate_tower = CandidateProfileTower(
            feature_dim=candidate_feature_dim,
            bio_embedding_dim=bio_embedding_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        # Cross-attention mechanism
        self.cross_attention = MultiHeadCrossAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Similarity matching layers
        self.similarity_matcher = nn.Sequential(
            nn.Linear(embedding_dim * 3, self.hidden_dims[2]),  # user + candidate + cross-attended
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dims[2], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        user_features: torch.Tensor,
        user_bio_preferences: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_bio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with cross-attention
        
        Returns:
            similarity_score: How similar candidate is to user's preferred profiles (0-1)
            attention_weights: Cross-attention weights for interpretability
        """
        # Encode user interaction patterns and candidate profiles
        user_emb = self.user_tower(user_features, user_bio_preferences)  # (batch, embed_dim)
        candidate_emb = self.candidate_tower(candidate_features, candidate_bio)  # (batch, embed_dim)
        
        # Add sequence dimension for cross-attention
        user_seq = user_emb.unsqueeze(1)      # (batch, 1, embed_dim)
        candidate_seq = candidate_emb.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Cross-attention: How does user's interaction pattern attend to candidate profile?
        user_attended_to_candidate, attention_weights = self.cross_attention(
            query=user_seq,
            key=candidate_seq,
            value=candidate_seq
        )
        user_attended_to_candidate = user_attended_to_candidate.squeeze(1)  # (batch, embed_dim)
        
        # L2 normalize embeddings
        user_emb_norm = F.normalize(user_emb, p=2, dim=1)
        candidate_emb_norm = F.normalize(candidate_emb, p=2, dim=1)
        cross_attended_norm = F.normalize(user_attended_to_candidate, p=2, dim=1)
        
        # Combine all representations
        combined = torch.cat([
            user_emb_norm,
            candidate_emb_norm,
            cross_attended_norm
        ], dim=1)  # (batch, embed_dim * 3)
        
        # Final similarity score
        similarity_score = self.similarity_matcher(combined).squeeze()
        
        return similarity_score, attention_weights.squeeze()
    
    def encode_user_interaction_pattern(
        self, 
        user_features: torch.Tensor, 
        user_bio_preferences: torch.Tensor
    ) -> torch.Tensor:
        """Encode user's interaction pattern for vector database"""
        user_emb = self.user_tower(user_features, user_bio_preferences)
        return F.normalize(user_emb, p=2, dim=1)
    
    def encode_candidate_profile(
        self, 
        candidate_features: torch.Tensor, 
        candidate_bio: torch.Tensor
    ) -> torch.Tensor:
        """Encode candidate profile for vector database"""
        candidate_emb = self.candidate_tower(candidate_features, candidate_bio)
        return F.normalize(candidate_emb, p=2, dim=1)
    
    def get_recommendation_explanation(
        self,
        user_features: torch.Tensor,
        user_bio_preferences: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_bio: torch.Tensor
    ) -> Dict:
        """Get detailed explanation of why candidate was recommended"""
        
        with torch.no_grad():
            similarity_score, attention_weights = self.forward(
                user_features, user_bio_preferences, candidate_features, candidate_bio
            )
            
            # Get individual embeddings
            user_emb = self.encode_user_interaction_pattern(user_features, user_bio_preferences)
            candidate_emb = self.encode_candidate_profile(candidate_features, candidate_bio)
            
            # Calculate cosine similarity
            cosine_sim = F.cosine_similarity(user_emb, candidate_emb, dim=1)
            
            explanation = {
                'similarity_score': float(similarity_score),
                'cosine_similarity': float(cosine_sim),
                'attention_weight': float(attention_weights.mean()),
                'recommendation_strength': 'high' if similarity_score > 0.8 else 'medium' if similarity_score > 0.6 else 'low'
            }
            
            return explanation


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = CrossAttentionRecommendationModel(
        feature_dim=8,
        bio_embedding_dim=384,
        embedding_dim=128,
        num_attention_heads=8
    )
    
    # Example data
    batch_size = 16
    user_features = torch.randn(batch_size, 8)  # age, gender, lat, long, etc.
    user_bio_preferences = torch.randn(batch_size, 384)  # learned from liked profiles
    candidate_features = torch.randn(batch_size, 8)
    candidate_bio = torch.randn(batch_size, 384)
    
    # Forward pass
    similarity_scores, attention_weights = model(
        user_features, user_bio_preferences, candidate_features, candidate_bio
    )
    
    print(f"Similarity scores shape: {similarity_scores.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Sample similarity scores: {similarity_scores[:5]}")
    
    # Get embeddings for vector database
    user_embeddings = model.encode_user_interaction_pattern(user_features, user_bio_preferences)
    candidate_embeddings = model.encode_candidate_profile(candidate_features, candidate_bio)
    
    print(f"User embeddings shape: {user_embeddings.shape}")
    print(f"Candidate embeddings shape: {candidate_embeddings.shape}")
    
    # Get recommendation explanation
    explanation = model.get_recommendation_explanation(
        user_features[:1], user_bio_preferences[:1], 
        candidate_features[:1], candidate_bio[:1]
    )
    print(f"Recommendation explanation: {explanation}")
