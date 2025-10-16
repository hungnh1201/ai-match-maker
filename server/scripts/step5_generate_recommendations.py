"""
Step 5: Generate Gender-Aware Recommendations
Final recommendation system with proper gender filtering and similarity matching
"""

import logging
from typing import List, Dict, Tuple
from datetime import datetime
import argparse
import faiss
import pickle
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenderAwareRecommendationEngine:
    """
    Final recommendation engine with proper gender filtering
    Female users → Male profiles similar to profiles they liked
    Male users → Female profiles similar to profiles they liked
    """

    def __init__(self, vector_db_dir: str, profiles_path: str):
        """
        Initialize gender-aware recommendation engine

        Args:
            vector_db_dir: Directory containing gender-aware vector database
            profiles_path: Path to profiles data
        """
        self.vector_db_dir = Path(vector_db_dir)
        # Load configuration
        config_file = self.vector_db_dir / "latest_config.json"
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # Load user embeddings
        user_embeddings_file = self.vector_db_dir / "latest_user_embeddings.pkl"
        with open(user_embeddings_file, 'rb') as f:
            self.user_embeddings = pickle.load(f)

        # Load gender-separated indexes and candidate IDs
        self._load_gender_indexes()

        # Load profiles
        logger.info("Loading profiles...")
        if profiles_path.endswith('.parquet'):
            self.profiles_df = pd.read_parquet(profiles_path)
        else:
            self.profiles_df = pd.read_csv(profiles_path)

        self.profiles_df = self.profiles_df.set_index('userid')

        logger.info(f"✅ Gender-aware recommendation engine ready!")
        logger.info(f"   Users: {len(self.user_embeddings):,}")
        logger.info(f"   Male candidates: {len(self.male_candidate_ids):,}")
        logger.info(
            f"   Female candidates: {len(self.female_candidate_ids):,}")
        logger.info(f"   Profiles: {len(self.profiles_df):,}")

    def _load_gender_indexes(self):
        """Load gender-separated FAISS indexes"""

        # Load male candidates index and IDs
        if self.config['files']['male_index']:
            male_index_file = self.vector_db_dir / "latest_male_index.faiss"
            self.male_index = faiss.read_index(str(male_index_file))
        else:
            self.male_index = None

        male_ids_file = self.vector_db_dir / "latest_male_candidate_ids.pkl"
        with open(male_ids_file, 'rb') as f:
            self.male_candidate_ids = pickle.load(f)

        # This system only uses male candidates for female users
        self.female_index = None
        self.female_candidate_ids = []

    def get_recommendations(
        self,
        user_id: int,
        top_k: int = 20,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Get gender-appropriate recommendations for a user

        Args:
            user_id: User to get recommendations for
            top_k: Number of recommendations to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of recommended profiles with similarity scores and explanations
        """

        # Check if user exists
        if user_id not in self.user_embeddings:
            logger.error(f"User {user_id} not found in embeddings")
            return []

        if user_id not in self.profiles_df.index:
            logger.error(f"User {user_id} not found in profiles")
            return []

        # Get user info
        user_data = self.user_embeddings[user_id]
        user_gender = user_data['gender']
        user_profile = self.profiles_df.loc[user_id]

        logger.info(
            f"User {user_id} is {user_gender}, searching for {self._get_target_gender(user_gender)} candidates")

        # Search for gender-appropriate candidates (search more to account for filtering)
        search_k = max(top_k * 5, 100)  # Search at least 5x more candidates or minimum 100
        similar_candidates = self._search_gender_appropriate_candidates(
            user_id, search_k)
        
        logger.info(f"Found {len(similar_candidates)} similar candidates before filtering")

        if not similar_candidates:
            logger.warning(f"No similar candidates found for user {user_id}")
            return []

        # Filter and enrich recommendations
        recommendations = self._create_recommendations(
            user_id=user_id,
            user_gender=user_gender,
            user_profile=user_profile,
            similar_candidates=similar_candidates,
            min_similarity=min_similarity,
            top_k=top_k
        )

        logger.info(
            f"✅ Generated {len(recommendations)} gender-appropriate recommendations")

        return recommendations

    def _search_gender_appropriate_candidates(self, user_id: int, search_k: int) -> List[Tuple[int, float]]:
        """Search for gender-appropriate candidates"""

        user_data = self.user_embeddings[user_id]
        user_gender = user_data['gender']
        user_embedding = user_data['embedding'].reshape(
            1, -1).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(user_embedding)

        # This system only supports female users getting male candidates
        if user_gender == 'female':
            # Female user gets male candidates
            target_index = self.male_index
            target_candidate_ids = self.male_candidate_ids
            target_gender = "male"
        else:
            logger.error(f"This system only supports female users. User {user_id} is {user_gender}")
            return []

        if target_index is None:
            logger.error(f"No male candidates index available")
            return []

        # Search for similar candidates
        logger.info(f"Searching for {search_k} candidates in {target_gender} index (size: {target_index.ntotal})")
        
        # For IVF indexes, ensure nprobe is set high enough
        if hasattr(target_index, 'nprobe'):
            original_nprobe = target_index.nprobe
            # Increase nprobe if it's too low to get more results
            min_nprobe = min(32, max(16, target_index.ntotal // 100))
            if target_index.nprobe < min_nprobe:
                target_index.nprobe = min_nprobe
                logger.info(f"Increased nprobe from {original_nprobe} to {target_index.nprobe} for better results")
        
        similarities, indices = target_index.search(user_embedding, search_k)
        
        logger.info(f"FAISS returned {len(indices[0])} indices, {len(similarities[0])} similarities")
        valid_results = sum(1 for idx in indices[0] if idx != -1)
        logger.info(f"Valid results (idx != -1): {valid_results}")

        # Convert to results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx != -1:  # Valid result
                candidate_id = target_candidate_ids[idx]
                results.append((candidate_id, float(similarity)))

        logger.info(f"Final results after conversion: {len(results)}")
        return results

    def _create_recommendations(
        self,
        user_id: int,
        user_gender: str,
        user_profile: pd.Series,
        similar_candidates: List[Tuple[int, float]],
        min_similarity: float,
        top_k: int
    ) -> List[Dict]:
        """Create detailed recommendation entries"""

        recommendations = []
        target_gender = self._get_target_gender(user_gender)
        
        filtered_counts = {
            'similarity_threshold': 0,
            'self_match': 0,
            'profile_missing': 0,
            'gender_mismatch': 0,
            'accepted': 0
        }

        for candidate_id, similarity_score in similar_candidates:
            # Skip if below similarity threshold
            if similarity_score < min_similarity:
                filtered_counts['similarity_threshold'] += 1
                continue

            # Skip self (shouldn't happen with gender filtering, but safety check)
            if candidate_id == user_id:
                filtered_counts['self_match'] += 1
                continue

            # Check if candidate profile exists
            if candidate_id not in self.profiles_df.index:
                filtered_counts['profile_missing'] += 1
                continue

            candidate_profile = self.profiles_df.loc[candidate_id]

            # Verify gender (double-check)
            if candidate_profile['gender'] != target_gender:
                filtered_counts['gender_mismatch'] += 1
                logger.warning(
                    f"Gender mismatch: expected {target_gender}, got {candidate_profile['gender']}")
                continue
            
            filtered_counts['accepted'] += 1

            # Create recommendation entry
            recommendation = {
                'user_id': int(candidate_id),
                'similarity_score': float(similarity_score),
                'gender': candidate_profile['gender'],
                'age': int(candidate_profile['age']),
                'bio': self._truncate_bio(str(candidate_profile['bio'])),
                'city_lat': float(candidate_profile['city_lat']),
                'city_long': float(candidate_profile['city_long']),
                'city_name': candidate_profile.get('city_name', 'Unknown'),
                'recommendation_type': 'cross_attention_gender_aware',
                'explanation': self._generate_explanation(
                    user_id, user_profile, candidate_id, candidate_profile, similarity_score
                )
            }

            recommendations.append(recommendation)

            # Stop when we have enough
            if len(recommendations) >= top_k:
                break

        # Log filtering statistics
        logger.info(f"Filtering results for user {user_id}:")
        logger.info(f"  - Similarity threshold filtered: {filtered_counts['similarity_threshold']}")
        logger.info(f"  - Self matches filtered: {filtered_counts['self_match']}")
        logger.info(f"  - Missing profiles filtered: {filtered_counts['profile_missing']}")
        logger.info(f"  - Gender mismatches filtered: {filtered_counts['gender_mismatch']}")
        logger.info(f"  - Accepted candidates: {filtered_counts['accepted']}")
        logger.info(f"  - Final recommendations: {len(recommendations)}")

        return recommendations

    def _generate_explanation(
        self,
        user_id: int,
        user_profile: pd.Series,
        candidate_id: int,
        candidate_profile: pd.Series,
        similarity_score: float
    ) -> Dict:
        """Generate explanation for why this candidate was recommended"""

        # Create detailed explanation based on profile analysis
        age_diff = abs(candidate_profile['age'] - user_profile['age'])
        bio_similarity = "meaningful bio" if len(
            str(candidate_profile['bio'])) > 50 else "short bio"

        # Calculate distance if coordinates available
        distance_info = ""
        if user_profile['city_lat'] != 0 and candidate_profile['city_lat'] != 0:
            distance = self._calculate_distance(
                user_profile['city_lat'], user_profile['city_long'],
                candidate_profile['city_lat'], candidate_profile['city_long']
            )
            if distance <= 25:
                distance_info = f"very close ({distance:.1f}km)"
            elif distance <= 50:
                distance_info = f"nearby ({distance:.1f}km)"
            elif distance <= 100:
                distance_info = f"moderate distance ({distance:.1f}km)"
            else:
                distance_info = f"distant ({distance:.1f}km)"

        # Create detailed explanation
        explanation_parts = []
        explanation_parts.append(
            f"Recommended {candidate_profile['gender']} profile")

        if age_diff <= 5:
            explanation_parts.append(
                f"very similar age ({candidate_profile['age']} vs your {user_profile['age']})")
        elif age_diff <= 10:
            explanation_parts.append(
                f"compatible age range ({candidate_profile['age']} vs your {user_profile['age']})")
        else:
            explanation_parts.append(
                f"different age ({candidate_profile['age']} vs your {user_profile['age']})")

        if distance_info:
            explanation_parts.append(distance_info)

        explanation_parts.append(f"with {bio_similarity}")

        # Add pattern matching info
        if similarity_score > 1.04:
            explanation_parts.append(
                "- strong pattern match to profiles you've interacted with")
        elif similarity_score > 1.02:
            explanation_parts.append(
                "- good pattern match to your interaction history")
        else:
            explanation_parts.append(
                "- moderate pattern match based on your preferences")

        detailed_explanation = ", ".join(explanation_parts)

        explanation = {
            'method': 'cross_attention_gender_aware',
            'similarity_score': float(similarity_score),
            'user_gender': user_profile['gender'],
            'candidate_gender': candidate_profile['gender'],
            'explanation_text': detailed_explanation,
            'confidence': 'high' if similarity_score > 1.04 else 'medium' if similarity_score > 1.02 else 'low'
        }

        # Add detailed comparisons
        try:
            # Age comparison
            age_diff = abs(candidate_profile['age'] - user_profile['age'])
            explanation['age_compatibility'] = {
                'user_age': int(user_profile['age']),
                'candidate_age': int(candidate_profile['age']),
                'age_difference': int(age_diff),
                'compatibility': 'excellent' if age_diff <= 5 else 'good' if age_diff <= 10 else 'fair'
            }

            # Location comparison
            if user_profile['city_lat'] != 0 and candidate_profile['city_lat'] != 0:
                distance = self._calculate_distance(
                    user_profile['city_lat'], user_profile['city_long'],
                    candidate_profile['city_lat'], candidate_profile['city_long']
                )
                explanation['location_compatibility'] = {
                    'distance_km': round(distance, 1),
                    'user_city': user_profile.get('city_name', 'Unknown'),
                    'candidate_city': candidate_profile.get('city_name', 'Unknown'),
                    'compatibility': 'excellent' if distance <= 50 else 'good' if distance <= 100 else 'fair'
                }

            # Bio length comparison (as a proxy for communication style)
            user_bio_length = len(str(user_profile['bio']))
            candidate_bio_length = len(str(candidate_profile['bio']))

            explanation['communication_style'] = {
                'user_bio_length': user_bio_length,
                'candidate_bio_length': candidate_bio_length,
                'style_match': 'similar' if abs(user_bio_length - candidate_bio_length) < 100 else 'different'
            }

        except Exception as e:
            logger.warning(f"Could not generate detailed explanation: {e}")

        return explanation

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km"""
        from math import radians, sin, cos, sqrt, atan2

        R = 6371  # Earth radius in km

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    def _get_target_gender(self, user_gender: str) -> str:
        """Get target gender for recommendations - only supports female users"""
        if user_gender == 'female':
            return 'male'
        else:
            raise ValueError(f"This system only supports female users, got: {user_gender}")

    def _truncate_bio(self, bio: str, max_length: int = 100) -> str:
        """Truncate bio text for display"""
        if len(bio) <= max_length:
            return bio
        return bio[:max_length] + "..."

    def batch_recommendations(
        self,
        user_ids: List[int],
        top_k: int = 20,
        **kwargs
    ) -> Dict[int, List[Dict]]:
        """Generate recommendations for multiple users"""

        results = {}

        for user_id in user_ids:
            try:
                recommendations = self.get_recommendations(
                    user_id, top_k, **kwargs)
                results[user_id] = recommendations
            except Exception as e:
                logger.error(
                    f"Error generating recommendations for user {user_id}: {e}")
                results[user_id] = []

        return results

    def get_statistics(self) -> Dict:
        """Get recommendation engine statistics"""

        # Analyze user gender distribution
        male_users = sum(
            1 for data in self.user_embeddings.values() if data['gender'] == 'male')
        female_users = sum(
            1 for data in self.user_embeddings.values() if data['gender'] == 'female')

        stats = {
            'total_users': len(self.user_embeddings),
            'male_users': male_users,
            'female_users': female_users,
            'male_candidates': len(self.male_candidate_ids),
            'female_candidates': 0,  # System only uses male candidates
            'total_profiles': len(self.profiles_df),
            'gender_separated': True,
            'recommendation_logic': {
                'female_users_get': 'male_candidates',
                'supported_users': 'female_only'
            }
        }

        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate gender-aware recommendations')
    parser.add_argument('--vector-db-dir', type=str, required=True,
                        help='Directory containing gender-aware vector database')
    parser.add_argument('--profiles-path', type=str, required=True,
                        help='Path to profiles data')
    parser.add_argument('--user-id', type=int, required=True,
                        help='User ID to generate recommendations for')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of recommendations to generate')
    parser.add_argument('--min-similarity', type=float, default=0.0,
                        help='Minimum similarity threshold')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Optional output file to save recommendations')

    args = parser.parse_args()

    print("🚀 GENERATING RECOMMENDATIONS")
    print("Loading recommendation engine...")

    # Initialize recommendation engine
    engine = GenderAwareRecommendationEngine(
        vector_db_dir=args.vector_db_dir,
        profiles_path=args.profiles_path
    )

    print(f"✅ Ready! {len(engine.user_embeddings):,} users, {len(engine.profiles_df):,} profiles")

    # Generate recommendations
    recommendations = engine.get_recommendations(
        user_id=args.user_id,
        top_k=args.top_k,
        min_similarity=args.min_similarity
    )

    # Display results
    if recommendations:
        print(f"\n🎯 TOP {len(recommendations)} RECOMMENDATIONS FOR USER {args.user_id}")
        user_info = engine.profiles_df.loc[args.user_id] if args.user_id in engine.profiles_df.index else None
        if user_info is not None:
            print(f"User: {user_info['gender'].upper()}, Age: {user_info['age']}")
        print("-" * 60)

        for i, rec in enumerate(recommendations[:5], 1):  # Show only top 5
            print(f"{i}. User {rec['user_id']} | Score: {rec['similarity_score']:.3f}")
            print(f"   {rec['gender'].upper()}, Age: {rec['age']}, {rec['city_name']}")
            print(f"   {rec['bio'][:80]}{'...' if len(rec['bio']) > 80 else ''}")
            print()

        if len(recommendations) > 5:
            print(f"... and {len(recommendations) - 5} more recommendations")

    else:
        print(f"\n❌ No recommendations found for user {args.user_id}")

    # Save to file
    output_file = args.output_file or f"outputs/recommendations_user_{args.user_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Generated {len(recommendations)} recommendations → {output_file}")


if __name__ == "__main__":
    main()
