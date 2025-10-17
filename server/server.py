"""
Simple Flask Server for AI Recommendation System
Provides user analysis and recommendations
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import glob
import json
from pathlib import Path
import logging
from geopy.distance import geodesic
import sys
import os

# Add the scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationServer:
    def __init__(self):
        self.profiles_df = None
        self.interactions_df = None
        self.load_data()

    def load_data(self):
        """Load the latest data files"""
        try:
            # Load latest profiles
            profile_files = glob.glob('outputs/data/user_profiles.parquet')
            if profile_files:
                latest_profile = max(profile_files, key=os.path.getctime)
                self.profiles_df = pd.read_parquet(latest_profile)
                logger.info(f"Loaded profiles: {len(self.profiles_df)} users")
            else:
                logger.warning(
                    "No profile files found, creating minimal dataset")
                self.create_minimal_dataset()

            # Load latest interactions
            interaction_files = glob.glob(
                'outputs/data/user_interactions.parquet')
            if interaction_files:
                latest_interaction = max(
                    interaction_files, key=os.path.getctime)
                self.interactions_df = pd.read_parquet(latest_interaction)
                logger.info(
                    f"Loaded interactions: {len(self.interactions_df)} interactions")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.info("Creating minimal dataset for deployment")
            self.create_minimal_dataset()

    def create_minimal_dataset(self):
        """Create minimal dataset for deployment when files are missing"""
        # Create sample profiles
        sample_profiles = [
            {"userid": 5404674, "age": 38, "gender": "female",
                "bio": "Love traveling and good food", "city_lat": 48.8566, "city_long": 2.3522},
            {"userid": 5404726, "age": 32, "gender": "female",
                "bio": "Yoga instructor and nature lover", "city_lat": 48.8566, "city_long": 2.3522},
            {"userid": 5404784, "age": 29, "gender": "female",
                "bio": "Artist and coffee enthusiast", "city_lat": 48.8566, "city_long": 2.3522},
        ]

        # Add male candidates
        for i in range(5405001, 5405021):
            sample_profiles.append({
                "userid": i,
                "age": np.random.randint(25, 50),
                "gender": "male",
                "bio": f"Demo male user {i}",
                "city_lat": 48.8566 + np.random.uniform(-0.1, 0.1),
                "city_long": 2.3522 + np.random.uniform(-0.1, 0.1)
            })

        self.profiles_df = pd.DataFrame(sample_profiles)

        # Create sample interactions
        sample_interactions = []
        for user_id in [5404674, 5404726, 5404784]:
            for candidate_id in range(5405001, 5405011):
                sample_interactions.append({
                    "user_id": user_id,
                    "candidate_id": candidate_id,
                    "action": np.random.choice(["accept_contact", "skip", "refuse_contact"], p=[0.1, 0.7, 0.2])
                })

        self.interactions_df = pd.DataFrame(sample_interactions)
        logger.info(
            f"Created minimal dataset: {len(self.profiles_df)} profiles, {len(self.interactions_df)} interactions")

    def get_user_profile(self, user_id):
        """Get user profile by ID"""
        if self.profiles_df is None:
            return None

        user_data = self.profiles_df[self.profiles_df['userid'] == user_id]
        if len(user_data) == 0:
            return None

        return user_data.iloc[0].to_dict()

    def analyze_user_behavior(self, user_id):
        """Analyze user interaction patterns with distance analysis"""
        if self.interactions_df is None:
            return {"error": "No interaction data available"}

        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]

        if len(user_interactions) == 0:
            return {
                "message": "No interaction history found for this user",
                "user_type": "cold_start",
                "total_interactions": 0
            }

        # Merge with candidate profiles to get full profile info
        # Check what columns are available in profiles_df
        available_columns = ['userid', 'age', 'gender']
        if 'city_lat' in self.profiles_df.columns:
            available_columns.append('city_lat')
        if 'city_long' in self.profiles_df.columns:
            available_columns.append('city_long')

        interactions_with_profiles = user_interactions.merge(
            self.profiles_df[available_columns].rename(
                columns={'userid': 'candidate_id'}),
            on='candidate_id',
            how='left'
        )

        # Get user's location for distance calculations
        user_profile = self.get_user_profile(user_id)
        user_lat = user_profile.get('city_lat') if user_profile else None
        user_lon = user_profile.get('city_long') if user_profile else None

        # Calculate distances if location data is available
        distance_analysis = {}
        if user_lat and user_lon and 'city_lat' in interactions_with_profiles.columns:
            from geopy.distance import geodesic

            distances = []
            for _, row in interactions_with_profiles.iterrows():
                if pd.notna(row['city_lat']) and pd.notna(row['city_long']):
                    try:
                        distance = geodesic(
                            (user_lat, user_lon),
                            (row['city_lat'], row['city_long'])
                        ).kilometers
                        distances.append({
                            'candidate_id': row['candidate_id'],
                            'action': row['action'],
                            'distance_km': round(distance, 2)
                        })
                    except Exception:
                        continue

            if distances:
                # Analyze distance patterns by action
                distance_df = pd.DataFrame(distances)
                distance_analysis = {
                    "average_distance_by_action": {},
                    "distance_ranges": {},
                    "total_with_distance": len(distances)
                }

                for action in distance_df['action'].unique():
                    action_distances = distance_df[distance_df['action']
                                                   == action]['distance_km']
                    if len(action_distances) > 0:
                        distance_analysis["average_distance_by_action"][action] = {
                            "avg_km": round(action_distances.mean(), 2),
                            "min_km": round(action_distances.min(), 2),
                            "max_km": round(action_distances.max(), 2),
                            "count": len(action_distances)
                        }

                # Distance range analysis
                distance_df['distance_range'] = pd.cut(
                    distance_df['distance_km'],
                    bins=[0, 5, 15, 30, 50, 100, float('inf')],
                    labels=['0-5km', '5-15km', '15-30km',
                            '30-50km', '50-100km', '100km+']
                )

                for action in distance_df['action'].unique():
                    action_data = distance_df[distance_df['action'] == action]
                    if len(action_data) > 0:
                        if action not in distance_analysis["distance_ranges"]:
                            distance_analysis["distance_ranges"][action] = {}
                        distance_analysis["distance_ranges"][action] = action_data['distance_range'].value_counts(
                        ).to_dict()

        # Analyze patterns
        analysis = {
            "user_type": "existing_user",
            "total_interactions": len(user_interactions),
            "action_breakdown": user_interactions['action'].value_counts().to_dict(),
            "age_preferences": {},
            "gender_interactions": interactions_with_profiles['gender'].value_counts().to_dict() if 'gender' in interactions_with_profiles.columns else {},
            "distance_patterns": distance_analysis,
            "interaction_timeline": {
                "first_interaction": user_interactions['timestamp'].min() if 'timestamp' in user_interactions.columns else None,
                "last_interaction": user_interactions['timestamp'].max() if 'timestamp' in user_interactions.columns else None,
                "days_active": None
            }
        }

        # Calculate days active if timestamp data is available
        if 'timestamp' in user_interactions.columns:
            try:
                first_date = pd.to_datetime(
                    user_interactions['timestamp'].min())
                last_date = pd.to_datetime(
                    user_interactions['timestamp'].max())
                analysis["interaction_timeline"]["days_active"] = (
                    last_date - first_date).days + 1
            except Exception:
                pass

        # Age group analysis
        if 'age' in interactions_with_profiles.columns:
            interactions_with_profiles['age_group'] = pd.cut(
                interactions_with_profiles['age'],
                bins=[0, 25, 30, 35, 40, 45, 50, 100],
                labels=['18-25', '26-30', '31-35',
                        '36-40', '41-45', '46-50', '50+']
            )

            for action in ['accept_contact', 'skip', 'refuse_contact']:
                action_data = interactions_with_profiles[interactions_with_profiles['action'] == action]
                if len(action_data) > 0:
                    analysis["age_preferences"][action] = action_data['age_group'].value_counts(
                    ).to_dict()

            # Detailed age statistics
            candidate_ages = interactions_with_profiles['age'].dropna()
            if len(candidate_ages) > 0:
                analysis["candidate_age_stats"] = {
                    "min_age": int(candidate_ages.min()),
                    "max_age": int(candidate_ages.max()),
                    "average_age": round(candidate_ages.mean(), 1),
                    "median_age": round(candidate_ages.median(), 1),
                    "total_candidates": len(candidate_ages)
                }

                # Age statistics by action
                analysis["age_by_action"] = {}
                for action in interactions_with_profiles['action'].unique():
                    action_data = interactions_with_profiles[interactions_with_profiles['action'] == action]
                    action_ages = action_data['age'].dropna()
                    if len(action_ages) > 0:
                        analysis["age_by_action"][action] = {
                            "count": len(action_ages),
                            "min_age": int(action_ages.min()),
                            "max_age": int(action_ages.max()),
                            "average_age": round(action_ages.mean(), 1)
                        }

        # Age gap analysis (candidate age - user age)
        if user_profile and 'age' in user_profile and 'age' in interactions_with_profiles.columns:
            user_age = user_profile['age']
            interactions_with_profiles['age_gap'] = interactions_with_profiles['age'] - user_age

            age_gaps = interactions_with_profiles['age_gap'].dropna()
            if len(age_gaps) > 0:
                analysis["age_gap_analysis"] = {
                    "average_gap": round(age_gaps.mean(), 1),
                    "median_gap": round(age_gaps.median(), 1),
                    "min_gap": int(age_gaps.min()),
                    "max_gap": int(age_gaps.max()),
                    "user_age": int(user_age),
                    "description": f"Candidates are on average {abs(round(age_gaps.mean(), 1))} years {'older' if age_gaps.mean() > 0 else 'younger'} than user"
                }

                # Age gap by action
                analysis["age_gap_by_action"] = {}
                for action in interactions_with_profiles['action'].unique():
                    action_data = interactions_with_profiles[interactions_with_profiles['action'] == action]
                    action_gaps = action_data['age_gap'].dropna()
                    if len(action_gaps) > 0:
                        analysis["age_gap_by_action"][action] = {
                            "count": len(action_gaps),
                            "average_gap": round(action_gaps.mean(), 1),
                            "median_gap": round(action_gaps.median(), 1)
                        }

        return analysis

    def get_recommendations(self, user_id, top_k=5):
        """Get recommendations for a user"""
        try:
            return self.use_full_ai_model(user_id, top_k)
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")

    def use_full_ai_model(self, user_id, top_k=5):
        """Generate recommendations using pattern-aware engine (supports cold-start users)"""
        try:
            # Import the recommendation script
            from step5_generate_recommendations import GenderAwareRecommendationEngine

            # Find latest vector database
            vector_db_dirs = glob.glob('outputs/vector_db')
            if not vector_db_dirs:
                logger.warning("No vector database found in outputs/vector_db")
                return None

            # Initialize recommendation engine with pattern-aware filtering
            # This supports both cold-start and existing users with bio-dominant strategy
            engine = GenderAwareRecommendationEngine(
                vector_db_dir=vector_db_dirs[0],
                profiles_path=glob.glob(
                    'outputs/data/user_profiles.parquet')[-1],
                interactions_path=glob.glob(
                    'outputs/data/user_interactions.parquet')[-1]
            )

            # Get recommendations
            raw_recommendations = engine.get_recommendations(
                user_id=user_id, top_k=top_k)

            # Transform the AI model response to match client expectations
            recommendations = []
            for rec in raw_recommendations:
                # The AI model returns 'user_id' but we need 'candidate_id'
                candidate_id = rec.get('user_id', rec.get('candidate_id'))

                # Extract distance from explanation if available
                distance_km = "Unknown"
                if 'explanation' in rec and 'location_compatibility' in rec['explanation']:
                    distance_km = rec['explanation']['location_compatibility'].get(
                        'distance_km', "Unknown")

                # Create standardized recommendation format
                standardized_rec = {
                    'candidate_id': candidate_id,
                    'age': rec.get('age', 0),
                    'bio': rec.get('bio', ''),
                    'similarity_score': round(rec.get('similarity_score', 0), 3),
                    'distance_km': distance_km
                }

                recommendations.append(standardized_rec)

            return recommendations

        except Exception as e:
            logger.error(f"Full AI model not available: {e}")
            return None


# Initialize server
rec_server = RecommendationServer()

# Predefined credentials
VALID_CREDENTIALS = {
    "admin": "password123",
    "demo": "demo123",
    "test": "test123"
}


@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/login', methods=['POST'])
def login():
    """Handle login requests"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    print(username, password)
    if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
        return jsonify({"success": True, "message": "Login successful"})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401


@app.route('/api/analyze', methods=['POST'])
def analyze_user():
    """Analyze user and get recommendations with detailed user type detection"""
    data = request.get_json()
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({"error": "User ID must be a number"}), 400

    # Check if user exists in profiles
    user_profile = rec_server.get_user_profile(user_id)

    # Analyze behavior (this will determine if user is cold_start or existing)
    behavior_analysis = rec_server.analyze_user_behavior(user_id)

    # Determine user status
    user_status = {
        "exists_in_profiles": user_profile is not None,
        "user_type": behavior_analysis.get("user_type", "unknown"),
        "profile_completeness": "complete" if user_profile else "missing"
    }

    # Handle different user scenarios
    if not user_profile:
        # User ID not found in profiles - return error
        return jsonify({
            "error": "User not found",
            "message": f"User ID {user_id} does not exist in our database.",
            "suggestion": "Please check the user ID or create a profile for this user first.",
            "user_id": user_id
        }), 404

    # User exists in profiles
    if behavior_analysis.get("user_type") == "cold_start":
        # User has profile but no interactions yet
        user_status["user_type"] = "cold_start"
        user_status["message"] = "User has a profile but no interaction history yet."
    else:
        # Existing user with interaction history
        user_status["user_type"] = "existing_user"
        user_status["message"] = f"Active user with {behavior_analysis.get('total_interactions', 0)} interactions."

    # Get recommendations
    recommendations = rec_server.get_recommendations(user_id, top_k=5)

    # Enhanced user profile information
    enhanced_profile = {
        "userid": user_profile['userid'],
        "age": user_profile['age'],
        "gender": user_profile['gender'],
        "bio": user_profile['bio']
    }

    # Add location info if available
    if 'city_lat' in user_profile and 'city_long' in user_profile:
        enhanced_profile["location"] = {
            "latitude": user_profile['city_lat'],
            "longitude": user_profile['city_long'],
            "has_location": True
        }
    else:
        enhanced_profile["location"] = {"has_location": False}

    # Determine recommendation confidence based on user type
    interaction_count = behavior_analysis.get("total_interactions", 0)
    has_recommendations = recommendations is not None and len(
        recommendations) > 0

    # Cold-start users get recommendations based on bio similarity
    # Existing users get pattern-aware recommendations
    if interaction_count > 10:
        rec_confidence = "high"
        rec_strategy = "pattern-aware (learned preferences)"
    elif interaction_count > 0:
        rec_confidence = "medium"
        rec_strategy = "pattern-learning (emerging preferences)"
    else:
        rec_confidence = "medium" if has_recommendations else "low"
        rec_strategy = "bio-dominant cold-start (loose filters)" if has_recommendations else "no recommendations"

    return jsonify({
        "user_status": user_status,
        "user_profile": enhanced_profile,
        "behavior_analysis": behavior_analysis,
        "recommendations": recommendations,
        "analysis_summary": {
            "can_generate_ai_recommendations": has_recommendations,
            "recommendation_confidence": rec_confidence,
            "recommendation_strategy": rec_strategy,
            "data_completeness": {
                "has_profile": True,
                "has_interactions": interaction_count > 0,
                "has_location": enhanced_profile["location"]["has_location"],
                "interaction_count": interaction_count
            }
        }
    })


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "profiles_loaded": len(rec_server.profiles_df) if rec_server.profiles_df is not None else 0,
        "interactions_loaded": len(rec_server.interactions_df) if rec_server.interactions_df is not None else 0
    })


if __name__ == '__main__':
    # Create templates directory
    Path('templates').mkdir(exist_ok=True)

    # Get port from environment variable (for Railway, Heroku, etc.)
    import os
    port = int(os.environ.get('PORT', 8080))

    print("🚀 Starting AI Matchmaker Server...")
    print(f"📊 Server will be available on port: {port}")
    print("🔑 Login credentials:")
    for username, password in VALID_CREDENTIALS.items():
        print(f"   Username: {username}, Password: {password}")

    # Use debug=False for production
    debug_mode = os.environ.get('APP_ENV', 'development') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
