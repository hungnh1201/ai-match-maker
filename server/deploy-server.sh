#!/bin/bash

# AI Matchmaker Server Deployment Script
echo "🚀 Deploying AI Matchmaker Server..."

# Check if we're in the right directory
if [ ! -f "server/server.py" ]; then
    echo "❌ Error: Please run this script from the ai-matchmaker-deploy directory"
    exit 1
fi

echo "📦 Server files ready for deployment:"
echo "  ✅ server.py (Flask application)"
echo "  ✅ Dockerfile (Production container)"
echo "  ✅ requirements.txt (Python dependencies)"
echo "  ✅ railway.json (Railway configuration)"
echo "  ✅ Data files (15,144 users, 180,348 interactions)"
echo "  ✅ AI models (trained neural network)"

echo ""
echo "🎯 Next steps:"
echo "1. Create a new Git repository with the server/ folder contents"
echo "2. Push to GitHub:"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'AI Matchmaker Server'"
echo "   git push origin main"
echo ""
echo "3. Deploy to Railway:"
echo "   - Go to https://railway.app"
echo "   - Click 'Deploy from GitHub repo'"
echo "   - Select your repository"
echo "   - Set environment: APP_ENV=production"
echo ""
echo "4. Your server will be available at:"
echo "   https://your-app.railway.app"

echo ""
echo "📊 What will be deployed:"
echo "  - Full AI recommendation engine"
echo "  - 15,144 real user profiles"
echo "  - 180,348 real interactions"
echo "  - Trained ML model with vector database"
echo "  - Beautiful web interface"
echo "  - Health checks and monitoring"
