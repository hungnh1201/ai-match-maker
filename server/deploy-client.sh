#!/bin/bash

# AI Matchmaker Client Deployment Script
echo "🌐 Deploying AI Matchmaker Client..."

# Check if we're in the right directory
if [ ! -f "client/index.html" ]; then
    echo "❌ Error: Please run this script from the ai-matchmaker-deploy directory"
    exit 1
fi

echo "📦 Client files ready for deployment:"
echo "  ✅ index.html (Beautiful responsive UI)"
echo "  ✅ vercel.json (Vercel configuration)"

echo ""
echo "⚠️  IMPORTANT: Update API URL first!"
echo "Edit client/index.html line 275 with your Railway server URL:"
echo "  value=\"https://your-railway-app.railway.app\""

echo ""
echo "🎯 Next steps:"
echo "1. Update the API URL in client/index.html"
echo "2. Install Vercel CLI:"
echo "   npm i -g vercel"
echo ""
echo "3. Deploy to Vercel:"
echo "   cd client/"
echo "   vercel --prod"
echo ""
echo "4. Your client will be available at:"
echo "   https://your-app.vercel.app"

echo ""
echo "🎨 What will be deployed:"
echo "  - Modern dark theme with glassmorphism"
echo "  - Number-only input validation"
echo "  - Real-time AI recommendations display"
echo "  - Responsive design for all devices"
echo "  - Interactive animations and loading states"

echo ""
echo "🧪 Test the complete system:"
echo "  - Login: admin / password123"
echo "  - Test User: 5404674"
echo "  - Expect: Real AI recommendations!"
