#!/bin/bash
echo "Setting up FinRAG..."

# Setup backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Please enter your GOOGLE_API_KEY (or press enter to use the existing one if set):"
    read -r GOOGLE_API_KEY
    if [ ! -z "$GOOGLE_API_KEY" ]; then
        # Use awk or sed to replace the placeholder
        sed -i '' "s/GOOGLE_API_KEY=.*/GOOGLE_API_KEY=\"$GOOGLE_API_KEY\"/" .env 2>/dev/null || sed -i "s/GOOGLE_API_KEY=.*/GOOGLE_API_KEY=\"$GOOGLE_API_KEY\"/" .env
    fi
fi

# Setup frontend
cd finrag-ui
npm install
if [ ! -f .env.local ]; then
    cp .env.example .env.local
fi
cd ..

echo "Setup complete! Run the project with:"
echo "Terminal 1: cd finrag-ui && npm run dev"
echo "Terminal 2: source .venv/bin/activate && uvicorn finrag.api.app:create_app --factory --reload --port 8002 --env-file .env"
