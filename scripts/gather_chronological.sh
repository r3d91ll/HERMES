#!/bin/bash
# Gather papers chronologically from 1998 onwards

echo "Starting chronological paper gathering from 1998..."
echo "This will download 50 papers per year, starting from the oldest papers"
echo ""

# Kill any existing arxiv gathering processes
pkill -f "gather_arxiv_papers.py" 2>/dev/null && echo "Stopped existing download process"

# Start chronological gathering
poetry run python gather_arxiv_papers.py \
    --chronological \
    --start-year 1998 \
    --papers-per-year 50 \
    --output-dir ./data/ml_papers_chronological

echo "Chronological gathering started!"