#!/bin/bash
# Complete repository setup script for traffic-demographic-analysis project
# Run this in your cloned repository directory

set -e  # Exit on any error

echo "🚀 Setting up traffic-demographic-analysis repository..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directory structure
echo -e "${BLUE}📁 Creating directory structure...${NC}"
mkdir -p configs
mkdir -p src/{models,data,analysis,visualization}
mkdir -p scripts notebooks docs tests outputs/figures
mkdir -p data/raw/{cameras,census,lodes}
mkdir -p data/processed/{detections,demographics}
mkdir -p data/results/{statistics,visualizations}
mkdir -p models/{openseed,yolov8}

# Create .gitkeep files to preserve empty directories
echo -e "${BLUE}📌 Creating .gitkeep files...${NC}"
touch data/.gitkeep
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/results/.gitkeep
touch models/.gitkeep
touch outputs/figures/.gitkeep

# Create __init__.py files for Python packages
echo -e "${BLUE}🐍 Creating Python package structure...${NC}"
touch src/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/analysis/__init__.py
touch src/visualization/__init__.py
touch tests/__init__.py

# Create documentation placeholders
echo -e "${BLUE}📚 Creating documentation files...${NC}"
cat > docs/setup_instructions.md << 'EOF'
# Setup Instructions

## Prerequisites
- Ubuntu 22.04 LTS
- NVIDIA GPU with 12GB+ VRAM
- CUDA 11.7+ or 12.1+
- Python 3.8+

## Installation

See main README.md for installation steps.
EOF

cat > docs/data_sources.md << 'EOF'
# Data Sources

## Traffic Camera Data
- **Source**: Caltrans CWWP2 API
- **URL**: https://cwwp2.dot.ca.gov/
- **Coverage**: District 8 (Riverside & San Bernardino Counties)

## Demographic Data
- **Source**: U.S. Census Bureau American Community Survey
- **API**: https://api.census.gov/data
- **Years**: 2018-2022 5-year estimates

## Employment Data
- **Source**: LEHD LODES
- **URL**: https://lehd.ces.census.gov/data/
- **Version**: LODES 8

See individual data collection scripts for detailed documentation.
EOF

cat > docs/methodology.md << 'EOF'
# Methodology

## Overview
This project combines computer vision and spatial statistics to analyze traffic patterns.

## Workflow
1. Data Collection (4-8 weeks)
2. Vehicle Detection (OpenSeeD or YOLOv8)
3. Demographic Integration (Census ACS + LODES)
4. Spatial Correlation Analysis
5. Statistical Modeling

## Statistical Methods
- Spatial autocorrelation (Moran's I)
- Ordinary least squares regression
- Spatial lag/error models
- Time series decomposition

See notebooks for detailed implementation.
EOF

cat > docs/results_summary.md << 'EOF'
# Results Summary

## Key Findings

(To be completed after analysis)

## Visualizations

See `outputs/figures/` for all plots and maps.

## Statistical Models

See `data/results/statistics/` for detailed model outputs.
EOF

# Create README for outputs
cat > outputs/README.md << 'EOF'
# Outputs Directory

This directory contains final deliverables:

- `figures/` - Publication-quality visualizations
- `report.pdf` - Final project report
- `presentation.pptx` - Presentation slides

Note: PDF and PPTX files are gitignored. Commit final versions manually if needed.
EOF

# Create example test file
cat > tests/test_models.py << 'EOF'
"""
Example unit tests for model implementations
"""
import pytest

def test_placeholder():
    """Placeholder test - replace with actual tests"""
    assert True

# Add actual tests for your models here
# Example:
# def test_openseed_analyzer_initialization():
#     from src.models.openseed_analyzer import OpenSeeDATrafficAnalyzer
#     analyzer = OpenSeeDATrafficAnalyzer(...)
#     assert analyzer is not None
EOF

# Create example notebook
cat > notebooks/00_template.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Template Notebook\n",
    "\n",
    "Use this as a starting point for analysis notebooks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Your analysis code here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo ""
echo -e "${GREEN}✅ Directory structure created!${NC}"
echo ""
echo -e "${BLUE}📋 Repository structure:${NC}"
tree -L 2 -I '__pycache__|*.pyc' || ls -R

echo ""
echo -e "${GREEN}✨ Setup complete!${NC}"
echo ""
echo -e "${BLUE}📝 Next steps:${NC}"
echo "  1. Review the created files and directories"
echo "  2. Copy the .gitignore, requirements.txt, and environment.yml from the guide"
echo "  3. Update README.md with your information"
echo "  4. Run: git add ."
echo "  5. Run: git commit -m 'Initial project structure'"
echo "  6. Run: git push"
echo ""
echo -e "${GREEN}Happy coding! 🎉${NC}"
