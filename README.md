# traffic-demographic-analysis
computer vision analysis of traffic patterns correlated with demographic data.


# Traffic Pattern Analysis for Riverside & San Bernardino Counties

Computer vision-based traffic analysis correlated with demographic data to understand transportation patterns in Southern California's largest logistics hub.

![Project Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project analyzes traffic camera footage from Caltrans District 8 (Riverside and San Bernardino Counties) using deep learning to detect and classify vehicles, then correlates traffic patterns with demographic data from the U.S. Census Bureau to understand:

- **Why certain times have more traffic** - correlation with employment patterns and commute timing
- **Car vs truck ratios** - identifying logistics corridors vs residential routes  
- **How demographics predict traffic** - regression models using population, employment, and infrastructure
- **Temporal patterns** - peak hours, weekday/weekend differences, seasonal variation

### Key Features

- 🚗 **Vehicle Detection & Classification** using OpenSeeD or YOLOv8
- 📊 **Demographic Integration** with Census ACS and LODES employment data
- 🗺️ **Spatial Analysis** using geospatial correlation methods
- 📈 **Statistical Modeling** with regression and time series analysis
- 📷 **Automated Data Collection** from Caltrans CWWP2 camera feeds