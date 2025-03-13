# Difficult Conversations Analysis Dashboard

## Overview

This project analyzes a dataset of AI-powered difficult conversations from ConsciousInsights' leadership development platform. The dashboard provides insights into user interactions, feedback patterns, and conversation characteristics to help learning managers understand how their teams engage with the platform.

## Features

- **Quantitative Analysis**: Visualization of user feedback scores, conversation length metrics, and correlation analysis
- **Topic Classification**: LLM-powered categorization of conversations by main topics (Performance Related, Career Development, etc.)
- **Communication Style Analysis**: Detection of user communication styles and emotional tones
- **Interactive Visualizations**: Plotly-powered charts for exploring conversation data

## Technology Stack

- **Streamlit**: Web application framework for interactive dashboard
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualization
- **LangChain**: Framework for LLM integration and structured prompting
- **Large Language Models**: Used for topic classification and communication style analysis

## Installation

```shell
xcode-select --install
poetry install
```

## Run streamlit app

```shell
make run
```
