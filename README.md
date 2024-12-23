# bitter-lesson-cvpr

A data analysis project to evaluate computer vision papers through the lens of Richard Sutton's "Bitter Lesson" principles using LLMs.

## Project Overview

This project analyzes CVPR (Computer Vision and Pattern Recognition) papers to assess their alignment with the principles outlined in Richard Sutton's "The Bitter Lesson". It uses LLM-based evaluation to score papers across multiple dimensions related to scalable machine learning approaches versus human-engineered solutions.

## Key Features

- Paper data collection via Semantic Scholar API
- SQLite database storage for papers and evaluation scores
- Multiple LLM evaluation implementations (GPT-4, Claude)
- Scoring system across multiple bitter lesson dimensions:
    - Learning over engineering
    - Search over heuristics
    - Scalability with computation
    - Generality over specificity
    - Favoring fundamental principles

## Project Structure

bitter-lesson-cvpr/
├── src/
│   └── bitter_lesson_cvpr/
│       ├── llm_evaluation/ # LLM scoring implementations
│       ├── semantic_scholar_api/ # Paper data collection
│       └── utils/ # Helper utilities
├── notebooks/ # Analysis notebooks
├── dbs/ # SQLite database storage

## Setup

1. Clone the repository
2. Install dependencies:
    
    pip install -r requirements.txt
    
3. Set up environment variables:
    - `MAGENTIC_ANTHROPIC_API_KEY` - Anthropic API key
    - `S2_API_KEY` - Semantic Scholar API key
    - `OUTPUT_DIR` - Directory for output files

## Usage

Collect paper data:


python -m bitter_lesson_cvpr.semantic_scholar_api_call


Run LLM evaluations:


python -m bitter_lesson_cvpr.llm_evaluation.evaluate_abstracts_v2


Generate analysis plots:


python notebooks/plots.py


## Database Schema

The project uses two main tables:

- `papers`: Stores paper metadata and abstracts
- `bitter_lesson_scores_v2`: Stores LLM evaluation scores across different dimensions

## Analysis & Visualization

The `notebooks` directory contains Plotly-based visualizations for:

- Temporal trends in bitter lesson alignment
- Score distributions across different dimensions
- Comparative analysis between different LLM evaluators

## Citation


@inproceedings{yousefi-collins-2024-learning,
    title = "Learning the Bitter Lesson: Empirical Evidence from 20 Years of {CVPR} Proceedings",
    author = "Yousefi, Mojtaba  and       Collins, Jack",
    editor = "Peled-Cohen, Lotem  and       Calderon, Nitay  and       Lissak, Shir  and       Reichart, Roi",
    booktitle = "Proceedings of the 1st Workshop on NLP for Science (NLP4Science)",
    month = nov,
    year = "2024",
    address = "Miami, FL, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.nlp4science-1.15",
    doi = "10.18653/v1/2024.nlp4science-1.15",
    pages = "175--187",
    abstract = "This study examines the alignment of Conference on Computer Vision and Pattern Recognition (CVPR) research with the principles of the {``}bitter lesson{''} proposed by Rich Sutton. We analyze two decades of CVPR abstracts and titles using large language models (LLMs) to assess the field{'}s embracement of these principles. Our methodology leverages state-of-the-art natural language processing techniques to systematically evaluate the evolution of research approaches in computer vision. The results reveal significant trends in the adoption of general-purpose learning algorithms and the utilization of increased computational resources. We discuss the implications of these findings for the future direction of computer vision research and its potential impact on broader artificial intelligence development. This work contributes to the ongoing dialogue about the most effective strategies for advancing machine learning and computer vision, offering insights that may guide future research priorities and methodologies in the field.",
}

