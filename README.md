
bitter-lesson-cvpr
==================

A data analysis project to evaluate computer vision papers through the lens of Richard Sutton's _Bitter Lesson_ principles using large language models (LLMs).

Project Overview
----------------

This project analyzes CVPR (Computer Vision and Pattern Recognition) papers to assess their alignment with the principles outlined in Richard Sutton’s _The Bitter Lesson_. It leverages LLM-based evaluations to score papers across multiple dimensions that compare scalable machine learning approaches to human-engineered solutions.

Key Features
------------

*   **Paper data collection** via the Semantic Scholar API
*   **SQLite database** for storing papers and evaluation scores
*   **Multiple LLM evaluation** methods (e.g., GPT-4, Claude)
*   **Scoring system** across key _Bitter Lesson_ dimensions:
    *   Learning over engineering
    *   Search over heuristics
    *   Scalability with computation
    *   Generality over specificity
    *   Favoring fundamental principles

Project Structure
-----------------

```bash
bitter-lesson-cvpr/
├── src/
│   └── bitter_lesson_cvpr/
│       ├── llm_evaluation/        # LLM scoring implementations
│       ├── semantic_scholar_api/  # Paper data collection
│       └── utils/                 # Helper utilities
├── notebooks/                     # Analysis notebooks
├── dbs/                           # SQLite database storage
```

Setup
-----

1.  **Clone the repository**:
    
    ```bash
    git clone https://github.com/yourusername/bitter-lesson-cvpr.git
    ```
2.  **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up environment variables**:
    
    *   `MAGENTIC_ANTHROPIC_API_KEY` — Anthropic API key
    *   `S2_API_KEY` — Semantic Scholar API key
    *   `OUTPUT_DIR` — Directory for output files

Usage
-----

1.  **Collect paper data**:
    
    ```bash
    python -m bitter_lesson_cvpr.semantic_scholar_api_call
    ```
2.  **Run LLM evaluations**:
    
    ```bash
    python -m bitter_lesson_cvpr.llm_evaluation.evaluate_abstracts_v2
    ```
3.  **Generate analysis plots**:
    
    ```bash
    python notebooks/plots.py
    ```

Database Schema
---------------

The project uses two main tables:

*   **papers**: Stores paper metadata and abstracts
*   **bitter\_lesson\_scores\_v2**: Stores LLM evaluation scores across different dimensions

Analysis & Visualization
------------------------

The `notebooks` directory contains Plotly-based visualization notebooks for:

*   Temporal trends in _Bitter Lesson_ alignment
*   Score distributions across different dimensions
*   Comparative analysis between various LLM evaluators

Citation
--------

```sql
@inproceedings{yousefi-collins-2024-learning,
  title     = "Learning the Bitter Lesson: Empirical Evidence from 20 Years of {CVPR} Proceedings",
  author    = "Yousefi, Mojtaba and Collins, Jack",
  editor    = "Peled-Cohen, Lotem and Calderon, Nitay and Lissak, Shir and Reichart, Roi",
  booktitle = "Proceedings of the 1st Workshop on NLP for Science (NLP4Science)",
  month     = nov,
  year      = "2024",
  address   = "Miami, FL, USA",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2024.nlp4science-1.15",
  doi       = "10.18653/v1/2024.nlp4science-1.15",
  pages     = "175--187",
  abstract  = "This study examines the alignment of Conference on Computer Vision and Pattern Recognition (CVPR) research with the principles of the {``}bitter lesson{''} proposed by Rich Sutton. We analyze two decades of CVPR abstracts and titles using large language models (LLMs) to assess the field's embracement of these principles. Our methodology leverages state-of-the-art natural language processing techniques to systematically evaluate the evolution of research approaches in computer vision. The results reveal significant trends in the adoption of general-purpose learning algorithms and the utilization of increased computational resources. We discuss the implications of these findings for the future direction of computer vision research and its potential impact on broader artificial intelligence development. This work contributes to the ongoing dialogue about the most effective strategies for advancing machine learning and computer vision, offering insights that may guide future research priorities and methodologies in the field.",
}
```
