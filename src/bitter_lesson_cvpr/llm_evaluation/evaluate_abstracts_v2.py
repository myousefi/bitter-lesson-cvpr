import os
import sqlite3
import random
import time

from typing import List, Tuple

from magentic import OpenaiChatModel
from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from openai import RateLimitError

from bitter_lesson_cvpr.llm_evaluation.prompt_v2 import (
    evaluate_bitter_lesson_alignment,
    BitterLessonScores,
)

DATABASE_PATH = "dbs/cvpr_papers.db"
SAMPLES_PER_YEAR = 200 - 22


def create_scores_table_if_not_exists():
    """Creates the bitter_lesson_scores_v2 table if it doesn't exist."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bitter_lesson_scores_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                model TEXT,
                learning_over_engineering_score INTEGER,
                search_over_heuristics_score INTEGER,
                scalability_with_computation_score INTEGER,
                generality_over_specificity_score INTEGER,
                favoring_fundamental_principles_score INTEGER,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
            """
        )


def get_scored_papers(year: int) -> List[Tuple[int, str, str]]:
    """Fetches all papers from the database for a specific year that have a gpt-4o score."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.id, p.title, p.abstract 
            FROM papers p
            INNER JOIN bitter_lesson_scores_v2 b ON p.id = b.paper_id
            WHERE p.year = ?
            AND p.abstract IS NOT NULL
            AND b.model = 'gpt-4o'
            AND NOT EXISTS (
                SELECT 1
                FROM bitter_lesson_scores_v2 b2
                WHERE b2.paper_id = p.id
                AND b2.model = 'claude-3-5-sonnet-20240620'
            )
            """,
            (year,),
        )
        return cursor.fetchall()


def evaluate_and_store_scores(papers: List[Tuple[int, str, str]]):
    """Evaluates papers using the prompt template and stores the scores."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()

        # model = "gpt-4o-mini-2024-07-18"
        # with OpenaiChatModel(model, temperature=0):
        model = "claude-3-5-sonnet-20240620"
        with AnthropicChatModel(model=model, temperature=0, api_key=os.getenv("MAGENTIC_ANTHROPIC_API_KEY")):
            for paper_id, title, abstract in papers:
                while True:
                    try:
                        scores: BitterLessonScores = evaluate_bitter_lesson_alignment(
                            title=title, 
                            abstract=abstract)
                        break
                    except RateLimitError as e:
                        print(f"Rate limit error: {e}")
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Error: {e}")
                        break

                cursor.execute(
                    """
                    INSERT INTO bitter_lesson_scores_v2 (
                        paper_id,
                        model,
                        learning_over_engineering_score, 
                        search_over_heuristics_score,
                        scalability_with_computation_score,
                        generality_over_specificity_score,
                        favoring_fundamental_principles_score
                    ) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id,
                        model,
                        scores.learning_over_engineering_score,
                        scores.search_over_heuristics_score,
                        scores.scalability_with_computation_score,
                        scores.generality_over_specificity_score,
                        scores.favoring_fundamental_principles_score,
                    ),
                )
            conn.commit()


def main():
    """Main function to orchestrate the evaluation process."""
    create_scores_table_if_not_exists()  # Create the table if it doesn't exist

    # with sqlite3.connect(DATABASE_PATH) as conn:
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT DISTINCT year FROM papers")
    #     years = [row[0] for row in cursor.fetchall()]

    for year in range(2004, 2025):
        print(f"Processing year {year}...")
        random_papers = get_scored_papers(year)
        evaluate_and_store_scores(random_papers)


if __name__ == "__main__":
    main()
