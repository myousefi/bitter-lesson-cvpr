import os
import sqlite3
import random
import time

from typing import List, Tuple

from magentic import OpenaiChatModel
from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from openai import RateLimitError
import tqdm

from bitter_lesson_cvpr.llm_evaluation.prompt_v2 import (
    evaluate_bitter_lesson_alignment,
    BitterLessonScores,
)

DATABASE_PATH = "dbs/cvpr_papers.db"
SAMPLES_PER_YEAR = 200

def get_paper_count_by_year(model: str) -> List[Tuple[int, int]]:
    """Fetches the count of papers with scores for a specific model grouped by year."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.year, COUNT(*) as count
            FROM papers p
            INNER JOIN bitter_lesson_scores_v2 b ON p.id = b.paper_id
            WHERE b.model = ?
            GROUP BY p.year
            """,
            (model,),
        )
        return cursor.fetchall()

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


def get_papers_to_score(year: int, model: str, limit: int) -> List[Tuple[int, str, str]]:
    """Fetches papers from the database for a specific year that don't have a score for the given model."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.id, p.title, p.abstract
            FROM papers p
            WHERE p.year = ?
            AND p.abstract IS NOT NULL
            AND NOT EXISTS (
                SELECT 1
                FROM bitter_lesson_scores_v2 b
                WHERE b.paper_id = p.id
                AND b.model = ?
            )
            LIMIT ?
            """,
            (year, model, limit),
        )
        return cursor.fetchall()

def evaluate_and_store_scores(papers: List[Tuple[int, str, str]]):
    """Evaluates papers using the prompt template and stores the scores."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()

        model = "gpt-4o-mini-2024-07-18"
        with OpenaiChatModel(model, temperature=0):
        # model = "claude-3-5-sonnet-20240620"
        # with AnthropicChatModel(model=model, temperature=0, api_key=os.getenv("MAGENTIC_ANTHROPIC_API_KEY")):
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

    model = "gpt-4o"
    paper_counts = get_paper_count_by_year(model)

    for year, count in sorted(paper_counts):
        if count < SAMPLES_PER_YEAR and year > 2004:
            papers_to_score = get_papers_to_score(year, model, SAMPLES_PER_YEAR - count)
            evaluate_and_store_scores(tqdm.tqdm(papers_to_score, desc=f"Evaluating papers for {year}"))

if __name__ == "__main__":
    main()
