import sqlite3
import random
from typing import List, Tuple

from bitter_lesson_cvpr.llm_evaluation.prompt_templates import (
    evaluate_bitter_lesson_alignment,
    BitterLessonScores,
)

DATABASE_PATH = "cvpr_papers.db"  # Replace with the actual path to your database
SAMPLES_PER_YEAR = 90


def create_scores_table_if_not_exists():
    """Creates the bitter_lesson_scores table if it doesn't exist."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bitter_lesson_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER UNIQUE,
                generality_of_approach_score INTEGER,
                reliance_on_human_knowledge_score INTEGER,
                scalability_with_computation_score INTEGER,
                leveraging_search_and_learning_score INTEGER,
                complexity_handling_score INTEGER,
                adaptability_and_generalization_score INTEGER,
                autonomy_and_discovery_score INTEGER,
                overall_bitter_lesson_alignment_score INTEGER,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
            """
        )


def get_random_papers(year: int, limit: int) -> List[Tuple[int, str, str]]:
    """Fetches random papers from the database for a specific year."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, title, abstract 
            FROM papers 
            WHERE year = ?
            AND id NOT IN (SELECT paper_id FROM bitter_lesson_scores)
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (year, limit),
        )
        return cursor.fetchall()


def evaluate_and_store_scores(papers: List[Tuple[int, str, str]]):
    """Evaluates papers using the prompt template and stores the scores."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        for paper_id, title, abstract in papers:
            scores: BitterLessonScores = evaluate_bitter_lesson_alignment(
                title=title, abstract=abstract
            )
            cursor.execute(
                """
                INSERT INTO bitter_lesson_scores (
                    paper_id, 
                    generality_of_approach_score, 
                    reliance_on_human_knowledge_score,
                    scalability_with_computation_score,
                    leveraging_search_and_learning_score,
                    complexity_handling_score,
                    adaptability_and_generalization_score,
                    autonomy_and_discovery_score,
                    overall_bitter_lesson_alignment_score
                ) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    scores.generality_of_approach_score,
                    scores.reliance_on_human_knowledge_score,
                    scores.scalability_with_computation_score,
                    scores.leveraging_search_and_learning_score,
                    scores.complexity_handling_score,
                    scores.adaptability_and_generalization_score,
                    scores.autonomy_and_discovery_score,
                    scores.overall_bitter_lesson_alignment_score,
                ),
            )
        conn.commit()


def main():
    """Main function to orchestrate the evaluation process."""
    create_scores_table_if_not_exists()  # Create the table if it doesn't exist

    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT year FROM papers")
        years = [row[0] for row in cursor.fetchall()]

    for year in years:
        print(f"Processing year {year}...")
        random_papers = get_random_papers(year, SAMPLES_PER_YEAR)
        evaluate_and_store_scores(random_papers)


if __name__ == "__main__":
    main()
