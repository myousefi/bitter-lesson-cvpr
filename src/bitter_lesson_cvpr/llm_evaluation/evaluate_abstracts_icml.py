import os
import sqlite3
import random
import time
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm

from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from anthropic import RateLimitError

from bitter_lesson_cvpr.llm_evaluation.prompt_templates import (
    evaluate_bitter_lesson_alignment,
    BitterLessonScores,
)

DATABASE_PATH = "dbs/icml_papers.db" 


def create_scores_table_if_not_exists():
    """Creates the bitter_lesson_scores table if it doesn't exist."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bitter_lesson_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER,
                generality_of_approach_score INTEGER,
                reliance_on_human_knowledge_score INTEGER,
                scalability_with_computation_score INTEGER,
                leveraging_search_and_learning_score INTEGER,
                complexity_handling_score INTEGER,
                adaptability_and_generalization_score INTEGER,
                autonomy_and_discovery_score INTEGER,
                overall_bitter_lesson_alignment_score INTEGER,
                model TEXT DEFAULT 'gpt-3.5-turbo',
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
            """
        )


def get_random_papers() -> List[Tuple[int, str, str]]:
    """Fetches random papers from the database for a specific year."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, title, abstract 
            FROM papers
            WHERE id NOT IN (SELECT DISTINCT paper_id FROM bitter_lesson_scores WHERE model='claude-3-haiku-20240307')
            ORDER BY RANDOM()
            """,
        )
        return cursor.fetchall()


def evaluate_and_store_scores(papers: List[Tuple[int, str, str]]):
    """Evaluates papers using the prompt template and stores the scores."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        model = "claude-3-haiku-20240307"
        with AnthropicChatModel(model=model, api_key=os.getenv("MAGENTIC_ANTHROPIC_API_KEY")):
            for paper_id, title, abstract in tqdm(papers):
                while True:
                    try:
                        scores: BitterLessonScores = evaluate_bitter_lesson_alignment(
                            title=title, abstract=abstract
                        )
                        time.sleep(0.5)
                        break  # Exit the loop if successful
                    except RateLimitError as e:
                        print(f"Rate limit exceeded. Retrying in 1 second... Error: {e}")
                        time.sleep(1)  # Wait for 1 second before retrying

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
                        overall_bitter_lesson_alignment_score,
                        model
                    ) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        model
                    ),
                )
            
            conn.commit()


def main():
    """Main function to orchestrate the evaluation process."""
    create_scores_table_if_not_exists()  # Create the table if it doesn't exist

    random_papers = get_random_papers()
    evaluate_and_store_scores(random_papers)


if __name__ == "__main__":
    main()
