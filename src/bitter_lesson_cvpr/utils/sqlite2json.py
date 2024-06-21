import sqlite3
import json

from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_PATH = "dbs/cvpr_papers.db"
OUTPUT_FILE = "cvpr_papers_with_scores.json"
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def extract_data():
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
              p.year,
              p.title,
              p.abstract,
              (
                SELECT
                  json_object(
                    'learning_over_engineering_score',
                    v2.learning_over_engineering_score,
                    'search_over_heuristics_score',
                    v2.search_over_heuristics_score,
                    'scalability_with_computation_score',
                    v2.scalability_with_computation_score,
                    'generality_over_specificity_score',
                    v2.generality_over_specificity_score,
                    'favoring_fundamental_principles_score',
                    v2.favoring_fundamental_principles_score
                  )
                FROM bitter_lesson_scores_v2 AS v2
                WHERE
                  v2.paper_id = p.id AND v2.model = 'gpt-4o'
              ) AS gpt_scores,
              (
                SELECT
                  json_object(
                    'learning_over_engineering_score',
                    v2.learning_over_engineering_score,
                    'search_over_heuristics_score',
                    v2.search_over_heuristics_score,
                    'scalability_with_computation_score',
                    v2.scalability_with_computation_score,
                    'generality_over_specificity_score',
                    v2.generality_over_specificity_score,
                    'favoring_fundamental_principles_score',
                    v2.favoring_fundamental_principles_score
                  )
                FROM bitter_lesson_scores_v2 AS v2
                WHERE
                  v2.paper_id = p.id AND v2.model = 'claude-3-5-sonnet-20240620'
              ) AS claude_scores
            FROM papers AS p
            WHERE
              p.id IN (
                SELECT DISTINCT
                  paper_id
                FROM bitter_lesson_scores_v2
              );
        """)

        papers = []
        for row in cursor.fetchall():
            year, title, abstract, gpt_scores, claude_scores = row
            papers.append({
                'year': year,
                'title': title,
                'abstract': abstract,
                'gpt-4o': json.loads(gpt_scores) if gpt_scores else None,
                'claude-3-5-sonnet-20240620': json.loads(claude_scores) if claude_scores else None
            })
        return papers

def write_to_json(papers):
    with open(OUTPUT_DIR+OUTPUT_FILE, 'w') as f:
        json.dump(papers, f, indent=2)

if __name__ == "__main__":
    papers = extract_data()
    write_to_json(papers)
    print(f"Data extracted and saved to {OUTPUT_DIR+OUTPUT_FILE}")
