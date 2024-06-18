from magentic import prompt
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()


class BitterLessonScores(BaseModel):
    general_methods_usage_justification: str = Field(
        min_length=10,
        max_length=500,
        description="A brief 1-2 sentence justification for the general methods usage score.",
    )
    general_methods_usage_score: int = Field(
        ge=0,
        le=10,
        description="How extensively does the abstract rely on general, scalable methods versus specialized, domain-specific techniques? (0-10)",
    )
    computation_data_scaling_justification: str = Field(
        min_length=10,
        max_length=500,
        description="A brief 1-2 sentence justification for the computation and data scaling score.",
    )
    computation_data_scaling_score: int = Field(
        ge=0,
        le=10,
        description="To what extent does the methodology emphasize leveraging large-scale data or computational power? (0-10)",
    )
    feature_learning_vs_engineering_justification: str = Field(
        min_length=10,
        max_length=500,
        description="A brief 1-2 sentence justification for the feature learning vs engineering score.",
    )
    feature_learning_vs_engineering_score: int = Field(
        ge=0,
        le=10,
        description="Does the abstract highlight systems that learn features and behaviors autonomously rather than through hand-engineered processes? (0-10)",
    )
    robustness_adaptability_justification: str = Field(
        min_length=10,
        max_length=500,
        description="A brief 1-2 sentence justification for the robustness and adaptability score.",
    )
    robustness_adaptability_score: int = Field(
        ge=0,
        le=10,
        description="How well does the approach adapt to new or varying data sets without requiring re-engineering? (0-10)",
    )


@prompt(
    """
Title: {title}
Abstract: {abstract}

Evaluate the alignment of the abstract to the "bitter lesson" based on the following dimensions:

General Methods Usage: How extensively does the abstract rely on general, scalable methods versus specialized, domain-specific techniques? Provide a brief 1-2 sentence justification, then a score from 0 (does not demonstrate this principle at all) to 10 (fully embodies this principle). 5 would indicate the abstract somewhat adheres to the principle.

Computation and Data Scaling: To what extent does the methodology emphasize leveraging large-scale data or computational power? Provide a brief 1-2 sentence justification, then a score from 0 to 10.

Feature Learning versus Engineering: Does the abstract highlight systems that learn features and behaviors autonomously rather than through hand-engineered processes? Provide a brief 1-2 sentence justification, then a score from 0 to 10.

Robustness and Adaptability: How well does the approach adapt to new or varying data sets without requiring re-engineering? Provide a brief 1-2 sentence justification, then a score from 0 to 10.
"""
)
def evaluate_bitter_lesson_alignment(
    title: str, abstract: str
) -> BitterLessonScores: ...


if __name__ == "__main__":
    title = "The Neglected Tails in Vision-Language Models"
    abstract = """
    Vision-language models (VLMs) excel in zero-shot recognition but their performance varies greatly across different visual concepts. For example although CLIP achieves impressive accuracy on ImageNet (60-80%) its performance drops below 10% for more than ten concepts like night snake presumably due to their limited presence in the pretraining data. However measuring the frequency of concepts in VLMs' large-scale datasets is challenging. We address this by using large language models (LLMs) to count the number of pretraining texts that contain synonyms of these concepts. Our analysis confirms that popular datasets such as LAION exhibit a long-tailed concept distribution yielding biased performance in VLMs. We also find that downstream applications of VLMs including visual chatbots (e.g. GPT-4V) and text-to-image models (e.g. Stable Diffusion) often fail to recognize or generate images of rare concepts identified by our method. To mitigate the imbalanced performance of zero-shot VLMs we propose REtrieval-Augmented Learning (REAL). First instead of prompting VLMs using the original class names REAL uses their most frequent synonyms found in pretraining texts. This simple change already outperforms costly human-engineered and LLM-enriched prompts over nine benchmark datasets. Second REAL trains a linear classifier on a small yet balanced set of pretraining data retrieved using concept synonyms. REAL surpasses the previous zero-shot SOTA using 400x less storage and 10000x less training time!
    """

    scores = evaluate_bitter_lesson_alignment(title=title, abstract=abstract)
    print(f"Title: {title}")
    print(f"Abstract: {abstract}")
    print("\nBitter Lesson Scores:")
    for field, value in scores:
        print(f"{field}: {value}")
