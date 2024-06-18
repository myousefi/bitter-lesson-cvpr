from magentic import prompt
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field

class BitterLessonScores(BaseModel):
    generality_of_approach_score: int = Field(
        ge=0,
        le=10,
        description="Rate the generality of the approach described in the abstract. Is it a highly specialized and tailored solution for a specific domain or problem (score closer to 0), or is it a completely general approach that can be applied to any domain or problem (score closer to 10)? Consider the breadth of applicability and the level of domain-specificity.",
    )
    reliance_on_human_knowledge_score: int = Field(
        ge=0,
        le=10,
        description="Evaluate the extent to which the approach relies on human knowledge, heuristics, and domain-specific insights (score closer to 0), or if it operates without any reliance on human knowledge or domain-specific heuristics (score closer to 10). Consider the level of human involvement and the use of hand-crafted rules or heuristics.",
    )
    scalability_with_computation_score: int = Field(
        ge=0,
        le=10,
        description="Assess how well the performance of the approach scales with increased computation power. If the performance is limited by the available computation and does not scale well (score closer to 0), or if the performance scales seamlessly with increased computation power (score closer to 10). Consider the computational complexity and the ability to leverage more resources.",
    )
    leveraging_search_and_learning_score: int = Field(
        ge=0,
        le=10,
        description="Evaluate the extent to which the approach leverages search and learning techniques as core methods. If it does not leverage search or learning at all (score closer to 0), or if it heavily leverages search and learning (score closer to 10). Consider the use of optimization, reinforcement learning, or other search and learning algorithms.",
    )
    complexity_handling_score: int = Field(
        ge=0,
        le=10,
        description="Rate the ability of the approach to handle complex, high-dimensional, or arbitrary problem spaces. If it struggles with complexity (score closer to 0), or if it excels at handling complex, high-dimensional, and arbitrary problem spaces (score closer to 10). Consider the dimensionality, non-linearity, and complexity of the problem domain.",
    )
    adaptability_and_generalization_score: int = Field(
        ge=0,
        le=10,
        description="Assess the adaptability and generalization capabilities of the approach. If it is rigid and inflexible, with poor generalization to new scenarios (score closer to 0), or if it is highly adaptable and capable of generalizing to new, unseen scenarios (score closer to 10). Consider the ability to transfer knowledge and adapt to new data or environments.",
    )
    autonomy_and_discovery_score: int = Field(
        ge=0,
        le=10,
        description="Evaluate the level of autonomy and discovery exhibited by the approach. If it relies heavily on human-provided knowledge and discoveries (score closer to 0), or if it is capable of autonomous discovery and learning without human input (score closer to 10). Consider the level of human involvement in the learning process and the ability to discover new knowledge independently.",
    )
    overall_bitter_lesson_alignment_score: int = Field(
        ge=0,
        le=10,
        description="Provide an overall assessment of how well the abstract aligns with the principles of the Bitter Lesson, considering all factors. A score closer to 0 indicates poor alignment, while a score closer to 10 indicates strong alignment with the Bitter Lesson's emphasis on general methods that leverage computation over human knowledge and domain-specific heuristics.",
    )

@prompt(
    """
Title: {title}
Abstract: {abstract}

The "bitter lesson" by Rich Sutton states that general methods that leverage computation are ultimately the most effective, and that leveraging computation and learning should be prioritized over human knowledge in AI systems.

Evaluate the alignment of the abstract to the "bitter lesson" based on the following dimensions, assigning a score from 0 (does not demonstrate this principle at all) to 10 (fully embodies this principle). 5 would indicate the abstract somewhat adheres to the principle. 

Generality of Approach: Rate the generality of the approach described in the abstract. Is it a highly specialized and tailored solution for a specific domain or problem (score closer to 0), or is it a completely general approach that can be applied to any domain or problem (score closer to 10)? Consider the breadth of applicability and the level of domain-specificity. :

Reliance on Human Knowledge: Evaluate the extent to which the approach relies on human knowledge, heuristics, and domain-specific insights (score closer to 0), or if it operates without any reliance on human knowledge or domain-specific heuristics (score closer to 10). Consider the level of human involvement and the use of hand-crafted rules or heuristics. :

Scalability with Computation: Assess how well the performance of the approach scales with increased computation power. If the performance is limited by the available computation and does not scale well (score closer to 0), or if the performance scales seamlessly with increased computation power (score closer to 10). Consider the computational complexity and the ability to leverage more resources. :

Leveraging Search and Learning: Evaluate the extent to which the approach leverages search and learning techniques as core methods. If it does not leverage search or learning at all (score closer to 0), or if it heavily leverages search and learning (score closer to 10). Consider the use of optimization, reinforcement learning, or other search and learning algorithms. :

Complexity Handling: Rate the ability of the approach to handle complex, high-dimensional, or arbitrary problem spaces. If it struggles with complexity (score closer to 0), or if it excels at handling complex, high-dimensional, and arbitrary problem spaces (score closer to 10). Consider the dimensionality, non-linearity, and complexity of the problem domain. :

Adaptability and Generalization: Assess the adaptability and generalization capabilities of the approach. If it is rigid and inflexible, with poor generalization to new scenarios (score closer to 0), or if it is highly adaptable and capable of generalizing to new, unseen scenarios (score closer to 10). Consider the ability to transfer knowledge and adapt to new data or environments. :

Autonomy and Discovery: Evaluate the level of autonomy and discovery exhibited by the approach. If it relies heavily on human-provided knowledge and discoveries (score closer to 0), or if it is capable of autonomous discovery and learning without human input (score closer to 10). Consider the level of human involvement in the learning process and the ability to discover new knowledge independently. :

Overall Bitter Lesson Alignment: Provide an overall assessment of how well the abstract aligns with the principles of the Bitter Lesson, considering all factors. A score closer to 0 indicates poor alignment, while a score closer to 10 indicates strong alignment with the Bitter Lesson's emphasis on general methods that leverage computation over human knowledge and domain-specific heuristics. :
"""
)
def evaluate_bitter_lesson_alignment(
    title: str, abstract: str
) -> BitterLessonScores: ...


if __name__ == "__main__":
    title = "Articulated Pose Estimation Using Discriminative Armlet Classifiers"
    abstract = """
    We propose a novel approach for human pose estimation in real-world cluttered scenes, and focus on the challenging problem of predicting the pose of both arms for each person in the image. For this purpose, we build on the notion of poselets [4] and train highly discriminative classifiers to differentiate among arm configurations, which we call armlets. We propose a rich representation which, in addition to standard HOG features, integrates the information of strong contours, skin color and contextual cues in a principled manner. Unlike existing methods, we evaluate our approach on a large subset of images from the PASCAL VOC detection dataset, where critical visual phenomena, such as occlusion, truncation, multiple instances and clutter are the norm. Our approach outperforms Yang and Ramanan [26], the state-of-the-art technique, with an improvement from 29.0% to 37.5% PCP accuracy on the arm keypoint prediction task, on this new pose estimation dataset.
    """

    scores = evaluate_bitter_lesson_alignment(title=title, abstract=abstract)
    print(f"Title: {title}")
    print(f"Abstract: {abstract}")
    print("\nBitter Lesson Scores:")
    for field, value in scores:
        print(f"{field}: {value}")
