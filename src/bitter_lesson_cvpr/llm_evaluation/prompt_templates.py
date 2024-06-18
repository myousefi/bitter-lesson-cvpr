from magentic import prompt
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()


class BitterLessonScores(BaseModel):
    generality_of_the_approach_justification: str = Field(
        min_length=10,
        max_length=750,
        description="A brief 1-2 sentence justification for the generality of the approach score.",
    )
    generality_of_the_approach_score: int = Field(
        ge=0,
        le=10,
        description="How general is the approach? 0: Highly specialized and tailored to a specific domain or problem. 10: A completely general approach that can be applied to any domain or problem. (0-10)",
    )
    reliance_on_human_knowledge_justification: str = Field(
        min_length=10,
        max_length=750,
        description="A brief 1-2 sentence justification for the reliance on human knowledge score.",
    )
    reliance_on_human_knowledge_score: int = Field(
        ge=0,
        le=10,
        description="How reliant is the approach on human knowledge? 0: Heavily reliant on human knowledge, heuristics, and domain-specific insights. 10: No reliance on human knowledge or domain-specific heuristics. (0-10)",
    )
    scalability_with_computation_justification: str = Field(
        min_length=10,
        max_length=750,
        description="A brief 1-2 sentence justification for the scalability with computation score.",
    )
    scalability_with_computation_score: int = Field(
        ge=0,
        le=10,
        description="How well does the approach scale with computation? 0: Performance is limited by the available computation and does not scale well. 10: Performance scales seamlessly with increased computation power. (0-10)",
    )
    leveraging_of_search_and_learning_justification: str = Field(
        min_length=10,
        max_length=750,
        description="A brief 1-2 sentence justification for the leveraging of search and learning score.",
    )
    leveraging_of_search_and_learning_score: int = Field(
        ge=0,
        le=10,
        description="How much does the approach leverage search and learning? 0: Does not leverage search or learning techniques. 10: Heavily leverages search and learning as core methods. (0-10)",
    )
    complexity_handling_justification: str = Field(
        min_length=10,
        max_length=750,
        description="A brief 1-2 sentence justification for the complexity handling score.",
    )
    complexity_handling_score: int = Field(
        ge=0,
        le=10,
        description="How well does the approach handle complexity? 0: Struggles to handle complex, high-dimensional, or arbitrary problem spaces. 10: Excels at handling complex, high-dimensional, and arbitrary problem spaces. (0-10)",
    )
    adaptability_and_generalization_justification: str = Field(
        min_length=10,
        max_length=750,
        description="A brief 1-2 sentence justification for the adaptability and generalization score.",
    )
    adaptability_and_generalization_score: int = Field(
        ge=0,
        le=10,
        description="How adaptable and generalizable is the approach? 0: Rigid and inflexible, with poor generalization to new scenarios. 10: Highly adaptable and capable of generalizing to new, unseen scenarios. (0-10)",
    )
    autonomy_and_discovery_justification: str = Field(
        min_length=10,
        max_length=750,
        description="A brief 1-2 sentence justification for the autonomy and discovery score.",
    )
    autonomy_and_discovery_score: int = Field(
        ge=0,
        le=10,
        description="How autonomous and discovery-driven is the approach? 0: Relies on human-provided knowledge and discoveries. 10: Capable of autonomous discovery and learning without human input. (0-10)",
    )


@prompt(
    """
Title: {title}
Abstract: {abstract}

## The Bitter Lesson Assessment Framework

Rate each dimension on a scale of 0 to 10, where 0 represents complete reliance on human knowledge and domain-specific heuristics, and 10 represents a pure, general method that scales with increased computation. Provide a brief 1-2 sentence justification for each score.

**1. Generality of the Approach (0-10)**

0: Highly specialized and tailored to a specific domain or problem.

10: A completely general approach that can be applied to any domain or problem.

**2. Reliance on Human Knowledge (0-10)**

0: Heavily reliant on human knowledge, heuristics, and domain-specific insights.

10: No reliance on human knowledge or domain-specific heuristics.

**3. Scalability with Computation (0-10)**

0: Performance is limited by the available computation and does not scale well.

10: Performance scales seamlessly with increased computation power.

**4. Leveraging of Search and Learning (0-10)**

0: Does not leverage search or learning techniques.

10: Heavily leverages search and learning as core methods.

**5. Complexity Handling (0-10)**

0: Struggles to handle complex, high-dimensional, or arbitrary problem spaces.

10: Excels at handling complex, high-dimensional, and arbitrary problem spaces.

**6. Adaptability and Generalization (0-10)**

0: Rigid and inflexible, with poor generalization to new scenarios.

10: Highly adaptable and capable of generalizing to new, unseen scenarios.

**7. Autonomy and Discovery (0-10)**

0: Relies on human-provided knowledge and discoveries.

10: Capable of autonomous discovery and learning without human input.
"""
)
def evaluate_bitter_lesson_alignment(
    title: str, abstract: str
) -> BitterLessonScores: ...


if __name__ == "__main__":
    title = "Gaussian Shell Maps for Efficient 3D Human Generation"

    abstract = """
    Efficient generation of 3D digital humans is important in several industries including virtual reality social media and cinematic production. 3D generative adversarial networks (GANs) have demonstrated state-of-the-art (SOTA) quality and diversity for generated assets. Current 3D GAN architectures however typically rely on volume representations which are slow to render thereby hampering the GAN training and requiring multi-view-inconsistent 2D upsamplers. Here we introduce Gaussian Shell Maps (GSMs) as a framework that connects SOTA generator network architectures with emerging 3D Gaussian rendering primitives using an articulable multi shell--based scaffold. In this setting a CNN generates a 3D texture stack with features that are mapped to the shells. The latter represent inflated and deflated versions of a template surface of a digital human in a canonical body pose. Instead of rasterizing the shells directly we sample 3D Gaussians on the shells whose attributes are encoded in the texture features. These Gaussians are efficiently and differentiably rendered. The ability to articulate the shells is important during GAN training and at inference time to deform a body into arbitrary user-defined poses. Our efficient rendering scheme bypasses the need for view-inconsistent upsamplers and achieves high-quality multi-view consistent renderings at a native resolution of 512 x512 pixels. We demonstrate that GSMs successfully generate 3D humans when trained on single-view datasets including SHHQ and DeepFashion.
    """

    scores = evaluate_bitter_lesson_alignment(title=title, abstract=abstract)
    print(f"Title: {title}")
    print(f"Abstract: {abstract}")
    print("\nBitter Lesson Scores:")
    for field, value in scores:
        print(f"{field}: {value}")
