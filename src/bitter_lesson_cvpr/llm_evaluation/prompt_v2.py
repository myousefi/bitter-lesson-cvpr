
from magentic import prompt
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field

class BitterLessonScores(BaseModel):
    learning_over_engineering_score: int = Field(
        ge=0,
        le=10,
        description="Learning over Engineering: To what extent does the idea prioritize leveraging computation through data-driven learning and statistical methods (e.g., machine learning, deep learning, neural networks, probabilistic models, unsupervised learning, supervised learning, reinforcement learning, generative models, discriminative models, ensemble methods, online learning, active learning, semi-supervised learning) over relying on human-engineered knowledge, heuristics, and domain expertise (e.g., hand-crafted features, rule-based systems, expert systems, symbolic AI, knowledge representation, logic programming, constraint satisfaction)?\n\nPlease rate on a scale from 0 to 10, where:\n0 = Completely relies on human engineering, 5 = Equal emphasis on learning and engineering, 10 = Completely prioritizes learning from data"
    )
    search_over_heuristics_score: int = Field(
        ge=0,
        le=10,
        description="Search over Heuristics: To what degree does the idea emphasize leveraging computation through search algorithms (e.g., gradient descent, stochastic gradient descent, evolutionary algorithms, genetic algorithms, simulated annealing, Monte Carlo methods, Markov chain Monte Carlo, beam search, branch and bound, A* search, heuristic search) and optimization techniques (e.g., convex optimization, stochastic optimization, combinatorial optimization, integer programming, quadratic programming, linear programming, non-linear optimization, multi-objective optimization), rather than depending on human-designed heuristics and problem-specific strategies (e.g., hand-tuned parameters, domain-specific rules, expert knowledge, case-based reasoning, heuristic functions)?\n\nPlease rate on a scale from 0 to 10, where:\n0 = Completely relies on human-designed heuristics, 5 = Equal emphasis on search and heuristics, 10 = Completely prioritizes search and optimization"
    )
    scalability_with_computation_score: int = Field(
        ge=0,
        le=10,
        description="Scalability with Computation: To what extent is the idea based on methods that can continuously scale and improve performance as the available computational resources (e.g., processing power, memory, storage, data, distributed computing, cloud computing, GPU acceleration, TPU acceleration, high-performance computing, edge computing, quantum computing) increase, taking full advantage of the exponential growth in computing capabilities (e.g., Moore's Law, Dennard scaling, Amdahl's Law, Gustafson's Law)?\n\nPlease rate on a scale from 0 to 10, where:\n0 = Does not scale with computation at all, 5 = Scales moderately with computation, 10 = Scales exceptionally well with computation"
    )
    generality_over_specificity_score: int = Field(
        ge=0,
        le=10,
        description="Generality over Specificity: To what degree does the approach emphasize general, flexible, and adaptable methods that can learn and capture arbitrary complexity from data (e.g., deep learning, transfer learning, meta-learning, representation learning, multi-task learning, few-shot learning, zero-shot learning, self-supervised learning, unsupervised pre-training, domain adaptation, continual learning, lifelong learning, incremental learning) rather than attempting to build in complex and detailed models of the world through manual engineering and domain-specific knowledge (e.g., hand-designed features, domain-specific ontologies, knowledge graphs, expert systems, rule-based systems, symbolic representations, logic-based representations)?\n\nPlease rate on a scale from 0 to 10, where:\n0 = Completely domain-specific and manually engineered, 5 = Balance of generality and specificity, 10 = Maximally general, flexible and adaptable"
    )
    favoring_fundamental_principles_score: int = Field(
        ge=0,
        le=10,
        description="Favoring Fundamental Principles: To what extent does the approach adhere to fundamental principles of computation, mathematics, and information theory (e.g., algorithmic efficiency, computational complexity, statistical learning theory, information entropy, Bayesian inference, Kolmogorov complexity, Occam's razor, Minimum Description Length, PAC learning, VC dimension, Rademacher complexity, concentration inequalities, regularization, sparsity, smoothness, stability, convergence, consistency) rather than focusing on emulating the specific details of human cognition or biological intelligence (e.g., neuroscience-inspired architectures, cognitive architectures, embodied cognition, situated cognition, enactivism, dynamical systems theory, ecological psychology)?\n\nPlease rate on a scale from 0 to 10, where:\n0 = Completely focused on emulating human/biological details, 5 = Equal focus on principles and human/biological details, 10 = Completely grounded in fundamental principles"
    )

@prompt(
    """
Title: {title}
Abstract: {abstract}

Evaluate the alignment of the abstract with the following principles, assigning a score from 0 to 10 for each. 

**0 indicates the principle is completely absent, 5 indicates a moderate presence, and 10 indicates a strong and clear embodiment of the principle.**

* **Learning over Engineering:** To what extent does the idea prioritize leveraging computation through data-driven learning and statistical methods over relying on human-engineered knowledge, heuristics, and domain expertise? 

* **Search over Heuristics:** To what degree does the idea emphasize leveraging computation through search algorithms and optimization techniques, rather than depending on human-designed heuristics and problem-specific strategies?

* **Scalability with Computation:** To what extent is the idea based on methods that can continuously scale and improve performance as the available computational resources increase, taking full advantage of the exponential growth in computing capabilities?

* **Generality over Specificity:** To what degree does the approach emphasize general, flexible, and adaptable methods that can learn and capture arbitrary complexity from data rather than attempting to build in complex and detailed models of the world through manual engineering and domain-specific knowledge?

* **Favoring Fundamental Principles:** To what extent does the approach adhere to fundamental principles of computation, mathematics, and information theory rather than focusing on emulating the specific details of human cognition or biological intelligence?
"""
)
def evaluate_bitter_lesson_alignment(
    title: str, abstract: str
) -> BitterLessonScores: ...


if __name__ == "__main__":
    title = "Going Deeper with Convolutions"
    abstract = """
    We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.
    """

    scores = evaluate_bitter_lesson_alignment(title=title, abstract=abstract)
    print(f"Title: {title}")
    print(f"Abstract: {abstract}")
    print("\nBitter Lesson Scores:")
    for field, value in scores:
        print(f"{field}: {value}")
