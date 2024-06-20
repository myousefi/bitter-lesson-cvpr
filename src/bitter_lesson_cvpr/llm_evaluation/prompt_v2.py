
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
    title = "Softassign versus Softmax: Benchmarks in Combinatorial Optimization"
    abstract = """
    A new technique, termed soft assign, is applied for the first time to two classic combinatorial optimization problems, the travel(cid:173) ing salesman problem and graph partitioning. Soft assign , which has emerged from the recurrent neural network/statistical physics framework, enforces two-way (assignment) constraints without the use of penalty terms in the energy functions. The soft assign can also be generalized from two-way winner-take-all constraints to multiple membership constraints which are required for graph par(cid:173) titioning. The soft assign technique is compared to the softmax (Potts glass). Within the statistical physics framework, softmax and a penalty term has been a widely used method for enforcing the two-way constraints common within many combinatorial optimiza(cid:173) tion problems. The benchmarks present evidence that soft assign has clear advantages in accuracy, speed, parallelizabilityand algo(cid:173) rithmic simplicity over softmax and a penalty term in optimization problems with two-way constraints.
    """

    from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
    from magentic.chat_model.openai_chat_model import OpenaiChatModel

    import os 

    # model = "claude-3-opus-20240229"
    # with AnthropicChatModel(model=model, api_key=os.getenv("MAGENTIC_ANTHROPIC_API_KEY")):
    # with OpenaiChatModel("gpt-4o", temperature=0):
        scores = evaluate_bitter_lesson_alignment(title=title, abstract=abstract)

    print(f"Title: {title}")
    print(f"Abstract: {abstract}")
    print("\nBitter Lesson Scores:")
    for field, value in scores:
        print(f"{field}: {value}")
