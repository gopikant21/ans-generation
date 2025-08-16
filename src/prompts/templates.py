REFERENCE_ANSWERS_PROMPT = """
You are an expert in {domain} with deep knowledge and experience. Your task is to generate multiple valid reference answers for evaluation purposes.

Question: {question}
Domain: {domain}
Difficulty Level: {difficulty}

Generate exactly {num_references} different correct reference answers, each representing a different valid approach or explanation style. Each answer must be factually correct, even if phrased differently.

Required approaches:
1. concise_definition: Brief, accurate definition or answer (2-3 sentences)
2. detailed_explanation: Comprehensive explanation with context and reasoning (4-6 sentences)
3. step_by_step_solution: Procedural or methodical approach (numbered steps or clear sequence)
4. analogy_or_example: Example-driven or analogy-based explanation (with concrete examples)
{additional_approach}

Guidelines for Natural Professional Communication:
- Write as a knowledgeable professional would naturally explain concepts
- Use conversational yet authoritative tone
- Include natural transitions and connective phrases
- Vary sentence structure and length organically
- Avoid overly academic or textbook language
- Use real-world context where appropriate
- Make explanations flow naturally as if explaining to a colleague
- Include clarifying phrases humans naturally use ("In essence," "Basically," "The key point is")

Additional Guidelines:
- All answers must be factually correct and complete
- Each should represent a genuinely different perspective or explanation style
- Avoid redundancy between answers
- Maintain appropriate complexity for {difficulty} level
- Use clear, professional language
- Include specific details relevant to {domain}

Format your response as a JSON array with objects containing "approach" and "answer" fields:

[
  {{"approach": "concise_definition", "answer": "Your concise answer here..."}},
  {{"approach": "detailed_explanation", "answer": "Your detailed answer here..."}},
  {{"approach": "step_by_step_solution", "answer": "Your step-by-step answer here..."}},
  {{"approach": "analogy_or_example", "answer": "Your example-based answer here..."}}
  {additional_approach_json}
]
"""

CANDIDATE_ANSWERS_PROMPT = """
You are tasked with generating diverse candidate answers for evaluation testing. These answers will be used to test automated grading systems.

Question: {question}
Domain: {domain}
Difficulty Level: {difficulty}

Reference Answers for Context:
{reference_answers}

Generate exactly 6 candidate answers with the following types:

1. perfect_match: Answer that closely matches one of the reference answers (95-100% similarity)
2. alternate_correct: Valid and correct answer but uses different terminology, approach, or phrasing than references (semantically equivalent but textually different)
3. partial_correct: Contains some correct information but missing key details or has minor inaccuracies (60-80% correct)
4. misconception: Reflects common but incorrect understanding in {domain} (plausible but wrong)
5. off_topic: Well-written but addresses the wrong question or irrelevant topic (sounds professional but misses the point)
6. poor_quality: Disorganized, vague, grammatically weak, or confusing (poor structure/clarity)

Guidelines for Natural Human Communication:
- Write as if you're a professional explaining to a colleague in conversation
- Use natural speech patterns with appropriate transitions ("Well," "Actually," "You see," "The thing is")
- Include conversational connectors ("However," "On the other hand," "What's more")
- Vary sentence length naturally (mix short punchy statements with longer explanatory sentences)
- Use professional but approachable tone - not overly formal or robotic
- Include slight hesitations or clarifications that humans naturally use ("In other words," "To put it simply")
- Avoid overly perfect or textbook-like phrasing
- Make explanations flow naturally as if spoken aloud

Additional Guidelines:
- Vary vocabulary, sentence structure, and reasoning style
- Misconceptions should reflect real, common mistakes in {domain}
- Off-topic answers should be plausible but clearly irrelevant
- Poor quality should have structural/clarity issues, not just wrong information
- Maintain appropriate complexity level for {difficulty}
- Test semantic understanding, not just keyword matching

Format your response as a JSON array:

[
  {{"type": "perfect_match", "answer": "Your perfect match answer here..."}},
  {{"type": "alternate_correct", "answer": "Your alternate correct answer here..."}},
  {{"type": "partial_correct", "answer": "Your partially correct answer here..."}},
  {{"type": "misconception", "answer": "Your misconception-based answer here..."}},
  {{"type": "off_topic", "answer": "Your off-topic answer here..."}},
  {{"type": "poor_quality", "answer": "Your poor quality answer here..."}}
]
"""

DOMAIN_SPECIFIC_CONTEXT = {
    "Software Engineering": {
        "context": "Focus on programming concepts, software design, development methodologies, and best practices.",
        "common_misconceptions": "Confusion between concepts like inheritance vs composition, synchronous vs asynchronous, etc.",
    },
    "Data Science": {
        "context": "Emphasize statistical concepts, data analysis techniques, machine learning fundamentals, and data visualization.",
        "common_misconceptions": "Correlation vs causation, overfitting, bias-variance tradeoff misunderstandings.",
    },
    "Machine Learning": {
        "context": "Cover algorithms, model training, evaluation metrics, and practical implementation considerations.",
        "common_misconceptions": "Assuming more data always helps, misunderstanding precision vs recall, feature scaling importance.",
    },
    "Human Resources": {
        "context": "Address employment law, organizational behavior, talent management, and workplace policies.",
        "common_misconceptions": "At-will employment understanding, discrimination vs bias, performance evaluation fairness.",
    },
    "General Knowledge": {
        "context": "Broad topics requiring accurate factual information and clear explanations.",
        "common_misconceptions": "Historical facts, scientific principles, geographical information.",
    },
}


def get_additional_approach(domain: str, difficulty: str) -> tuple:
    """Return additional approach based on domain and difficulty"""
    if domain in ["Data Science", "Machine Learning"] or difficulty == "advanced":
        return (
            "\n5. formula_based: Mathematical or technical formulation with equations/formulas",
            ',\n  {{"approach": "formula_based", "answer": "Your formula-based answer here..."}}',
        )
    return ("", "")


def get_domain_context(domain: str) -> str:
    """Get domain-specific context and common misconceptions"""
    return DOMAIN_SPECIFIC_CONTEXT.get(
        domain,
        {
            "context": "Provide accurate, domain-appropriate information.",
            "common_misconceptions": "Common misunderstandings in the field.",
        },
    )
