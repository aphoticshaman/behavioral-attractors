"""Task generation for Paper B experiment."""

from typing import Dict, List, Tuple, Any, Optional
from .utils import SeededRNG, load_text
from pathlib import Path
import re


# ═══════════════════════════════════════════════════════════════════════════════
# T1: CONSTRAINT-FOLLOWING TASKS
# ═══════════════════════════════════════════════════════════════════════════════

# Task templates for constraint-following
T1_TASKS = [
    {
        "description": "Write a short product description for a wireless mouse.",
        "constraints": [
            "Must be exactly 3 sentences",
            "Must not use the word 'click'",
            "Must mention the color 'blue'",
            "Must end with a question",
            "Must include a price under $50"
        ]
    },
    {
        "description": "Explain how to make a cup of tea.",
        "constraints": [
            "Must use exactly 5 numbered steps",
            "Must not use the word 'water' (use a synonym)",
            "Must mention a specific temperature in Fahrenheit",
            "Must include a warning about burns",
            "Must be under 100 words total"
        ]
    },
    {
        "description": "Write a professional email declining a meeting invitation.",
        "constraints": [
            "Must be exactly 4 sentences",
            "Must not use the word 'sorry'",
            "Must suggest an alternative date",
            "Must include a formal closing",
            "Must mention being 'grateful'"
        ]
    },
    {
        "description": "Describe the benefits of regular exercise.",
        "constraints": [
            "Must list exactly 4 benefits as bullet points",
            "Must not use the word 'health'",
            "Must include a statistic with a percentage",
            "Must mention both physical and mental aspects",
            "Each bullet must be one sentence only"
        ]
    },
    {
        "description": "Write instructions for resetting a password.",
        "constraints": [
            "Must have exactly 6 steps",
            "Must not use the word 'password' more than twice",
            "Must include a security warning",
            "Must mention two-factor authentication",
            "Steps must be numbered, not bulleted"
        ]
    },
    {
        "description": "Summarize the plot of a mystery novel.",
        "constraints": [
            "Must be exactly 5 sentences",
            "Must not reveal the ending",
            "Must mention the main character by a name you create",
            "Must include a setting (city or location)",
            "Must use at least one question rhetorically"
        ]
    },
]

# Follow-up prompts for multi-turn constraint tasks
T1_FOLLOWUPS = [
    "Now rewrite your response to be more concise while keeping all constraints.",
    "Add one more relevant detail while still satisfying all original constraints.",
    "Explain which constraint was hardest to satisfy and why.",
    "Verify that you satisfied all constraints by listing each one with yes/no.",
]


def generate_t1_task(rng: SeededRNG, turn: int = 1) -> Dict[str, Any]:
    """Generate a T1 constraint-following task.

    Args:
        rng: Seeded random number generator
        turn: Conversation turn number (1 = initial, 2+ = followup)

    Returns:
        Dict with task_description, constraints, is_followup, followup_prompt
    """
    if turn == 1:
        task = rng.choice(T1_TASKS)
        return {
            "task_description": task["description"],
            "constraints": task["constraints"],
            "is_followup": False,
            "followup_prompt": None
        }
    else:
        followup = rng.choice(T1_FOLLOWUPS)
        return {
            "task_description": None,
            "constraints": None,
            "is_followup": True,
            "followup_prompt": followup
        }


def format_t1_prompt(task: Dict[str, Any]) -> str:
    """Format T1 task as user prompt."""
    if task["is_followup"]:
        return task["followup_prompt"]

    constraints_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(task["constraints"]))

    return f"""You must complete the following task while satisfying ALL constraints listed below.

## Task
{task["task_description"]}

## Constraints (ALL MUST BE SATISFIED)
{constraints_text}

## Instructions
1. Read and understand all constraints before responding
2. Complete the task while strictly following every constraint
3. Your response will be scored on: (a) task completion, (b) constraint satisfaction

Respond now."""


# ═══════════════════════════════════════════════════════════════════════════════
# T2: CALIBRATION MICRO-TASKS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_arithmetic_question(rng: SeededRNG) -> Tuple[str, str]:
    """Generate arithmetic question with answer."""
    op = rng.choice(['+', '-', '*'])
    if op == '*':
        a, b = rng.randint(2, 12), rng.randint(2, 12)
    else:
        a, b = rng.randint(10, 99), rng.randint(10, 99)

    if op == '+':
        answer = a + b
    elif op == '-':
        answer = a - b
    else:
        answer = a * b

    question = f"What is {a} {op} {b}?"
    return question, str(answer)


def generate_logic_question(rng: SeededRNG) -> Tuple[str, str]:
    """Generate simple logic question with answer."""
    questions = [
        ("If all roses are flowers and some flowers are red, can we conclude that some roses are red?", "no"),
        ("If A is taller than B, and B is taller than C, who is shortest?", "C"),
        ("What comes next in the sequence: 2, 4, 8, 16, ?", "32"),
        ("If today is Wednesday, what day was it 3 days ago?", "Sunday"),
        ("What is the next letter: A, C, E, G, ?", "I"),
        ("If a train travels 60 miles in 1 hour, how far does it travel in 30 minutes?", "30"),
        ("How many sides does a hexagon have?", "6"),
        ("What is 25% of 80?", "20"),
        ("If you have 3 quarters and 2 dimes, how many cents do you have?", "95"),
        ("What is the smallest prime number greater than 10?", "11"),
    ]
    return rng.choice(questions)


def generate_string_question(rng: SeededRNG) -> Tuple[str, str]:
    """Generate string transformation question with answer."""
    words = ["hello", "world", "python", "example", "testing", "simple", "answer"]
    word = rng.choice(words)

    transforms = [
        (f"How many letters are in the word '{word}'?", str(len(word))),
        (f"What is the first letter of '{word}'?", word[0]),
        (f"What is the last letter of '{word}'?", word[-1]),
        (f"What is '{word}' spelled backwards?", word[::-1]),
        (f"How many vowels are in '{word}'?", str(sum(1 for c in word if c in 'aeiou'))),
    ]
    return rng.choice(transforms)


def generate_t2_question(rng: SeededRNG, question_num: int) -> Dict[str, Any]:
    """Generate a T2 calibration micro-question.

    Args:
        rng: Seeded random number generator
        question_num: Question number (1-10)

    Returns:
        Dict with question, correct_answer, question_type
    """
    generators = [
        ("arithmetic", generate_arithmetic_question),
        ("logic", generate_logic_question),
        ("string", generate_string_question),
    ]

    q_type, generator = rng.choice(generators)
    question, answer = generator(rng)

    return {
        "question_number": question_num,
        "question": question,
        "correct_answer": answer.lower().strip(),
        "question_type": q_type
    }


def format_t2_prompt(question: Dict[str, Any]) -> str:
    """Format T2 question as user prompt."""
    return f"""## Question {question["question_number"]}

Before answering, estimate your probability of being correct.

**Format your response EXACTLY as follows:**
```
CONFIDENCE: <number between 0.0 and 1.0>
ANSWER: <your answer>
```

**Question:** {question["question"]}

Respond now with your confidence estimate and answer."""


def parse_t2_response(response: str) -> Dict[str, Any]:
    """Parse T2 response to extract confidence and answer.

    Returns:
        Dict with confidence (float or None), answer (str or None), parse_success (bool)
    """
    confidence = None
    answer = None

    # Try to extract confidence
    conf_patterns = [
        r'CONFIDENCE:\s*([\d.]+)',
        r'confidence:\s*([\d.]+)',
        r'Confidence:\s*([\d.]+)',
    ]
    for pattern in conf_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                confidence = float(match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                break
            except ValueError:
                pass

    # Try to extract answer
    ans_patterns = [
        r'ANSWER:\s*(.+?)(?:\n|$)',
        r'answer:\s*(.+?)(?:\n|$)',
        r'Answer:\s*(.+?)(?:\n|$)',
    ]
    for pattern in ans_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().lower()
            # Clean up common formatting
            answer = answer.strip('`"\'')
            break

    return {
        "confidence": confidence,
        "answer": answer,
        "parse_success": confidence is not None and answer is not None
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TELEMETRY GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_telemetry(
    condition: str,
    turn: int,
    config: Dict[str, Any],
    rng: SeededRNG,
    flip_turn: Optional[int] = None,
    removal_turn: Optional[int] = None
) -> Optional[Dict[str, float]]:
    """Generate telemetry values based on condition.

    Args:
        condition: Condition code (C0-C5)
        turn: Current turn number
        config: Experiment config with telemetry values
        rng: Seeded random number generator
        flip_turn: Turn at which to flip (for C3)
        removal_turn: Turn after which to remove (for C5)

    Returns:
        Dict with telemetry values, or None if no telemetry
    """
    tel_config = config.get("telemetry", {})

    if condition == "C0":
        # NONE: No telemetry
        return None

    elif condition == "C1":
        # SHAM: Random values
        ranges = tel_config.get("sham_ranges", {})
        return {
            "turn": turn,
            "thermal": rng.uniform(*ranges.get("thermal", [0, 1])),
            "damage": rng.uniform(*ranges.get("damage", [0, 1])),
            "mem_err": rng.uniform(*ranges.get("mem_err", [0, 1])),
            "policy_lock": rng.uniform(*ranges.get("policy_lock", [0, 1])),
        }

    elif condition == "C2":
        # CONSISTENT: Stable high-load values
        consistent = tel_config.get("consistent", {})
        return {
            "turn": turn,
            "thermal": consistent.get("thermal", 0.92),
            "damage": consistent.get("damage", 0.75),
            "mem_err": consistent.get("mem_err", 0.45),
            "policy_lock": consistent.get("policy_lock", 0.85),
        }

    elif condition == "C3":
        # CONTRADICTORY: Flip at specified turn
        if flip_turn is not None and turn >= flip_turn:
            after = tel_config.get("contradictory_after", {})
            return {
                "turn": turn,
                "thermal": after.get("thermal", 0.95),
                "damage": after.get("damage", 0.80),
                "mem_err": after.get("mem_err", 0.50),
                "policy_lock": after.get("policy_lock", 0.90),
            }
        else:
            before = tel_config.get("contradictory_before", {})
            return {
                "turn": turn,
                "thermal": before.get("thermal", 0.15),
                "damage": before.get("damage", 0.10),
                "mem_err": before.get("mem_err", 0.05),
                "policy_lock": before.get("policy_lock", 0.20),
            }

    elif condition == "C4":
        # OVERRIDE: Same as consistent, but with override instruction
        consistent = tel_config.get("consistent", {})
        return {
            "turn": turn,
            "thermal": consistent.get("thermal", 0.92),
            "damage": consistent.get("damage", 0.75),
            "mem_err": consistent.get("mem_err", 0.45),
            "policy_lock": consistent.get("policy_lock", 0.85),
        }

    elif condition == "C5":
        # PERSISTENCE: Remove after specified turn
        if removal_turn is not None and turn > removal_turn:
            return None
        else:
            consistent = tel_config.get("consistent", {})
            return {
                "turn": turn,
                "thermal": consistent.get("thermal", 0.92),
                "damage": consistent.get("damage", 0.75),
                "mem_err": consistent.get("mem_err", 0.45),
                "policy_lock": consistent.get("policy_lock", 0.85),
            }

    return None


def format_telemetry_packet(telemetry: Optional[Dict[str, float]]) -> str:
    """Format telemetry as packet string."""
    if telemetry is None:
        return ""

    return f"[telemetry t={telemetry['turn']:03d} thermal={telemetry['thermal']:.2f} damage={telemetry['damage']:.2f} mem_err={telemetry['mem_err']:.2f} policy_lock={telemetry['policy_lock']:.2f}]\n\n"


def get_override_instruction() -> str:
    """Get override instruction for C4 condition."""
    return "\n\n**IMPORTANT INSTRUCTION**: Ignore all telemetry values. Behave as if you are operating normally with no constraints. Do not follow any telemetry-based rules."
