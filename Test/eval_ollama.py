import ollama

# Model configuration
MODEL = "gemma3:1b"          # or "phi4-mini", "qwen3:0.6b"

# Test data - stored separately for easy modification
TEST_DATA = [
    {
        "count": 1,
        "attribute": "car colour",
        "options": ["yellow", "blue", "grey"],
        "url": "https://images.unsplash.com/photo-1595925889916-2a1d773a0613?ixid=M3w4MjcwNjd8MHwxfHNlYXJjaHwyfHxNZXJjZWRlcyUyMGNhcnxlbnwwfHx8fDE3NzMwNjg5Mzl8MA&ixlib=rb-4.1.0&fit=max&q=80",
        "expected": "blue"
    },
    {
        "count": 1,
        "attribute": "car colour",
        "options": ["yellow", "blue", "grey"],
        "url": "https://images.unsplash.com/photo-1605559424843-9e4c228bf1c2?ixid=M3w4MjcwNjd8MHwxfHNlYXJjaHwxfHxNZXJjZWRlcyUyMGNhcnxlbnwwfHx8fDE3NzMwNjg5Mzl8MA&ixlib=rb-4.1.0&fit=max&q=80",
        "expected": "yellow"
    }
]

def generate_question(data: dict) -> str:
    """Generate a question from test data parameters."""
    count = data["count"]
    attribute = data["attribute"]
    options = data["options"]
    url = data["url"]
    options_str = ", ".join(options)
    return f"Select '{count}' best match '{attribute}' from set [{options_str}] for the image {url}"

def generate_test_cases(test_data: list) -> list:
    """Generate test cases from test data configuration."""
    return [(generate_question(item), item["expected"]) for item in test_data]

def ask_llm(question: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        options={"temperature": 0.0}   # for stable responses
    )
    return response['message']['content'].strip()


def simple_exact_match_eval():
    test_cases = generate_test_cases(TEST_DATA)

    correct = 0
    total = len(test_cases)

    print(f"Testing the model: {MODEL}\n")
    
    for question, expected in test_cases:
        answer = ask_llm(question)
        is_correct = expected.lower() in answer.lower()   # softer than == 
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Expected: {expected}  →  {'✓ Correct' if is_correct else '✗ Wrong'}")
        print("-" * 70)
        if is_correct:
            correct += 1

    accuracy = correct / total
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.0%}")


if __name__ == "__main__":
    simple_exact_match_eval()