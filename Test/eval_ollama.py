import ollama

# will be used as variable
MODEL = "gemma3:1b"          # or "phi4-mini", "qwen3:0.6b" 

def ask_llm(question: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        options={"temperature": 0.0}   # for stable responses
    )
    return response['message']['content'].strip()


def simple_exact_match_eval():
    test_cases = [
        ("The capital of Belgium?", "Brussels"),
        ("2 + 2 = ?", "4"),
        ("Red planet?", "Mars"),
    ]

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