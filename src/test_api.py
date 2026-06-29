import json

from src.config import Config
from src.llm_client import generate_json_response

MODELS_TO_TEST = (
    ("Solver 1", Config.SOLVER_1_MODEL),
    ("Solver 2", Config.SOLVER_2_MODEL),
    ("Solver 3", Config.SOLVER_3_MODEL),
    ("Judge", Config.JUDGE_MODEL),
)


def run_connection_test():
    print("Groq API connection test\n")

    if not Config.GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY is not set. Add it to your .env file.")
        return

    passed = 0
    for label, model_name in MODELS_TO_TEST:
        print(f"Testing {label} ({model_name})...")
        response = generate_json_response(
            system_prompt="You are a strict JSON formatting bot.",
            user_prompt=(
                "Create a JSON object with a single key 'status' and the value "
                f"'{label} is online and ready'."
            ),
            model_name=model_name,
            temperature=0.1,
        )

        try:
            payload = json.loads(response)
            print(f"  OK: {payload}\n")
            passed += 1
        except json.JSONDecodeError:
            print(f"  FAIL: invalid JSON response: {response!r}\n")

    print("=== TEST COMPLETE ===")
    print(f"{passed}/{len(MODELS_TO_TEST)} models returned valid JSON.")
    if passed == len(MODELS_TO_TEST):
        print("Your Groq API key and model configuration look good.")
    else:
        print("Some models failed. Check rate limits or model names in src/config.py.")


if __name__ == "__main__":
    run_connection_test()
