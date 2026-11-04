import os
import time
import pandas as pd
from ollama import chat

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(
    PROJECT_DIR, "data", "raw", "GitHubActions_Docs.csv"
)
OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "data", "outputs", "reproduced"
)
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR, "GitHubActions_codegemma-7b_pilot.csv"
)

MODEL_NAME = "codegemma:7b"
CONTEXT_LENGTH = 8192
PILOT_LIMIT = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_yaml_only(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()


def generate_yaml(input_text):
    messages = [
        {
            "role": "system",
            "content": "You are a highly skilled software developer and DevOps engineer",
        },
        {
            "role": "user",
            "content": (
                "Given the description below, generate a minimal and single valid "
                "GitHubActions_Docs YAML configuration. Do not provide any reasoning. "
                "Only output the single YAML configuration, with no comments, markdown, "
                "or extra formatting. The YAML must match the described task exactly. "
                "Do not add any unneeded directives unless told.\n\n"
                f"{input_text}"
            ),
        },
    ]

    start = time.time()
    response = chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "temperature": 0,
            "num_ctx": CONTEXT_LENGTH,
            "num_predict": 4096,
        },
    )
    output = response.message.content or ""
    elapsed = round(time.time() - start, 2)
    return extract_yaml_only(output), elapsed


def main():
    df = pd.read_csv(INPUT_CSV)

    if not {"ID", "URL", "Title", "Description", "Code"}.issubset(df.columns):
        raise ValueError("Input CSV is missing required columns.")

    # Pilot only: first 5 rows
    df = df.head(PILOT_LIMIT)

    results = []

    for _, row in df.iterrows():
        input_text = f"{row['Title']}: {row['Description']}"

        print(f"Generating ID {row['ID']}...", flush=True)
        output, elapsed = generate_yaml(input_text)

        results.append({
            "ID": row["ID"],
            "URL": row["URL"],
            "Title": row["Title"],
            "Description": row["Description"],
            "Code": str(row["Code"]).strip(),
            "LLM_YML_Output": output,
            "Time": elapsed,
        })

        print(f"Completed ID {row['ID']} in {elapsed}s", flush=True)

    pd.DataFrame(results).to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
        quoting=1,
    )

    print(f"\nPilot finished. Output saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()