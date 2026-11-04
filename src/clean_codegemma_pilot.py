import pandas as pd

csv_path = "data/outputs/reproduced/GitHubActions_codegemma-7b_pilot.csv"

df = pd.read_csv(csv_path)

def clean_output(text):
    text = str(text).strip()
    lines = text.splitlines()

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()

df["LLM_YML_Output"] = df["LLM_YML_Output"].apply(clean_output)

df.to_csv(csv_path, index=False, encoding="utf-8", quoting=1)

print(f"Cleaned {len(df)} pilot outputs.")