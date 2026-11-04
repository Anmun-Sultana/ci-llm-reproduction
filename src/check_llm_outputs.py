import pandas as pd
from pathlib import Path

data_folder = Path("data/raw")

model_files = [
    "GitHubActions_gpt-4o_output.csv",
    "GitHubActions_gpt-4.1_output.csv",
    "GitHubActions_gemma3-12b_output.csv",
    "GitHubActions_llama3.1-8b_output.csv",
    "GitHubActions_codegemma-7b_output.csv",
    "GitHubActions_codellama-7b_output.csv",
]

for file_name in model_files:
    file_path = data_folder / file_name

    print("\n" + "=" * 60)
    print("FILE:", file_name)

    df = pd.read_csv(file_path)

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "LLM_YML_Output" in df.columns:
        print("Generated YAML column: FOUND")
        print("Non-empty generated outputs:",
              df["LLM_YML_Output"].fillna("").astype(str).str.strip().ne("").sum())

        print("\nExample generated YAML:")
        print(df["LLM_YML_Output"].dropna().iloc[0][:500])
    else:
        print("Generated YAML column: NOT FOUND")