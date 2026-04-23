import os
import pandas as pd
from github import Github, Auth
from dotenv import load_dotenv

load_dotenv()

QUERIES = [
    "topic:machine-learning topic:research stars:>50",
    "topic:deep-learning topic:paper stars:>50",
    "topic:nlp topic:research stars:>50",
    "topic:computer-vision topic:research stars:>50",
]

RAW_PATH = "data/raw/repos_raw.csv"
PROCESSED_PATH = "data/processed/repos_clean.csv"

def connect_github():
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN was not found in the .env file")

    auth = Auth.Token(github_token)
    return Github(auth=auth)

def fetch_repositories(github_client, queries, max_repos_per_query=25):
    repos_data = []
    seen_ids = set()

    for query in queries:
        results = github_client.search_repositories(query=query)

        for repo in results[:max_repos_per_query]:
            if repo.id in seen_ids:
                continue

            seen_ids.add(repo.id)

            repos_data.append({
                "repo_id": repo.id,
                "nombre": repo.full_name,
                "descripcion": repo.description,
                "lenguaje": repo.language,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "topics": repo.get_topics(),
                "fecha_creacion": repo.created_at,
                "ultimo_commit": repo.pushed_at,
                "url": repo.html_url,
            })

    return pd.DataFrame(repos_data)

def save_raw_dataset(df, output_path=RAW_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Raw dataset saved to: {output_path}")

def clean_and_engineer_features(df):
    df = df.copy()

    df["descripcion"] = df["descripcion"].fillna("No description")
    df["lenguaje"] = df["lenguaje"].fillna("Unknown")

    df["fecha_creacion"] = pd.to_datetime(df["fecha_creacion"], utc=True)
    df["ultimo_commit"] = pd.to_datetime(df["ultimo_commit"], utc=True)

    ahora = pd.Timestamp.now(tz="UTC")
    df["dias_sin_commit"] = (ahora - df["ultimo_commit"]).dt.days
    df["antiguedad_dias"] = (ahora - df["fecha_creacion"]).dt.days
    df["num_topics"] = df["topics"].apply(len)

    return df

def save_processed_dataset(df, output_path=PROCESSED_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")

def main():
    github_client = connect_github()

    # Step 1: ingestion from API
    raw_df = fetch_repositories(github_client, QUERIES)

    # Step 2: save raw dataset
    save_raw_dataset(raw_df)

    # Step 3: clean raw dataset
    clean_df = clean_and_engineer_features(raw_df)

    # Step 4: save processed dataset
    save_processed_dataset(clean_df)

    print(f"Final processed shape: {clean_df.shape}")
    print("\nMissing values after cleaning:")
    print(clean_df.isnull().sum())

if __name__ == "__main__":
    main()