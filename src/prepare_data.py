import pandas as pd
from config import DATASET_PATH, FEATURES, TARGET_SAMPLE_SIZE, SAMPLED_PATH

def prepare():
    collected = []
    total = 0

    for chunk in pd.read_csv(
        DATASET_PATH,
        usecols=FEATURES + ["Attack"],
        chunksize=1_000_000
    ):
        chunk = chunk.sample(frac=0.1, random_state=42)
        collected.append(chunk)
        total += len(chunk)

        if total >= TARGET_SAMPLE_SIZE:
            break

    df = pd.concat(collected).sample(TARGET_SAMPLE_SIZE, random_state=42)
    df.to_csv(SAMPLED_PATH, index=False)
    print("Saved sampled dataset:", SAMPLED_PATH)

if __name__ == "__main__":
    prepare()