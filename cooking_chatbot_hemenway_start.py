# from google.colab import drive
# drive.mount('/content/drive')


import csv
import pandas as pd




#------- Load a large corpus of text data -------
def load_recipes_csv(csv_path, max_rows=200):
    parts = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            name = row.get("Name", "")
            desc = row.get("Description", "")
            category = row.get("RecipeCategory", "")
            keywords = row.get("Keywords", "")
            ingredients = row.get("RecipeIngredientParts", "")
            instructions = row.get("RecipeInstructions", "")

            keywords = keywords.replace('c("', "").replace('")', "").replace('", "', " ")

            parts.append(
                f"recipe: {name}\n"
                f"description: {desc}\n"
                f"category: {category}\n"
                f"keywords: {keywords}\n"
                f"ingredients: {ingredients}\n"
                f"instructions: {instructions}\n"
            )

    return [p.lower() for p in parts]




#------- Load a large corpus of text data -------
def load_food_parquet(parquet_path, max_rows=200):
    df = pd.read_parquet(parquet_path)

    if max_rows is not None:
        df = df.head(max_rows)

    def col(name):
        if name in df.columns:
            return df[name].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index)

    names = col("name")
    desc = col("description")
    tags = col("tags")
    ingredients = col("ingredients")
    steps = col("steps")

    parts = []
    for i in range(len(df)):
        parts.append(
            f"recipe: {names.iloc[i]}\n"
            f"description: {desc.iloc[i]}\n"
            f"tags: {tags.iloc[i]}\n"
            f"ingredients: {ingredients.iloc[i]}\n"
            f"steps: {steps.iloc[i]}\n"
        )

    return [p.lower() for p in parts]




#------- Load a small corpus of text data -------
def load_cooking_conversions(txt_path, max_rows=None):
    docs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break

            line = line.strip()
            if line:
                docs.append(f"conversion: {line}\n")

    return [d.lower() for d in docs]




# ------- LOCAL -------
# DATASETS = [
#     ("recipes_csv", "project_data/original_files/recipes.csv", load_recipes_csv),
#     ("food_recipes_parquet", "project_data/original_files/food_recipes.parquet", load_food_parquet),
#     ("cooking_conversions", "project_data/original_files/cooking_conversions.txt", load_cooking_conversions),
# ]

# ------- GOOGLE COLAB -------
DATASETS = [
    ("recipes_csv", "/content/drive/MyDrive/Colab_Notebooks/data/recipes.csv", load_recipes_csv),
    ("food_recipes_parquet", "/content/drive/MyDrive/Colab_Notebooks/data/food_recipes.parquet", load_food_parquet),
    ("cooking_conversions", "/content/drive/MyDrive/Colab_Notebooks/data/cooking_conversions.txt", load_cooking_conversions),
]

CORPUS_SIZES = [200, 2000]


EPOCHS_LIST = [3, 10, 20]


ARCHITECTURES = {
    "Arch1_64": {"embed_dim":32, "lstm_units": [64]},
    "Arch2_128": {"embed_dim":64, "lstm_units": [128]},
    "Arch3_128x2": {"embed_dim":64, "lstm_units": [128, 128]},
}

SEQUENCE_LENGTH = 10
MAX_VOCAB = 5000
BATCH_SIZE = 128




def main():
    for dataset_name, path, loader in DATASETS:
        for max_rows in CORPUS_SIZES:

            docs = loader(path, max_rows=max_rows)




if __name__ == "__main__":
    main()