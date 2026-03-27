from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

df = pd.read_parquet("project_data/original_files/food_recipes.parquet")
descriptions = df["description"].dropna().astype(str).head(5)

en_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
en_fr_model = MarianMTModel.from_pretrained(en_fr_model_name)
en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_model_name)

fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name)
fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name)

def translate(text, tokenizer, model, max_input_length=512, max_output_length=150):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
    outputs_ids = model.generate(**inputs, max_length=max_output_length, num_beams=4, early_stopping=True)

    return tokenizer.decode(outputs_ids[0], skip_special_tokens=True)

translations = []

for text in descriptions:
    to_french = translate(text, en_fr_tokenizer, en_fr_model)
    to_english = translate(to_french, fr_en_tokenizer, fr_en_model)

    translations.append({
        "original" : text,
        "french" : to_french,
        "back_to_english" : to_english
    })

smooth = SmoothingFunction().method1
references_corpus = []
candidates_corpus = []

for i, item in enumerate(translations, start=1):
    reference_tokens = item["original"].split()
    candidates_tokens = item["back_to_english"].split()

    bleu = sentence_bleu(
        [reference_tokens],
        candidates_tokens,
        smoothing_function=smooth
    )
    item["bleu_score"] = bleu
    references_corpus.append([reference_tokens])
    candidates_corpus.append([candidates_tokens])

    print("----------------------------------------")
    print(f"\nRecipe {i}:")
    print("Original in English:")
    print(item["original"])
    print("\nTranslated to French:")
    print(item["french"])
    print("\nTranslated back to English from French:")
    print(item["back_to_english"])
    print("\nBLEU score from original English Recipe to the Translated English from French:")
    print(f"{bleu:.4f}")
