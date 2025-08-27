import pandas as pd
import re
import json
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import text  
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def clean_text(text: str):
    if pd.isna(text):
        return []
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\s']", ' ', text)     
    text = re.sub(r'\s+', ' ', text).strip()  
    tokens = text.lower().split() 
    return tokens

def stop_word(children : str):
    en_stopwords = list(text.ENGLISH_STOP_WORDS) #vì nó là dạng frozenset :v  
    with open(rf"Data\stopwords\stopwords_CNTT_{children}.txt", "r", encoding="utf-8") as f:
        vn_stopwords = [line.strip() for line in f]
    stopword = en_stopwords + vn_stopwords
    return stopword 

def remove_stop_word(token: list,children :str ):
    stop_words = stop_word(children)
    filtered = [w for w in token if w not in stop_words]
    return filtered

def correct_repeat(original_tokens, processed_tokens):
    """
    Giữ số lần xuất hiện token trong processed_tokens không vượt quá bản gốc
    """
    #i dunno ask chatgpt 
    original_counter = Counter(original_tokens)
    temp_counter = Counter()
    corrected = []

    for token in processed_tokens:
        if temp_counter[token] < original_counter.get(token, 1):
            corrected.append(token)
            temp_counter[token] += 1
    return corrected
def process(file_path:str,children:str,):
    result = []
    with open(r"Data/output.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    jobs = pd.json_normalize(raw["listJob"])
    table = jobs[children]
    coutn_loop = 0
    for i in table:
        coutn_loop = coutn_loop +1
        text = clean_text(i)
        original_text = text
        text_str = " ".join(text)
        vectorizer = CountVectorizer(ngram_range=(2,4), stop_words=None)
        if len(text_str.split()) < 2:
            # print("detected null")
            # print(i,coutn_loop)
            continue 
        X = vectorizer.fit_transform([text_str])
        blacklist_text = vectorizer.get_feature_names_out()
        blacklisted_text = remove_stop_word(blacklist_text,children)
        final_text= " ".join(blacklisted_text)
        final_text = clean_text(final_text)
        final_text = remove_stop_word(final_text,children)
        final = correct_repeat(original_text,final_text)
        for i2 in final:
            result.append(i2)
    return result


def main():

    print(process(r"Data/output.json","jobExperience"))


if __name__ == "__main__":
    main()
