from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocess import process,stop_word
def get_top_words(tokens,childrens:str):
    """
    tokens: list các từ đã được tiền xử lý (clean + lowercase + remove stopwords)
    top_n: số lượng từ top muốn lấy
    Children: element từ output.json
    """
    mod_stop_word = stop_word(childrens)
    # 1. Top từ theo tần suất
    word_freq = Counter(tokens)
    # print("Tần suất từ:")
    # print(word_freq.most_common(30))

    # 3. TF-IDF áp dụng trên toàn bộ text (từ list tokens -> chuỗi)
    text_str = " ".join(tokens)
    vectorizer = TfidfVectorizer(stop_words=None) #tách r nên k phải stop word nữa
    tfidf_matrix = vectorizer.fit_transform([text_str])
    scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
    top_tfidf = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_freq = word_freq.most_common()

    return top_freq, top_tfidf

def scoring_hook(top_freq,top_tfidf,rate_tfidf):
    combined = {}
    for word, freq in top_freq:
        tfidf = dict(top_tfidf).get(word, 0)
        combined[word] = freq*(1-rate_tfidf)+ tfidf*rate_tfidf 

    top_words = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return top_words

def word_rank(n_top,file_path,children):
    token = process(file_path, children)
    top_freq, top_tfidf =get_top_words(token,children)
    result = scoring_hook(top_freq, top_tfidf)[:n_top]
    return result


# print(word_rank(20,r"Data/output.json","privilege"))