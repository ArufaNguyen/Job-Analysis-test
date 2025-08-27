from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from clean_jobs_skibidied import process,stop_word
def get_top_words(tokens, top_n,childrens:str):
    """
    tokens: list các từ đã được tiền xử lý (clean + lowercase + remove stopwords)
    top_n: số lượng từ top muốn lấy
    stopwords: tập stopwords (VN + EN) nếu muốn, có thể None
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
    top_tfidf = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_freq = word_freq.most_common(top_n)

    return top_freq, top_tfidf
token = process(r"Data/output.json", "jobExperience")
print(get_top_words(token,30,"jobExperience"))