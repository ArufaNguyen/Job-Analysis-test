from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(txt_file):
    """
    Each document is one line, documents is already preprocess like: remove truncate, tokenize, strip, ...
    :param txt_file: path/to/text/file
    :return: list of documents
    """
    texts = []
    with open(txt_file, 'r', encoding='utf8') as fp:
        for line in fp.readlines():
            texts.append(line.strip())
    return texts


def get_stopwords(documents, threshold=3):
    """
    :param documents: list of documents
    :param threshold:
    :return: list of words has idf <= threshold
    """
    tfidf = TfidfVectorizer(min_df=8)
    tfidf_matrix = tfidf.fit_transform(documents)
    features = tfidf.get_feature_names_out()
    stopwords = []
    print(min(tfidf.idf_), max(tfidf.idf_), len(features))
    for index, feature in enumerate(features):
        if tfidf.idf_[index] <= threshold:
            stopwords.append(feature)
    return stopwords


def mains(file_path):
    docs = load_data(file_path)
    stopwords = get_stopwords(docs, threshold=4.15)
    with open(r'Module\vietnamese_stopwords_master\stopwords.txt', 'w', encoding='utf8') as fp:
        for word in stopwords:
            fp.write(word + '\n')

