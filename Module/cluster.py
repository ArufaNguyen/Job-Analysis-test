import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from .preprocess import stop_word
from .analysis import get_top_words,word_rank,scoring_hook
from .preprocess import process
import itertools
import collections

# Load token
# token = process(r"Data/output.json", "privilege")
def has_repeated_words(ngram):
    words = ngram.split()
    return any(words[i] == words[i+1] for i in range(len(words)-1))

def get_ngrams(tokens, n_min=2, n_max=4):
    ngrams = []
    for n in range(n_min, n_max+1):
        ngrams.extend([" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    return ngrams

def stacking(token,children,n_top,rate_tfidf):
    top_freq,top_tfidf=get_top_words(token,children)
    top_words_result = scoring_hook(top_freq,top_tfidf,rate_tfidf)
    top_words = [w for w, s in top_words_result]  # chỉ lấy từ, bỏ score
    ngrams = get_ngrams(top_words, 2, 4)
    # Lọc stopwords như cũ
    stopwords_phrase = set(stop_word(children))
    ngrams_filtered = [ng for ng in ngrams if ng not in stopwords_phrase and len(set(ng.split())) > 1]
    # Đếm tần suất
    top_ngrams = collections.Counter(ngrams_filtered).most_common(n_top)
    top_ngrams_clean = [(ng, cnt) for ng, cnt in top_ngrams if not has_repeated_words(ng)]
    return top_ngrams_clean

def draw_skidder(top_ngrams_clean):
    bigram_df = pd.DataFrame(top_ngrams_clean, columns=['bigram', 'count'])

    # Chuyển thành dict
    d = bigram_df.set_index('bigram').T.to_dict('records')

    # Tạo network
    G = nx.Graph()

    for k, v in d[0].items():
        # Chia n-gram thành các từ riêng, nối cạnh giữa các từ
        words = k.split()
        for i in range(len(words)-1):
            G.add_edge(words[i], words[i+1], weight=v*10)

    # Ví dụ thêm node đặc biệt
    G.add_node("suicide", weight=100)

    # Vẽ network
    fig, ax = plt.subplots(figsize=(17, 19))
    pos = nx.spring_layout(G, k=2)  # khoảng cách giữa các node

    nx.draw_networkx(G, pos,
                    edge_color='grey',
                    node_color='purple',
                    with_labels=False,
                    ax=ax)

    # Vẽ nhãn với offset
    for key, value in pos.items():
        x, y = value[0]+.135, value[1]+.045
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor='red', alpha=0.25),
                horizontalalignment='center', fontsize=13)

    plt.show()
# draw_skidder(stacking(token,"privilege"))