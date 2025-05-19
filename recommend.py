import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

q_df = pd.read_csv("questions.csv")
tfidf = TfidfVectorizer(max_features=500).fit_transform(q_df["text"])

def recommend_next(question_id, top_n=3):
    idx = q_df[q_df["id"]==question_id].index[0]
    sims = cosine_similarity(tfidf[idx], tfidf).flatten()
    sims[idx] = 0
    rec_idx = sims.argsort()[-top_n:][::-1]
    return q_df.iloc[rec_idx][["id","text"]].to_dict(orient="records")
