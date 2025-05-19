import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Load data & model ---
q_df = pd.read_csv("questions.csv")  # id, text, scale, branch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained("./moodly_bert_pt")
model = BertForSequenceClassification.from_pretrained("./moodly_bert_pt").to(device)

# --- 2. TF-IDF recommender ---
vectorizer = TfidfVectorizer(max_features=500).fit(q_df["text"])
tfidf_matrix = vectorizer.transform(q_df["text"])
def recommend_next(qid, top_n=3):
    idx = q_df.index[q_df["id"] == qid][0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = 0
    top = sims.argsort()[-top_n:][::-1]
    return [{"id": int(q_df.iloc[i]["id"]), "text": q_df.iloc[i]["text"]} for i in top]

# --- 3. Persona & dialogues ---
persona_map = {
    "Tệ": ["Chú chó tốt bụng", "Hàng xóm thân thiện"],
    "Cũng ổn": ["Crush lạnh lùng", "Bạn thân thoải mái"],
    "Bình thường": ["Bé mèo chảnh chọe"],
    "Tốt": ["Bà hàng xóm cục súc"],
}
sample_dialogues = {
    "Chú chó tốt bụng": ["Gâu gâu! Mình ở đây để an ủi bạn.", "Chạy loanh quanh cùng mình nhé!"],
    "Hàng xóm thân thiện": ["Chào bạn! Kể cho mình nghe đi.", "Mình luôn lắng nghe."],
    "Crush lạnh lùng": ["Có gì?", "Ừ."],
    "Bạn thân thoải mái": ["Hey, kể mình nghe nào!", "Cứ thoải mái nhé."],
    "Bé mèo chảnh chọe": ["Meo meo... sao vậy?", "Nói nhanh lên đi."],
    "Bà hàng xóm cục súc": ["Nói lẹ đi! Tao bận lắm", "Trời ơi có vậy thôi hả?"],
}

# --- 4. Session init ---
if "stage" not in st.session_state:
    st.session_state.stage     = "init"    # init -> asking -> result -> chat
    st.session_state.branch    = None
    st.session_state.questions = []
    st.session_state.current   = 0
    st.session_state.responses = []
    st.session_state.status    = None
    st.session_state.persona   = None
    st.session_state.chat      = []

st.set_page_config(page_title="Moodly", layout="centered")
st.title("🌈 Moodly – Trắc Nghiệm & Chat")

# --- 5. INIT ---
if st.session_state.stage == "init":
    st.write("Hôm nay bạn thế nào?")
    mood = st.radio("Chọn trạng thái", ["Tệ","Cũng ổn","Bình thường","Tốt"],
                    index=2, label_visibility="collapsed")
    if st.button("Bắt đầu"):
        st.session_state.branch = mood.lower()
        all_q = q_df[q_df["branch"] == st.session_state.branch].to_dict("records")
        st.session_state.questions = all_q[:10]
        st.session_state.stage = "asking"

# --- 6. ASKING ---
elif st.session_state.stage == "asking":
    idx = st.session_state.current
    q   = st.session_state.questions[idx]
    st.write(f"**Câu {idx+1}/10:** {q['text']}")

    key = f"ans_{idx}"
    if key not in st.session_state:
        st.session_state[key] = ""
    def on_answer():
        sel = st.session_state[key]
        if sel in ("Không","Có"):
            st.session_state.responses.append((q["id"],q["text"],sel))
            st.session_state.current += 1
            st.session_state[key] = ""
            if len(st.session_state.responses) >= 10:
                st.session_state.stage = "result"

    st.radio("Chọn đáp án", ["","Không","Có"], key=key,
             on_change=on_answer, label_visibility="collapsed")

    if idx>0 and st.button("◀ Quay lại"):
        if len(st.session_state.responses) > idx-1:
            st.session_state.responses.pop()
        st.session_state.current -= 1

# --- 7. RESULT ---
elif st.session_state.stage == "result":
    st.header("🔍 Kết quả sau 10 câu")
    texts = [f"{t} {1 if a=='Có' else 0}" for _,t,a in st.session_state.responses]
    enc   = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, axis=1).tolist()
    avg   = sum(preds)/len(preds)
    if avg>=2.5:
        status,desc = "Tốt","Bạn đang rất vui vẻ, mọi thứ đều ổn!"
    elif avg>=1.5:
        status,desc = "Bình thường","Bạn ổn định, tiếp tục duy trì nhé."
    elif avg>=0.5:
        status,desc = "Cũng ổn","Bạn vẫn ổn, hãy chăm sóc bản thân."
    else:
        status,desc = "Tệ","Tâm lý không ổn, bạn nên nghỉ ngơi hoặc chia sẻ."
    st.success(desc)
    st.session_state.status = status

    st.subheader("Chọn AI đồng hành")
    persona = st.radio("Persona", persona_map[status],
                       index=0, label_visibility="collapsed")
    if st.button("Bắt đầu chat"):
        st.session_state.persona = persona
        st.session_state.chat = [("AI", sample_dialogues[persona][0])]
        st.session_state.stage = "chat"

# --- 8. CHAT using st.chat_message + st.chat_input ---
else:  # chat
    st.header(f"💬 Chat với {st.session_state.persona}")

    # Display all previous messages
    for speaker, text in st.session_state.chat:
        role = "assistant" if speaker == "AI" else "user"
        with st.chat_message(role):
            st.markdown(text)

    # Accept user input via st.chat_input
    if user_msg := st.chat_input("Gõ tin nhắn của bạn…"):
        # append user message
        st.session_state.chat.append(("You", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        # generate bot reply
        replies = sample_dialogues[st.session_state.persona]
        ai_turn = sum(1 for sp,_ in st.session_state.chat if sp=="AI")
        reply = replies[ai_turn % len(replies)]

        st.session_state.chat.append(("AI", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)

    # Reset button
    if st.button("Làm lại từ đầu"):
        for k in ["stage","current","responses","questions","status","persona","chat"]:
            st.session_state.pop(k, None)
