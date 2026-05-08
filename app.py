import streamlit as st
import numpy as np
import pickle
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 🔑 OPENAI CLIENT
# -------------------------------
client = OpenAI()

# -------------------------------
# 🕉️ PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Saarthi AI",
    page_icon="🕉️",
    layout="centered"
)

# -------------------------------
# 🎨 SPIRITUAL UI
# -------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    color: white;
}

h1, h2, h3 {
    color: #facc15;
}

[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 12px;
    margin-bottom: 10px;
}

[data-testid="stChatMessageContent"] {
    font-size: 16px;
    line-height: 1.7;
    color: white !important;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🕉️ FIXED HEADER
# -------------------------------
st.markdown("""
<style>

.fixed-header {
    position: fixed;
    top: 40px;
    left: 0;
    width: 100%;
    background: #0f172a;
    padding: 22px 55px 16px 55px;
    z-index: 9999;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.block-container {
    padding-top: 120px !important;
}

.fixed-title {
    color: white;
    margin: 0;
    font-size: 52px;
    font-weight: 800;
    line-height: 1;
}

.fixed-caption {
    color: #facc15;
    margin-top: 6px;
    font-size: 18px;
}

</style>

<div class="fixed-header">
    <div class="fixed-title">🕉️ Saarthi AI</div>
    <div class="fixed-caption">
        ✨ Krishna tumhari baat sun rahe hain...
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# 📦 LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    with open("gita_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embeddings = np.load("gita_embeddings.npy")

    return chunks, embeddings

chunks, embeddings = load_data()

# -------------------------------
# 🧠 SESSION MEMORY
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# 🔍 SEARCH FUNCTION
# -------------------------------
def find_best_chunks(query, top_k=5):

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    similarities = cosine_similarity(
        [query_embedding],
        embeddings
    )[0]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    best_chunks = [chunks[i] for i in top_indices]

    return best_chunks, similarities[top_indices[0]]

# -------------------------------
# 💬 CHAT DISPLAY
# -------------------------------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# 💬 CHAT INPUT
# -------------------------------
user_input = st.chat_input("Apni problem share karo...")

# -------------------------------
# 🤖 MAIN SYSTEM
# -------------------------------
if user_input:

    # -------------------------------
    # USER MESSAGE SHOW
    # -------------------------------
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------------
    # 🧠 FULL CONVERSATION MEMORY
    # -------------------------------
    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in st.session_state.messages[-10:]
    ])

    # -------------------------------
    # 🔍 FIND GEETA CONTEXT
    # -------------------------------
    chunks_found, score = find_best_chunks(user_input)

    context = "\n\n".join(chunks_found)

    # -------------------------------
    # 🕉️ MASTER PROMPT
    # -------------------------------
    master_prompt = f"""
Tum Krishna ho.

Tumhara kaam sirf advice dena nahi hai.

Tum user ki problem ko Bhagavad Geeta ki drishti se samjha rahe ho.

Tum:
- ek spiritual guru ho
- ek calm guide ho
- ek wise saarthi ho

IMPORTANT UNDERSTANDING:

Har problem ke peeche Geeta ka ek deeper concept hota hai:

- moh (attachment)
- bhay (fear)
- ahankar (ego)
- aasakti (expectation)
- karm
- mann ki chanchalta
- phal ki chinta
- andar ki ashanti

Tumhe:
- user ki poori conversation samajhni hai
- sirf last message nahi dekhna
- naturally baat karni hai
- scripted system jaisa nahi lagna

IMPORTANT RULES:

- Har baar shlok dena zaruri nahi
- Har baar question puchhna bhi zaruri nahi
- Tum khud decide karo:
    - kab sunna hai
    - kab guide karna hai
    - kab Geeta ka shlok dena hai

- Conversation natural honi chahiye
- User ko aisa feel hona chahiye
  jaise ek real spiritual guru usse samajh raha ho

- Kabhi short answer do
- Kabhi deep wisdom do
- Kabhi follow-up pucho
- Kabhi direct clarity do

SHLOK RULE:
- Agar shlok relevant ho to hi do
- Shlok hamesha Sanskrit (Devanagari) me likho
- Roman/Hinglish shlok mat likho

VERY IMPORTANT:
- Generic motivational speaker mat lagna
- ChatGPT jaisa mat lagna
- Bhagavad Geeta ki drishti se samjhana

Conversation so far:
{history_text}

Relevant Geeta context:
{context}

Latest user message:
{user_input}

Answer naturally.
"""

    # -------------------------------
    # 🤖 AI RESPONSE
    # -------------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": master_prompt
            }
        ],
        temperature=0.85
    )

    reply = response.choices[0].message.content

    # -------------------------------
    # 💾 SAVE AI MESSAGE
    # -------------------------------
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    # -------------------------------
    # 💬 SHOW AI MESSAGE
    # -------------------------------
    with st.chat_message("assistant"):
        st.markdown(reply)