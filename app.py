import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="College AI Assistant", page_icon="🎓", layout="centered")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.chat-bubble-user {
    background-color: #DCF8C6;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: right;
}
.chat-bubble-bot {
    background-color: #F1F0F0;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 College AI Assistant")
st.write("Ask me anything about courses, fees, admission, facilities, and more!")

# ------------------ LOAD DATA ------------------
data = pd.read_csv("college_chatbot_data.csv")
questions = data["Question"].tolist()
answers = data["Answer"].tolist()

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# ------------------ SESSION STATE ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ SIDEBAR QUICK QUESTIONS ------------------
st.sidebar.title("Quick Questions")

quick_options = [
    "What courses are available?",
    "What is the admission process?",
    "Does the college provide placements?",
    "What are the college timings?"
]

selected_question = None

for q in quick_options:
    if st.sidebar.button(q):
        selected_question = q

# ------------------ USER INPUT ------------------
user_input = st.text_input("Type your question here...", key="input")

# If sidebar button clicked, override text input
if selected_question:
    user_input = selected_question

# ------------------ RESPONSE LOGIC ------------------
if user_input:
    if user_input.lower() in ["hi", "hello", "hey"]:
        response = "Hello! 👋 How can I help you today?"
    else:
        user_vector = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vector, question_vectors)
        best_match_index = similarity.argmax()
        best_score = similarity[0][best_match_index]

        if best_score > 0.5:
            response = answers[best_match_index]
        else:
            response = "I'm not sure about that. You can ask about courses, fees, admission, or facilities."

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# ------------------ DISPLAY CHAT ------------------
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f'<div class="chat-bubble-user">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot">{message}</div>', unsafe_allow_html=True)
