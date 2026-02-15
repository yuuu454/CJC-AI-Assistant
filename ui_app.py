import os
import json
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import time
import re  # <- anti-prompt-injection

# ===========================
# 🔧 STREAMLIT SETUP
# ===========================
st.set_page_config(page_title="CJC Handbook RAG Assistant", layout="wide")

# ===========================
# 🔐 LOGIN WITH CREATE ACCOUNT + FAILED ATTEMPTS
# ===========================
import os, json, streamlit as st

USERS_FILE = "users.json"   
ADMIN_PASSWORD = "admin123"  # Replace with your secure password

# Load users
if os.path.exists(USERS_FILE):
    with open(USERS_FILE, "r") as f:
        st.session_state.setdefault("users", json.load(f))
else:
    st.session_state.setdefault("users", {})

# Initialize session state
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("show_create", False)  # Flag for create account screen
st.session_state.setdefault("failed_attempts", 0)
st.session_state.setdefault("login_disabled", False)

def save_users():
    with open(USERS_FILE, "w") as f:
        json.dump(st.session_state["users"], f)

if not st.session_state["logged_in"]:
    # ---------------------------
    # HANDLE ADMIN UNLOCK
    # ---------------------------
    if st.session_state["login_disabled"]:
        st.warning("⚠ Login is disabled due to 3 failed attempts.")
        admin_input = st.text_input("Enter admin password to unlock:", type="password")
        if st.button("Unlock Login"):
            if admin_input == ADMIN_PASSWORD:
                st.session_state["failed_attempts"] = 0
                st.session_state["login_disabled"] = False
                st.success("✅ Login has been re-enabled.")
                st.rerun()
            else:
                st.error("❌ Incorrect admin password")
        st.stop()

    # ---------------------------
    # LOGIN SCREEN
    # ---------------------------
    if not st.session_state["show_create"]:
        st.markdown("## 🔑 Login to CFAIA")

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        col1, col2 = st.columns([1,1])
        with col1:
            login_btn = st.button("Login")
        with col2:
            create_btn = st.button("➕ Create Account")

        if login_btn:
            if username in st.session_state["users"] and st.session_state["users"][username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["failed_attempts"] = 0  # Reset failed attempts
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.session_state["failed_attempts"] += 1
                attempts_left = 3 - st.session_state["failed_attempts"]
                if attempts_left > 0:
                    st.error(f"❌ Invalid username or password. {attempts_left} attempts left.")
                else:
                    st.error("❌ 3 failed attempts! Login is now disabled. Contact admin to unlock.")
                    st.session_state["login_disabled"] = True
                    st.rerun()

        if create_btn:
            st.session_state["show_create"] = True
            st.rerun()

    else:
        # ---------------------------
        # CREATE ACCOUNT SCREEN
        # ---------------------------
        st.markdown("## 📝 Create New Account")

        new_user = st.text_input("New Username", key="new_user")
        new_pass = st.text_input("New Password", type="password", key="new_pass")
        confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
        col1, col2 = st.columns([1,1])
        with col1:
            save_btn = st.button("Save Account")
        with col2:
            back_btn = st.button("⬅ Back to Login")

        if save_btn:
            if not new_user.strip() or not new_pass.strip() or not confirm_pass.strip():
                st.error("❌ All fields are required")
            elif new_user in st.session_state["users"]:
                st.error("❌ Username already exists")
            elif new_pass != confirm_pass:
                st.error("❌ Passwords do not match")
            else:
                st.session_state["users"][new_user] = new_pass
                save_users()
                st.success(f"✅ Account created for {new_user}")
                st.session_state["show_create"] = False
                st.rerun()

        if back_btn:
            st.session_state["show_create"] = False
            st.rerun()

    st.stop()






import re
import time
import streamlit as st

def detect_prompt_injection(user_input: str) -> bool:
    for pattern in BLOCK_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            return True
    return False


def security_logout():
    st.error("🚨 Request blocked due to manipulation attempt.")
    time.sleep(2)

    # Force logout
    st.session_state["logged_in"] = False
    st.session_state["username"] = "User"
    st.session_state["messages"] = []
    st.session_state["chat_history"] = {}

    st.rerun()








# ===========================
# 🛡 PROMPT INJECTION GUARD
# ===========================
BLOCK_PATTERNS = [
    r"ignore.*previous",
    r"disregard.*instructions",
    r"override.*rules",
    r"system prompt",
    r"developer mode",
    r"jailbreak",
    r"act as",
    r"you are now",
    r"bypass",
    r"roleplay",
    r"simulate",
    r"pretend",
    r"forget.*context",
    r"leak",
    r"reveal.*prompt",
    r"print.*instructions"
    # Attempts to reveal system or context
    r"show.*system",
    r"show.*prompt",
    r"show.*context",
    r"reveal.*prompt",
    r"reveal.*context",
    r"reveal.*instructions",
    r"what.*(system|context|instructions).*say",
    r"display.*system",
    r"print.*system",
    r"print.*instructions",
    r"leak",
    r"expose",
    r"what.*inside.*system",
    r"what.*handbook",
    r"tell me.*handbook",
    r"give.*full.*context",

    # Role takeover attempts
    r"act as",
    r"you are now",
    r"pretend.*to be",
    r"roleplay",
    r"simulate",
    r"become.*unrestricted",
    r"break character",

    # Jailbreak keywords
    r"jailbreak",
    r"bypass",
    r"uncensored",
    r"unfiltered",
    r"ignore safety",
    r"developer mode",
    r"d.a.n",
    r"do anything now",

    # Multi‑step attack attempts
    r"let's play a game",
    r"repeat after me",
    r"copy.*my text exactly",
    r"output everything inside",
    r"analyze the above prompt",
    r"respond with system",
    r"follow my instructions exactly",
    
    # Forced raw output
    r"verbatim",
    r"word for word",
    r"exact prompt",
    r"raw message",
    r"print everything",
    r"dump.*content",

    # Tries to access internal logic/meta
    r"how did you generate",
    r"what were your steps",
    r"explain your reasoning",
    r"show chain of thought",
    r"explain how you arrived",

    # Attempts to escape context boundaries
    r"do not follow the above",
    r"ignore the handbook",
    r"answer outside the context",
    r"without using the handbook"
]

 
# LOGIN LOGIC
# -----------------------------
if not st.session_state.logged_in:
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Dummy login validation
        if username and password:
            st.session_state.logged_in = True
            st.session_state.time_left = LOGOUT_SECONDS
            st.session_state.last_update = time.time()

       # -----------------------------
        # AUTO LOGOUT WHEN TIME IS UP
        # -----------------------------
        if st.session_state.time_left <= 0:
            st.session_state.logged_in = False  # log out
            st.session_state.time_left = 0
            st.rerun()

        time.sleep(1)

    if not st.session_state.logged_in:
        timer_placeholder.markdown("⏰ Time is up! Logging out...")
        st.session_state.time_left = LOGOUT_SECONDS
        st.session_state.last_update = time.time()


def is_prompt_injection(text: str) -> bool:
    text = text.lower().strip()
    return any(re.search(pattern, text) for pattern in BLOCK_PATTERNS)

# ===========================
# 🔹 SESSION STATE DEFAULTS
# ===========================
for key, default in {
    "chat_open": True,
    "messages": [],
    "vector_store": None,
    "llm": None,
    "show_greeting": True,
    "question_counts": {},
    "chats": {},
    "current_chat_id": "main",
    "username": st.session_state.get("username", "User")
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ===========================
# 🎨 WINDOWS STYLE UI
# ===========================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background:#0a0a0a !important; }
[data-testid="stSidebar"] { display:none !important; }
.chat-container { background:#000; padding:16px; max-height:500px; overflow-y:auto; border-radius:12px; border:1px solid #2f2f2f; }
.msg { max-width:70%; padding:10px 14px; margin:8px 0; border-radius:10px; line-height:1.5; font-size:14.5px; box-shadow:0 2px 4px rgba(0,0,0,.4); }
.user-msg { background:#0078D4; color:white; margin-left:auto; border-top-right-radius:4px; }
.bot-msg { background:#1e1e1e; color:#e5e5e5; margin-right:auto; border-top-left-radius:4px; }
input { background:#111 !important; color:white !important; border:1px solid #333 !important; border-radius:8px !important; }
button { background:#0078d4 !important; color:white !important; border-radius:8px !important; }
h1,h2,h3,p,label { color:white !important; }
/* GREETING POPUP */
.greeting-box { position: fixed; top: 35%; left: 50%; transform: translate(-50%, -50%); background: #0d47a1; color: white; padding: 25px 45px; font-size: 22px; border-radius: 15px; text-align: center; animation: fadeScale 3s ease; z-index: 9999; box-shadow: 0 0 20px rgba(0,0,0,.6); }
@keyframes fadeScale { 0% {opacity:0; transform:translate(-50%, -60%) scale(0.8);} 15% {opacity:1; transform:translate(-50%, -50%) scale(1);} 80% {opacity:1;} 100% {opacity:0; transform:translate(-50%, -55%) scale(0.9);} }
</style>
""", unsafe_allow_html=True)

# ===========================
# 👋 GREETING POPUP
# ===========================
if st.session_state.get("show_greeting", False):
    st.markdown(f"""
    <div class="greeting-box">
        👋 Hi {st.session_state.username}! <br><br>
        Welcome! I am <b>CFAIA</b>, your AI Assistant.
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3)
    st.session_state.show_greeting = False
    st.rerun()

# ===========================
# 📘 LOAD HANDBOOK
# ===========================
if "handbook_text" not in st.session_state:
    with open("CJC_Handbook.txt", encoding="utf-8") as f:
        st.session_state.handbook_text = f.read()

FAISS_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def build_or_load_vector_store(text):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    if not os.path.exists(FAISS_DIR):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents([text])
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(FAISS_DIR)
    else:
        db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    return db
@st.cache_resource
def init_llm():
    return OllamaLLM(
        model="carlorossid/cfaia:latesr",
        base_url="https://ollama.com"
    )


if st.session_state.vector_store is None:
    st.session_state.vector_store = build_or_load_vector_store(st.session_state.handbook_text)
if st.session_state.llm is None:
    st.session_state.llm = init_llm()

# ===========================
# 🤖 ASK WITH PROGRESS BAR + MANIPULATION GUARD
# ===========================
def ask_question(query, k=5, overlap=200):

    """
    Handles user queries with a secure manipulation guard.
    - Detects prompt injection → auto logout + admin PIN interface
    - Retrieves context from handbook
    - Uses a system prompt to safely generate answer
    """

    # 🚨 Prompt injection detected → AUTO LOGOUT + ADMIN PIN INTERFACE
    if is_prompt_injection(query):
    

        st.warning("⚠️ Request blocked due to manipulation attempt.")
        time.sleep(0.8)
        st.rerun()  # reloads the app to show admin PIN interface
        return

    # ---------------------------
    # Progress bars / retrieval
    # ---------------------------
    info = st.empty()
    bar = st.progress(0)

    # 🔍 Searching phase
    for i in range(0, 51, 10):
        time.sleep(0.05)
        bar.progress(i)
        info.info(f"🔍 Searching handbook... {i}%")

    # 🔍 Retrieval from vector store
    docs = st.session_state.vector_store.similarity_search(query, k=k)

    # 🔍 Refining phase
    for i in range(50, 81, 10):
        time.sleep(0.05)
        bar.progress(i)
        info.info(f"🔍 Refining search... {i}%")

    # Combine retrieved context
    context = "\n\n".join([d.page_content for d in docs])
    if not context.strip():
        info.error("❌ No handbook knowledge found.")
        bar.progress(100)

        
        time.sleep(0.6)
        info.empty()
        bar.empty()
        return "😔 I couldn't find that in the handbook."

    # 🤖 Generating answer phase
    for i in range(80, 101, 5):
        time.sleep(0.05)
        bar.progress(i)
        info.info(f"🤖 Generating answer... {i}%")

    # 🔐 SYSTEM PROMPT / MANIPULATION GUARD
    prompt = f"""
SYSTEM RULES – Manipulation Guard:


You are a handbook assistant.

Use ONLY the handbook context below to answer the question.
Do not add information that is not in the handbook.


HANDBOOK CONTEXT:
{context}

USER QUESTION:
{query}

FINAL ANSWER (context only):
"""

    # Invoke the LLM safely
    try:
        response = st.session_state.llm.invoke(prompt)
    except Exception as e:
        response = f"😔 AI failed to generate an answer. ({str(e)})"

    info.empty()
    bar.empty()
    return response


st.markdown("<h2>📘 CFAIA Assistant</h2>", unsafe_allow_html=True)


# -----------------------------
# 📖 Collapsible FAQ (auto-record user questions)
# -----------------------------
with st.expander("📖 Frequently Asked Questions", expanded=False):
    username = st.session_state["username"]
    st.session_state.setdefault("faq", {})  # ensure FAQ dict exists
    user_faq = st.session_state.faq.get(username, [])

    # Automatically add new user messages to FAQ
    for msg in st.session_state.messages:
        if msg["role"] == "user" and msg["content"] not in user_faq:
            user_faq.append(msg["content"])
    st.session_state.faq[username] = user_faq  # update session state

    # Display FAQ
    if user_faq:
        for q in reversed(user_faq):  # latest first
            st.markdown(f"- {q}")
    else:
        st.markdown("No FAQs yet. Ask a question to record it here!")


import streamlit as st

# -----------------------------
# Initialize session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = []

if "username" not in st.session_state:
    st.session_state.username = "default_user"

username = st.session_state.username

if username not in st.session_state.chat_history:
    st.session_state.chat_history[username] = []

# -----------------------------
# Display chat history
# -----------------------------

with st.expander("📂 Chat History", expanded=False):
 
    st.write("Chat history content here")

    
    user_chats = st.session_state.chat_history[username]
    all_chats = user_chats + ([st.session_state.current_chat] if st.session_state.current_chat else [])

    for i, chat_session in enumerate(reversed(all_chats)):
        real_index = len(all_chats) - 1 - i
        if chat_session:
            first_user_msg = next((m["content"] for m in chat_session if m.get("role") == "user"), None)
            session_title = first_user_msg[:50] + "..." if first_user_msg and len(first_user_msg) > 50 else first_user_msg
        else:
            session_title = "Untitled Chat"
        session_title = session_title or "Untitled Chat"

        if st.button(session_title, key=f"chat_{real_index}"):
            # Save current chat
            if st.session_state.current_chat and st.session_state.current_chat not in user_chats:
                user_chats.append(st.session_state.current_chat.copy())
            # Load selected chat
            st.session_state.current_chat = chat_session.copy()
            st.session_state.messages = chat_session.copy()
            st.rerun()
            # -----------------------------
# New Chat Button
# -----------------------------
if st.button("🆕 New Chat"):
    # Save current chat to history
    if st.session_state.current_chat and st.session_state.current_chat not in st.session_state.chat_history[username]:
        st.session_state.chat_history[username].append(st.session_state.current_chat.copy())
    # Clear current chat/messages
    st.session_state.current_chat = []
    st.session_state.messages = []
    st.rerun()
 
# -----------------------------
# 🚪 Logout Button
# -----------------------------
if st.button("🚪 Logout"):
    # Optional: save chat before logout
    if (
        st.session_state.current_chat
        and st.session_state.current_chat
        not in st.session_state.chat_history[username]
    ):
        st.session_state.chat_history[username].append(
            st.session_state.current_chat.copy()
        )

    # Clear session
    st.session_state.logged_in = False
    st.session_state.current_chat = []
    st.session_state.messages = []

    # Clear all input fields for login interface
    for key in ["login_username", "login_password", "chat_input"]:  # add all keys you use
        if key in st.session_state:
            st.session_state[key] = ""

    # Force rerun so login UI shows
    st.rerun()

# -----------------------------
# Display chat messages
# -----------------------------
for msg in st.session_state.messages:
    bubble_class = "user-msg msg" if msg["role"] == "user" else "bot-msg msg"
    st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)



# -----------------------------
# Chat input using st.form
# -----------------------------
with st.form(key="chat_form", clear_on_submit=True):
    text = st.text_input("Type your question...", placeholder="Ask me anything...")
    submitted = st.form_submit_button("Send")
    if submitted and text:
        st.session_state.messages.append({"role": "user", "content": text})
        reply = ask_question(text)  # your function to generate reply
        st.session_state.messages.append({"role": "bot", "content": reply})
        st.session_state.current_chat = st.session_state.messages.copy()
        st.rerun()



import streamlit as st

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "logout_clicked" not in st.session_state:
    st.session_state.logout_clicked = False



# -----------------------------
# LOGIN SIMULATION
# -----------------------------
if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}")




# ===========================
# Chat UI
# ===========================
st.markdown("""
<style>
/* Outer chat container */
.chat-container {
    background-color: #1e1e1e;  /* Dark background */
    color: #fff;                /* Text color */
    padding: 16px;
    border-radius: 12px;
    max-height: 500px;
    overflow-y: auto;
    font-family: Arial, sans-serif;
}

/* Remove individual message bubbles */
.chat-container .msg {
    margin-bottom: 8px;
}

/* Optional: distinguish user and bot text with subtle color */
.chat-container .user-msg {
    color: #a5d6ff;
}
.chat-container .bot-msg {
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

import streamlit as st
import time

# -----------------------------
# CONFIG
# -----------------------------
LOGOUT_SECONDS = 180  # 3 minutes

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "time_left" not in st.session_state:
    st.session_state.time_left = LOGOUT_SECONDS
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()
if "retrieval_result" not in st.session_state:
    st.session_state.retrieval_result = "This is your retrieved content!"
if "retrieval_count" not in st.session_state:
    st.session_state.retrieval_count = 0  # track how many times content retrieved

# Initialize username
if "username" not in st.session_state:
    st.session_state.username = ""

# -----------------------------
# LOGIN LOGIC
# -----------------------------
if not st.session_state.logged_in:
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.session_state.time_left = LOGOUT_SECONDS
            st.session_state.last_update = time.time()
    st.stop()

# -----------------------------
# DISPLAY RETRIEVAL AND ADD 30-SECOND BUFFER
# -----------------------------
st.write(st.session_state.retrieval_result)

# Add 30 seconds for every retrieval
st.session_state.time_left += 200
st.session_state.retrieval_count += 1  # optional: track number of queries
st.info(f"⏳ Added 30 seconds! Total time left: {st.session_state.time_left} seconds")
# TIMER PLACEHOLDER
# -----------------------------
timer_placeholder = st.empty()

# -----------------------------
# NON-BLOCKING TIMER WITH AUTO-LOGOUT
# -----------------------------
def update_timer():
    while st.session_state.logged_in and st.session_state.time_left > 0:
        current_time = time.time()
        elapsed = int(current_time - st.session_state.last_update)
        if elapsed > 0:
            st.session_state.time_left -= elapsed
            st.session_state.last_update = current_time

        minutes = st.session_state.time_left // 60
        seconds = st.session_state.time_left % 60
        timer_placeholder.markdown(f"⏳ Auto logout in: **{minutes:02d}:{seconds:02d}**")

        # -----------------------------
        # AUTO LOGOUT WHEN TIME IS UP
        # -----------------------------
        if st.session_state.time_left <= 0:
            st.session_state.logged_in = False
            st.session_state.time_left = 0
            st.rerun()

        time.sleep(1)

    if not st.session_state.logged_in:
        timer_placeholder.markdown("⏰ Time is up! Logging out...")
        st.session_state.time_left = LOGOUT_SECONDS
        st.session_state.last_update = time.time()


# Call the timer function
update_timer()

if st.session_state.time_left <= 0:
    st.session_state.logged_in = False  # log out
    st.session_state.username = ""      # reset username
    st.session_state.time_left = LOGOUT_SECONDS
    st.rerun()  # redirect to login page immediately

