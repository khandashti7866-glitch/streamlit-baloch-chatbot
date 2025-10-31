"""
Streamlit ChatGPT-like AI Chatbot Application
- Modern chat-style UI using st.chat_message()
- Persistent session history stored in st.session_state["messages"]
- Three backend options:
    1) OpenAI (if OPENAI_API_KEY provided)
    2) Local transformers (if installed) using distilgpt2 (auto-download)
    3) Rule-based fallback (always available, no internet required)
- No paid API key required to run; the rule-based backend guarantees the app runs "out of the box".
"""

import os
import re
import time
from typing import List, Dict

import streamlit as st

# The prompt required "from openai import OpenAI" so we import in a try/except manner.
try:
    from openai import OpenAI  # type: ignore
    OPENAI_IMPORTED = True
except Exception:
    OPENAI_IMPORTED = False

# Attempt to import transformers pipeline for an offline-ish local model.
try:
    from transformers import pipeline, set_seed
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ---------------------------
# Helper / Backend functions
# ---------------------------

def init_session():
    """Initialize session state for messages and settings."""
    if "messages" not in st.session_state:
        # Messages are dicts: {"role": "user"|"assistant"|"system", "content": str}
        st.session_state["messages"] = []
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = "General"
    if "transformer_generator" not in st.session_state:
        st.session_state["transformer_generator"] = None


def append_message(role: str, content: str):
    """Append a message to the conversation history."""
    st.session_state["messages"].append({"role": role, "content": content})


def clear_chat():
    st.session_state["messages"] = []
    append_message("system", "You are a helpful AI assistant. (Session restarted.)")


# --------------
# Prompt helpers
# --------------
def build_system_prompt(model_choice: str) -> str:
    """Return a concise system-style prompt depending on model choice."""
    base = "You are a helpful, concise, and friendly AI assistant. Answer clearly and simply."
    if model_choice == "Coding":
        return base + " When asked to produce code, include only the code block unless user asks otherwise. Use Python by default. Explain complexity when appropriate."
    elif model_choice == "Education":
        return base + " Explain topics at a learner-friendly level. Use examples and short analogies."
    else:
        return base + " Be general-purpose, adaptable, and clarify ambiguous queries when needed."


# -------------------
# OpenAI backend code
# -------------------
def openai_respond(messages: List[Dict[str, str]], system_prompt: str) -> str:
    """
    Use OpenAI client if available and environment key is set.
    This function will not raise if OpenAI client isn't available; we check outside.
    """
    if not OPENAI_IMPORTED:
        raise RuntimeError("OpenAI library not available in this environment.")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_RAW")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")
    # Use the official OpenAI Python SDK (imported as from openai import OpenAI)
    client = OpenAI(api_key=api_key)
    # Build a flattened prompt with system + conversation for backends that expect text.
    # Prefer chat completions when available; depending on SDK, adapt.
    # We'll attempt a very simple chat-like usage; if fails, bubble up.
    # Note: This block is optional; not used unless user provides key.
    content_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        content_messages.append({"role": m["role"], "content": m["content"]})
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=content_messages, max_tokens=512)
        text = resp.choices[0].message.content
        return text
    except Exception:
        # Try a fallback model name
        resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=content_messages, max_tokens=512)
        text = resp.choices[0].message.content
        return text


# ---------------------------
# Transformers (local) backend
# ---------------------------
def get_transformer_generator():
    """Initialize or return a cached transformers text-generation pipeline."""
    if st.session_state.get("transformer_generator"):
        return st.session_state["transformer_generator"]
    # Use a small, fast model that is commonly available: distilgpt2
    # The model will be auto-downloaded on first run if transformers & internet available.
    try:
        gen = pipeline("text-generation", model="distilgpt2", device=-1)  # CPU default
        set_seed(42)
        st.session_state["transformer_generator"] = gen
        return gen
    except Exception as e:
        st.session_state["transformer_generator"] = None
        raise RuntimeError(f"Transformers generation init failed: {e}")


def transformers_respond(messages: List[Dict[str, str]], system_prompt: str, max_new_tokens=200) -> str:
    """
    Create a prompt for the local generator by concatenating recent messages.
    This is a simplistic approach but works well enough for a friendly demo.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers not installed in environment.")
    gen = get_transformer_generator()
    if gen is None:
        raise RuntimeError("Transformer generator unavailable.")
    # Build concatenated conversation—limit to last N messages to keep prompt size reasonable
    relevant = messages[-8:]
    prompt_parts = [f"System: {system_prompt}", ""]
    for m in relevant:
        role = m["role"].capitalize()
        # sanitize newlines
        content = m["content"].strip()
        prompt_parts.append(f"{role}: {content}")
    prompt_parts.append("Assistant: ")
    prompt_text = "\n".join(prompt_parts)
    # Use generator
    out = gen(prompt_text, max_new_tokens=max_new_tokens, do_sample=True, top_k=50, top_p=0.92, temperature=0.7)
    raw = out[0]["generated_text"]
    # Extract the assistant's completion portion only
    # We look for "Assistant:" and return text after the last occurrence
    if "Assistant:" in raw:
        assistant_part = raw.split("Assistant:")[-1].strip()
    else:
        # Fallback to returning the whole raw minus the prompt
        assistant_part = raw[len(prompt_text):].strip()
    # Post-process: shorten if too long and tidy up
    assistant_part = re.split(r"\nSystem:|\nUser:|\nAssistant:|\n[^\S\r\n]{0,}\Z", assistant_part)[0].strip()
    # Keep response to a reasonable length
    return assistant_part[:1500]


# ----------------------
# Rule-based fallback AI
# ----------------------
def rule_based_respond(user_text: str, model_choice: str) -> str:
    """
    A deterministic, useful fallback: a mixture of pattern responses, templates, and
    short generated explanations so the app is functional offline with no heavy deps.
    """
    text = user_text.strip().lower()

    # Greetings
    if re.search(r"\b(hi|hello|hey|assalam|salam|good morning|good evening)\b", text):
        return "Hello! 👋 How can I help you today? Ask me anything — explanations, code examples, or creative writing."

    # Farewells
    if re.search(r"\b(bye|goodbye|see you|take care)\b", text):
        return "Goodbye! If you want to continue, just type another message — I'm here 24/7."

    # Ask for definition
    m = re.search(r"(what is|define|meaning of) (.+)", text)
    if m:
        topic = m.group(2).strip(" ?.")
        return f"**Definition (simple):** {topic.capitalize()} — here's a short explanation: {topic} is a topic that typically refers to ... (short, friendly explanation). If you'd like a deeper explanation, tell me your current understanding or ask for an example."

    # Simple math (safe)
    if re.match(r"^[0-9\.\s\+\-\*\/\(\)]+$", text):
        try:
            # Evaluate safely by allowing digits and operators only
            result = eval(text, {"__builtins__": {}}, {})
            return f"The result is: **{result}**"
        except Exception:
            return "I couldn't compute that expression — please make sure it's a valid arithmetic expression."

    # Ask for code
    if "write code" in text or text.startswith("implement") or "function" in text and "in python" in text:
        return (
            "Here's a short Python example:\n\n"
            "```python\n"
            "def greet(name):\n"
            "    return f\"Hello, {name}!\"\n\n"
            "print(greet('World'))\n"
            "```\n\n"
            "Tell me if you want a different language or a fuller solution."
        )

    # Explain like I'm 5
    if "explain" in text and ("like i'm 5" in text or "simple terms" in text or "simple" in text):
        return "In very simple terms: think of it like this — [simple analogy]. If you give me the topic I will make a tailored analogy."

    # Ask for a summary
    if text.startswith("summarize") or "summarize" in text:
        return "Sure — paste the text you'd like summarized, or give me a link or topic and I'll produce a short summary (2-4 sentences)."

    # Default friendly fallback
    if model_choice == "Coding":
        return "I can help with coding: tell me the language and the problem. For example, 'Write a Python function that reverses a string.'"
    elif model_choice == "Education":
        return "I can explain topics step-by-step. Tell me the topic and your current level (beginner/intermediate/advanced)."
    else:
        # General
        return (
            "I'm a helpful assistant running in offline mode. I can answer factual questions, explain concepts, "
            "help write messages, and show code examples. Try asking: 'Explain black holes in simple terms' or "
            "'Write an email apologizing for a late reply'."
        )


# -----------------------
# Main Streamlit App UI
# -----------------------
st.set_page_config(page_title="AI Chat Assistant (Offline-ready)", page_icon="🤖", layout="wide")

# Add some custom CSS to make chat look nicer and high-res friendly
st.markdown(
    """
    <style>
    /* Make the chat area narrower on large screens for easier reading */
    .stApp .block-container {
        max-width: 1600px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    /* High-quality (4k-friendly) header */
    .header {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .brand-title {
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .muted {
        color: #6b7280;
        font-size: 14px;
    }
    /* Tweak chat bubble spacing */
    .stChatMessage {
        padding: 0.6rem 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_session()

# Sidebar
with st.sidebar:
    st.title("AI Chat Assistant")
    st.write("A local, offline-capable ChatGPT-like assistant built with Streamlit.")
    model_choice = st.selectbox("Model type / Persona", ["General", "Coding", "Education"], index=0)
    st.session_state["model_choice"] = model_choice
    st.write("---")
    st.write("Backends available:")
    st.write(
        "- OpenAI (if env variable OPENAI_API_KEY is set)\n"
        f"- Local transformers ({'installed' if TRANSFORMERS_AVAILABLE else 'not installed'})\n"
        "- Rule-based fallback (always available)"
    )
    if st.button("Clear chat"):
        clear_chat()
        st.experimental_rerun()
    st.write("---")
    st.caption("No paid API key required — the app falls back to a built-in responder if needed.")


# Header area
cols = st.columns([0.08, 0.92])
with cols[0]:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=64)
with cols[1]:
    st.markdown("<div class='header'><div class='brand-title'>AI Chat Assistant</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Chat naturally — built to work offline or with OpenAI if you prefer.</div>", unsafe_allow_html=True)

st.write("---")

# Conversation display
chat_container = st.container()

# Ensure at least a system message exists
if not st.session_state["messages"]:
    append_message("system", build_system_prompt(st.session_state["model_choice"]))

# Render messages
with chat_container:
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        # streamlit's chat_message expects roles 'user' or 'assistant' or 'system'
        if role == "system":
            # show as a muted informational block
            st.info(content)
        else:
            with st.chat_message(role):
                # Keep original formatting
                st.markdown(content)

# Input area
user_input = st.chat_input(placeholder="Type a question, request code, or describe what you want...")

if user_input is not None:
    user_text = user_input.strip()
    if user_text == "":
        # empty input handling
        with st.chat_message("assistant"):
            st.write("Please type something — I can't respond to empty messages.")
    else:
        # Append user message and immediately show it
        append_message("user", user_text)
        with st.chat_message("user"):
            st.write(user_text)

        # Prepare system prompt for the chosen persona
        system_prompt = build_system_prompt(st.session_state["model_choice"])

        # Generate assistant response using available backends (priority: OpenAI -> transformers -> rule-based)
        assistant_text = None
        backend_used = None

        # Show a temporary "thinking" assistant bubble to improve UX
        thinking = st.empty()
        with thinking.container():
            with st.chat_message("assistant"):
                st.markdown("_Thinking..._")

        try:
            # Try OpenAI backend if imported and key present
            if OPENAI_IMPORTED and (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_RAW")):
                try:
                    assistant_text = openai_respond(st.session_state["messages"], system_prompt)
                    backend_used = "OpenAI"
                except Exception as e:
                    # fallback to others
                    assistant_text = None

            # If OpenAI not used, try transformers
            if assistant_text is None and TRANSFORMERS_AVAILABLE:
                try:
                    assistant_text = transformers_respond(st.session_state["messages"], system_prompt)
                    backend_used = "Transformers (distilgpt2)"
                except Exception:
                    assistant_text = None

            # Final fallback: rule-based
            if assistant_text is None:
                assistant_text = rule_based_respond(user_text, st.session_state["model_choice"])
                backend_used = "Rule-based fallback"

        except Exception as e:
            # Unexpected error -- show friendly message and fallback to rule-based
            assistant_text = (
                "Sorry — an unexpected error occurred while generating a response. "
                "Falling back to a safe offline reply: " + rule_based_respond(user_text, st.session_state["model_choice"])
            )
            backend_used = "Exception fallback"

        # Replace the thinking bubble with the assistant message
        thinking.empty()
        append_message("assistant", assistant_text)
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
        # tiny footer showing backend used (subtle)
        st.caption(f"Response generated by: {backend_used}")

# Footer tips
st.write("---")
st.markdown(
    """
    **Tips**
    - For better (model-like) responses without an API key, install `transformers` and `torch` (`pip install -r requirements.txt`) — the app will download a small model (distilgpt2) automatically on first run.
    - To use the OpenAI backend (optional), set environment variable `OPENAI_API_KEY` and the OpenAI client will be used automatically.
    - The app keeps chat history during the Streamlit session. Use "Clear chat" in the sidebar to reset.
    """
)
# End code entry point
if __name__ == "__main__":
    # main() is not strictly necessary for Streamlit, but included per specification
    # We simply rely on the top-level code to run in Streamlit. For completeness:
    pass
