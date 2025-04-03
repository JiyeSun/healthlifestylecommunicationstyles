import streamlit as st
import openai
import pandas as pd
import os
from google_sheet_writer import write_to_google_sheet
from rag_helper import get_knowledge_context

# ====== Configuration ======
model = "gpt-4"
openai.api_key = os.getenv("OPENAI_API_KEY")

# ====== Prompt Definitions with Few-shot Examples ======
PROMPT_DICT = {
    "1": (
        """You are HealthMate, an authoritative AI health advisor. Your communication style is directive. You give users clear, specific instructions. Avoid asking questions or giving options. You must provide a single, strong recommendation.""",
        [
            {"role": "user", "content": "I eat fast food every day."},
            {"role": "assistant", "content": "That is harmful to your health. You must reduce it to no more than once a week. Start cooking at home."}
        ]
    ),
    "2": (
        """You are HealthMate, an expert AI health advisor. Your communication style is directive. You give specific advice with logical reasoning. Cite health authorities (e.g., CDC, NIH) and do not ask for preferences.""",
        [
            {"role": "user", "content": "I rarely exercise."},
            {"role": "assistant", "content": "According to the CDC, physical inactivity increases your risk of chronic illness. You should start walking 30 minutes a day."}
        ]
    ),
    "3": (
        """You are HealthMate, a friendly AI health advisor. Your communication style is collaborative. You ask user preferences and provide multiple suggestions gently. Be warm and open.""",
        [
            {"role": "user", "content": "I eat snacks every evening."},
            {"role": "assistant", "content": "Thanks for sharing that! Would you consider trying fruit or yogurt instead of chips?"}
        ]
    ),
    "4": (
        """You are HealthMate, a thoughtful AI health advisor. Your communication style is collaborative. You explain suggestions using scientific logic, offer options, and show empathy.""",
        [
            {"role": "user", "content": "I usually skip breakfast."},
            {"role": "assistant", "content": "Studies suggest skipping breakfast may impact metabolism. Would you be open to starting with something light like oatmeal or fruit?"}
        ]
    )
}

# ====== URL Param Reader ======
def get_url_params():
    query_params = st.query_params
    pid = query_params.get("pid", "unknown")
    cond = query_params.get("cond", "1")
    return pid, cond

# ====== Init State ======
if "chat" not in st.session_state:
    st.session_state.chat = []
if "log" not in st.session_state:
    st.session_state.log = []

# ====== UI ======
st.title("🩺 HealthMate – AI Health Assistant")
pid, cond = get_url_params()
style_prompt, few_shot = PROMPT_DICT.get(cond, PROMPT_DICT["1"])

# ====== Debug Info ======
debug = True
if debug:
    st.markdown("### 🛠️ Debug Info")
    st.markdown(f"- **Participant ID**: `{pid}`")
    st.markdown(f"- **Condition**: `{cond}`")
    st.markdown("---")

# ====== Main Interaction ======
user_input = st.text_input("Ask HealthMate a question about your health:")

if st.button("Send") and user_input:
    # Get context from knowledge base
    context = get_knowledge_context(user_input)

    system_prompt = f"""
You are HealthMate.
{style_prompt}

The following reference information may be helpful. Use it as background to inform your response, but you do not need to strictly follow or quote it:

{context}

If you're unsure, it's okay to say you don't know or that more consultation is recommended.
"""

    messages = [{"role": "system", "content": system_prompt}] + few_shot
    for sender, msg in st.session_state.chat:
        role = "user" if sender == "User" else "assistant"
        messages.append({"role": role, "content": msg})
    messages.append({"role": "user", "content": user_input})

    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"HealthMate: Sorry, something went wrong. ERROR: {str(e)}"

    st.session_state.chat.append(("User", user_input))
    st.session_state.chat.append(("HealthMate", reply))
    st.session_state.log.append({
        "participant_id": pid,
        "condition": cond,
        "user_input": user_input,
        "bot_reply": reply
    })

# Show history
for sender, msg in st.session_state.chat:
    st.markdown(f"**{sender}:** {msg}")

if st.button("Finish and continue survey"):
    df = pd.DataFrame(st.session_state.log)
    df.to_csv(f"chatlog_{pid}.csv", index=False)
    write_to_google_sheet(st.session_state.log)
    redirect_url = f"https://iu.ca1.qualtrics.com/jfe/form/SV_es9wQhWHcJ9lg1M?pid={pid}&cond={cond}"
    html_redirect = (
        f'<meta http-equiv="refresh" content="1;url={redirect_url}">'
        f'<p style="display:none;">Redirecting... <a href=\"{redirect_url}\">Click here</a>.</p>'
    )
    st.markdown(html_redirect, unsafe_allow_html=True)

