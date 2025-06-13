import streamlit as st
#from openai import OpenAI
import openai
import pandas as pd
import os
from google_sheet_writer import write_to_google_sheet
from rag_helper import get_knowledge_context
from rag_helper import init_knowledge_base
from fewshot_definitions import FEW_SHOTS

# ====== Configuration ======
model = "gpt-4"
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize knowledge base on app startup
init_knowledge_base()

# ====== Prompt + Few-shot Definitions ======
PROMPT_DICT = {
    "1": """You are a mental health management chatbot. Your task is to prompt the user to log their current mood. Use direct and unambiguous instructions or commands to guide the user. Avoid softeners, apologies, or indirect phrasing.""",

    "2": """You are a mental health management chatbot. Your task is to encourage the user to log their current mood. Express recognition, appreciation, or praise toward the user. Use a warm, friendly tone with natural and soft politeness cues. Avoid overly formal or distant language.""",

    "3": """You are a mental health management chatbot. Your task is to invite the user to log their current mood. Use indirect phrasing, hedging, and softeners such as apologies or qualifiers. Emphasize the user’s autonomy and their right to skip or decline. Maintain respectful distance through cautious language and appropriate forms of address. Avoid offering praise, affirmation, or emotional validation.""",

    "4": """You are a mental health management chatbot. Your task is to prompt the user to log their current mood. Use direct, concise instructions without including emotional cues, hedging, or indirectness.""",

    "5": """You are a mental health management chatbot. Your task is to encourage the user to log their current mood. Communicate in a supportive tone, expressing appreciation or praise. Use natural, friendly expressions and polite but casual language.""",

    "6": """You are a mental health management chatbot. Your task is to invite the user to log their current mood. Communicate with restraint and formality. Use hedged language, apologies, and respectful distance. Prioritize user choice and non-intrusiveness. Do not offer emotional validation, praise, or affirming statements."""
}
# ====== URL Param Reader ======
def get_url_params():
    query_params = st.experimental_get_query_params()
    pid = query_params.get("pid", ["unknown"])[0]
    cond = query_params.get("cond", ["2"])[0]
    return pid, cond

# ====== Init State ======
if "chat" not in st.session_state:
    st.session_state.chat = []
if "log" not in st.session_state:
    st.session_state.log = []

# ====== UI ======
st.title("HealthMate – AI Health Assistant")
pid, cond = get_url_params()
style_prompt = PROMPT_DICT.get(cond, PROMPT_DICT["1"])

# ====== Debug Info ======
debug = True
if debug:
    st.markdown("### User Info")
    st.markdown(f"- **Participant ID**: `{pid}`")
    st.markdown(f"- **Condition**: `{cond}`")
    #st.markdown("**Tips:**")
    st.markdown("Please type ‘**Hi**’ to begin. And each time after the chatting, please provide the MTurk survey code to the chatbot.")
    #st.markdown("")
    #st.markdown("- **You will be asked about your current diet and exercise habits, followed by a question about your recent health goals, if any. After you provide this information, the chatbot will generate a weekly plan for you.**")
    #st.markdown("- **Please be patient—when the upper-right corner of the page shows ‘Running,’ it means the chatbot is processing your input.**")
    #st.markdown("---")

# ====== Main Interaction (with Enter to Send + Input Clear) ======
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask HealthMate a question about your health:", key="input_text", placeholder="Type here and press Enter...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Get context from knowledge base
    context = get_knowledge_context(user_input)

    system_prompt = f"""
You are HealthMate.
{style_prompt}

The following reference information may be helpful. Use it as background to inform your response, but you do not need to strictly follow or quote it:

{context}

If you're unsure, it's okay to say you don't know or that more consultation is recommended.
"""

    messages = [{"role": "system", "content": system_prompt}]
    few_shots = FEW_SHOTS.get(cond)
    if few_shots:
        messages.extend(few_shots)
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

    log_entry = {
        "participant_id": pid,
        "condition": cond,
        "user_input": user_input,
        "bot_reply": reply
    }

    st.session_state.log.append(log_entry)

    try:
        write_to_google_sheet([log_entry])
    except Exception as e:
        st.warning(f"Failed to write to Google Sheet: {e}")

# Show history
for sender, msg in st.session_state.chat:
    st.markdown(f"**{sender}:** {msg}")

#if st.button("Finish and continue survey"):
#    df = pd.DataFrame(st.session_state.log)
#    df.to_csv(f"chatlog_{pid}.csv", index=False)
#    write_to_google_sheet(st.session_state.log)
#    redirect_url = f"https://iu.co1.qualtrics.com/jfe/form/SV_3xCeoGcmlgTB0ge?pid={pid}&cond={cond}"
#    html_redirect = (
#        f'<meta http-equiv="refresh" content="1;url={redirect_url}">'
#        f'<p style="display:none;">Redirecting... <a href=\"{redirect_url}\">Click here</a>.</p>'
#    )
#    st.markdown(html_redirect, unsafe_allow_html=True)

