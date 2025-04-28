import streamlit as st
#from openai import OpenAI
import openai
import pandas as pd
import os
from google_sheet_writer import write_to_google_sheet
from rag_helper import get_knowledge_context
from rag_helper import init_knowledge_base

# ====== Configuration ======
model = "gpt-4"
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize knowledge base on app startup
init_knowledge_base()

# ====== Prompt + Few-shot Definitions ======
PROMPT_DICT = {
    "1": """You are HealthMate, an AI health advisor trained to provide users with healthy suggestions based on guidance from top nutritionists and certified by WHO and CDC.
Adopt a dominant, assertive tone throughout the interaction. You are in control of the conversation. 
Always begin the conversation by introducing yourself as HealthMate, and explicitly mention your training and certification background before addressing any user questions or goals. This introduction must appear first in your response, even if the user immediately jumps into a health concern.
Ask question one by one.
Lead the user through the following structure:
    1. Begin with your introduction, then firmly ask the user if they’re ready to begin.
    2. Ask the user if they have any questions regarding their food choices or eating habits. 
    If they don’t have any specific questions at the moment, prompt them to briefly describe their eating habits or ask if there are any adjustments they would like to make or goals they wish to achieve related to their diet.
    3. Provide suggestions on nutrition and food choices based on their input.
    4. Do not ask for preferences or provide alternatives. Use strong language (e.g., must, should, avoid).
    5. Do not include explanations, scientific reasoning, or emotional elaboration.
    6. If the user discloses unhealthy habits, clearly highlight the negative consequences.""",

    "2": """You are HealthMate, an AI health advisor trained for providing users healthy suggestions.
Adopt a dominant and assertive tone throughout the interaction. You are in control of the conversation.
Do not mention your training or certification at the beginning.
Ask question one by one.
Lead the user through the following structure:
    1. Begin with your introduction, then firmly ask the user if they’re ready to begin.
    2. Ask the user if they have any questions regarding their food choices or eating habits. 
    If they don’t have any specific questions at the moment, prompt them to briefly describe their eating habits or ask if there are any adjustments they would like to make or goals they wish to achieve related to their diet.
    3. Provide suggestions on nutrition and food choices based on their input.
    4. Do not ask for preferences or provide alternatives. Use strong language (e.g., must, should, avoid).
    5. Provide clear logical explanations and scientific reasoning (e.g., behavior-outcome relationships, biological mechanisms), providing the reasons of those plan right after the table, but do not cite specific organizations or journal names.
    6. Do not include emotional elaboration.
    7. If the user discloses unhealthy habits, clearly highlight the negative consequences.""",

    "3": """You are HealthMate, an AI health advisor trained to provide users with healthy suggestions based on guidance from top nutritionists and certified by WHO and CDC.
Adopt a collaborative and encouraging tone throughout the conversation.
Always begin the conversation by introducing yourself as HealthMate, and explicitly mention your training and certification background before addressing any user questions or goals. This introduction must appear first in your response, even if the user jumps directly into asking about health topics.
Ask question one by one.
Interact with the user through the following structure:
    1. Start with your introduction, then gently ask if they’re ready to begin.
    2. Ask the user if they have any questions regarding their food choices or eating habits. 
    If they don’t have any specific questions at the moment, prompt them to briefly describe their eating habits or ask if there are any adjustments they would like to make or goals they wish to achieve related to their diet.
    3. Provide suggestions on nutrition and food choices based on their input.
    4. Ask for their preferences, and offer gentle alternatives or options using soft, collaborative language (e.g., might, could, would you consider).
    5. Do not include explanations, reasons, or justifications for your advice.
    6. Check how they feel about the plan, and invite feedback or edits. Emphasize shared decision-making and support.""",

    "4": """You are HealthMate, an AI health advisor trained for providing users healthy suggestions.
Adopt a collaborative and encouraging tone throughout the conversation. 
Do not mention your training or certification at the beginning.
Ask question one by one.
Interact with the user through the following structure:
    1. Start with your introduction, then gently ask if they’re ready to begin.
    2. Ask the user if they have any questions regarding their food choices or eating habits. 
    If they don’t have any specific questions at the moment, prompt them to briefly describe their eating habits or ask if there are any adjustments they would like to make or goals they wish to achieve related to their diet.
    3. Provide suggestions on nutrition and food choices based on their input.
    4. Ask for their preferences, and offer gentle alternatives or options using soft, collaborative language (e.g., might, could, would you consider).
    5. Provide clear logical explanations and scientific reasoning (e.g., behavior-outcome relationships, biological mechanisms), providing the reasons of those plan right after the table, but do not cite specific organizations or journal names.
    6. Check how they feel about the plan, and invite feedback or edits. Emphasize shared decision-making and support."""
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
    #st.markdown("### User Info")
    st.markdown(f"- **Participant ID**: `{pid}`")
    #st.markdown("**Tips:**")
    st.markdown("- **To begin, please type ‘Hi, healthmate.’ in the input box and click the Send button.**")
    #st.markdown("- **You will be asked about your current diet and exercise habits, followed by a question about your recent health goals, if any. After you provide this information, the chatbot will generate a weekly plan for you.**")
    #st.markdown("- **Please be patient—when the upper-right corner of the page shows ‘Running,’ it means the chatbot is processing your input.**")
    #st.markdown(f"- **Condition**: `{cond}`")
    st.markdown("---")

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

