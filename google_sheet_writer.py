
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

def write_to_google_sheet(log_entries):
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        creds_dict = st.secrets["gcp_service_account"]
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        gc = gspread.authorize(credentials)

        sheet = gc.open("HealthMate_Logs").sheet1

        for entry in log_entries:
            timestamp = datetime.utcnow().isoformat()
            sheet.append_row([
                timestamp,
                entry.get("participant_id"),
                entry.get("condition"),
                entry.get("user_input"),
                entry.get("bot_reply")
            ])
    except Exception as e:
        st.error(f"Failed to write to Google Sheet: {e}")
