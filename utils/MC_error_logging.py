import requests
import streamlit as st
import json

def send_teams_message(message):
    endpoint = st.secrets["TEAMS_ENDPOINT"]
    payload = {
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": message
                        }
                    ],
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "version": "1.0"
                }
            }
        ]
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(endpoint, data=json.dumps(payload), headers=headers)

    if response.status_code == 202:
        print("Message sent successfully")
    else:
        print(f"Failed to send message. Status code: {response.status_code}")
        print(f"Error message: {response.text}")