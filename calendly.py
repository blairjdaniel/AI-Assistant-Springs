import requests
from app_scripts.tour_logger import log_tour_details
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

CALENDLY_WEBHOOK_ID = os.getenv("CALENDLY_WEBHOOK_ID")
CALENDLY_CLIENT_ID = os.getenv("CALENDLY_CLIENT_ID")
CALENDLY_SECRET_ID = os.getenv("CALENDLY_SECRET_ID")

def register_calendly_webhook(oauth_token, webhook_url):
    url = "https://api.calendly.com/webhook_subscriptions"
    
    # Fetch the organization URL
    org_url = get_calendly_organization(oauth_token)
    if not org_url:
        print("Failed to fetch organization URL.")
        return

    payload = {
        "url": webhook_url,
        "events": ["invitee.created", "invitee.canceled"],
        "organization": org_url  # Add the organization URL here
    }
    headers = {
        "Authorization": f"Bearer {oauth_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:
        print("Webhook registered successfully:", response.json())
    else:
        print(f"Error: {response.status_code}, {response.text}")

def get_calendly_organization(oauth_token):
    url = "https://api.calendly.com/users/me"
    headers = {
        "Authorization": f"Bearer {oauth_token}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["resource"]["organization"]
    else:
        print(f"Error fetching organization: {response.status_code}, {response.text}")
        return None

def get_calendly_oauth_token():
    url = "https://auth.calendly.com/oauth/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": CALENDLY_CLIENT_ID,
        "client_secret": CALENDLY_SECRET_ID
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        token = response.json().get("access_token")
        print("OAuth Token:", token)
        return token
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    
def fetch_calendly_events(oauth_token):
    url = "https://api.calendly.com/scheduled_events"
    headers = {
        "Authorization": f"Bearer {oauth_token}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        events = response.json().get("collection", [])
        print(f"Fetched {len(events)} events.")
        return events
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []

# Example usage
oauth_token = get_calendly_oauth_token()
if oauth_token:
    events = fetch_calendly_events(oauth_token)
    for event in events:
        print(event)
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def register_calendly_webhook(oauth_token, webhook_url):
    url = "https://api.calendly.com/webhook_subscriptions"
    payload = {
        "url": webhook_url,
        "events": ["invitee.created", "invitee.canceled"],
        "organization": "https://api.calendly.com/organizations/your_organization_id"
    }
    headers = {
        "Authorization": f"Bearer {oauth_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:
        print("Webhook registered successfully:", response.json())
    else:
        print(f"Error: {response.status_code}, {response.text}")

# Example usage
if oauth_token:
    register_calendly_webhook(oauth_token, "https://your-webhook-url.com")



def fetch_calendly_events():
    url = "https://api.calendly.com/scheduled_events"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        events = response.json()["collection"]
        return events
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []

# Example usage
events = fetch_calendly_events()
for event in events:
    invitee_name = event["invitees"][0]["name"]
    invitee_email = event["invitees"][0]["email"]
    event_date = event["start_time"]  # ISO 8601 format
    log_tour_details(invitee_name, invitee_email, event_date)