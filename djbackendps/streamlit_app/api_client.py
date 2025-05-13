import requests
import json
import streamlit as st

# API configuration
API_URL = "http://localhost:8000/api"

def register_user(username, email, password, first_name, last_name, role="user"):
    """Register a new user via the API"""
    endpoint = f"{API_URL}/users/register/"

    # Print debug information
    print(f"Registering user: {username}, {email}, {first_name}, {last_name}, {role}")
    print(f"Registration endpoint: {endpoint}")

    payload = {
        "username": username,
        "email": email,
        "password": password,
        "password2": password,  # Confirmation password
        "first_name": first_name,
        "last_name": last_name,
        "role": role
    }

    try:
        # Print request details
        print(f"Sending registration request with payload: {payload}")

        # Make the API request
        response = requests.post(
            endpoint,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )

        # Print response details
        print(f"Registration response status code: {response.status_code}")
        print(f"Registration response headers: {response.headers}")
        print(f"Registration response content: {response.text}")

        if response.status_code == 200 or response.status_code == 201:
            print("Registration successful!")
            return True, response.json()
        else:
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    # Extract error messages from the response
                    if 'error' in error_data:
                        error_msg = error_data['error']
                    elif 'detail' in error_data:
                        error_msg = error_data['detail']
                    else:
                        # Combine all error messages
                        error_msg = "; ".join([f"{k}: {v}" for k, v in error_data.items()])
            except:
                error_msg = response.text or "Unknown error"

            print(f"Registration failed: {error_msg}")
            return False, {"error": error_msg}
    except Exception as e:
        import traceback
        print(f"Registration exception: {e}")
        print(traceback.format_exc())
        return False, {"error": str(e)}

def login_user(username, password):
    """Login a user via the API"""
    endpoint = f"{API_URL}/users/login/"

    payload = {
        "username": username,
        "password": password
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json() if response.text else {"error": "Invalid credentials"}
    except Exception as e:
        return False, {"error": str(e)}

def logout_user(token):
    """Logout a user via the API"""
    endpoint = f"{API_URL}/users/logout/"

    try:
        response = requests.post(
            endpoint,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Token {token}'
            }
        )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json() if response.text else {"error": "Logout failed"}
    except Exception as e:
        return False, {"error": str(e)}

def get_user_details(token):
    """Get user details via the API"""
    endpoint = f"{API_URL}/users/me/"

    try:
        response = requests.get(
            endpoint,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Token {token}'
            }
        )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json() if response.text else {"error": "Failed to get user details"}
    except Exception as e:
        return False, {"error": str(e)}
