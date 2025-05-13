import streamlit as st
import datetime
import uuid
from api_client import login_user as api_login, logout_user as api_logout, get_user_details

def initialize_session():
    """Initialize the session state with default values"""
    # Initialize session state for user management
    if 'session' not in st.session_state:
        st.session_state.session = {
            'logged_in': False,
            'username': None,
            'user_role': None,
            'login_time': None,
            'last_activity': None
        }

    # Create a session ID to track sessions
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        print(f"Created new session ID: {st.session_state.session_id}")

    # For backward compatibility, ensure these are always set
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Initialize user preferences
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {
            'theme': 'light',
            'save_predictions': True,
            'show_advanced_options': False,
            'default_model': 'diabetes'
        }

    # Initialize user activity history
    if 'activity_history' not in st.session_state:
        st.session_state.activity_history = []

    # Initialize login-related session state variables
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0

def check_url_params():
    """Check URL parameters for session information"""
    # Check for existing session in cookies
    if 'username' in st.query_params and st.query_params['username']:
        username = st.query_params['username']
        # Auto-login from URL parameter
        st.session_state.session['username'] = username
        st.session_state.session['logged_in'] = True
        st.session_state.session['user_role'] = "admin" if username == "admin" else "user"
        st.session_state.username = username
        st.session_state.logged_in = True
        print(f"Auto-logged in as {username} from URL parameter")

        # Set session ID in URL to maintain session
        st.query_params['session_id'] = st.session_state.session_id

        # Set login time if not already set
        if not st.session_state.session.get('login_time'):
            st.session_state.session['login_time'] = datetime.datetime.now()

        # Always update last activity time
        st.session_state.session['last_activity'] = datetime.datetime.now()

    elif 'session_id' in st.query_params and st.query_params['session_id']:
        # We have a session ID in the URL, check if it matches our current session
        url_session_id = st.query_params['session_id']
        if 'session_id' in st.session_state and url_session_id == st.session_state.session_id:
            print(f"Restored session from URL parameter: {url_session_id}")
            # Session is valid, make sure we're still logged in
            if st.session_state.session.get('username'):
                print(f"Maintaining login for user: {st.session_state.session['username']}")

                # Ensure username is also in st.session_state for backward compatibility
                st.session_state.username = st.session_state.session['username']
                st.session_state.logged_in = True

                # Always update last activity time
                st.session_state.session['last_activity'] = datetime.datetime.now()
        else:
            # Session ID doesn't match, create a new one
            st.session_state.session_id = url_session_id
            print(f"Adopted session ID from URL: {url_session_id}")

def login_user(username, password, users_db=None):
    """Authenticate and log in a user using the API"""
    # Try to authenticate with the API first
    success, response = api_login(username, password)

    if success:
        # Set session state
        st.session_state.session['logged_in'] = True
        st.session_state.session['username'] = username
        st.session_state.session['user_role'] = response.get('role', 'user')
        st.session_state.session['login_time'] = datetime.datetime.now()
        st.session_state.session['last_activity'] = datetime.datetime.now()
        st.session_state.session['token'] = response.get('token')
        st.session_state.session['user_id'] = response.get('user_id')

        # For backward compatibility
        st.session_state.username = username
        st.session_state.logged_in = True

        # Set URL parameter to maintain session across page refreshes
        st.query_params['username'] = username
        st.query_params['session_id'] = st.session_state.session_id

        return True

    # Fallback to local authentication if API fails and users_db is provided
    elif users_db and username in users_db and users_db[username]["password"] == password:
        # Set session state
        st.session_state.session['logged_in'] = True
        st.session_state.session['username'] = username
        st.session_state.session['user_role'] = users_db[username]["role"]
        st.session_state.session['login_time'] = datetime.datetime.now()
        st.session_state.session['last_activity'] = datetime.datetime.now()

        # For backward compatibility
        st.session_state.username = username
        st.session_state.logged_in = True

        # Set URL parameter to maintain session across page refreshes
        st.query_params['username'] = username
        st.query_params['session_id'] = st.session_state.session_id

        return True

    return False

def logout_user():
    """Log out the current user"""
    # Try to logout via API if we have a token
    if 'token' in st.session_state.session and st.session_state.session['token']:
        api_logout(st.session_state.session['token'])

    # Clear session state
    st.session_state.session['logged_in'] = False
    st.session_state.session['username'] = None
    st.session_state.session['user_role'] = None
    st.session_state.session['token'] = None
    st.session_state.session['user_id'] = None

    # For backward compatibility
    st.session_state.username = None
    st.session_state.logged_in = False

    # Clear URL parameters
    if 'username' in st.query_params:
        del st.query_params['username']
    if 'session_id' in st.query_params:
        del st.query_params['session_id']

def authenticate_user(username, password, users_db=None):
    """Authenticate a user against the API or local database"""
    # Try to authenticate with the API first
    success, response = api_login(username, password)

    if success:
        return True, response.get('role', 'user')

    # Fallback to local authentication if API fails and users_db is provided
    elif users_db and username in users_db and users_db[username]["password"] == password:
        return True, users_db[username]["role"]

    return False, None

def update_user_activity(activity_type, details=None):
    """Record user activity in the session state"""
    if st.session_state.session['logged_in']:
        timestamp = datetime.datetime.now()
        st.session_state.session['last_activity'] = timestamp

        activity = {
            'timestamp': timestamp,
            'username': st.session_state.session['username'],
            'activity_type': activity_type,
            'details': details
        }

        # Add to activity history (limit to last 50 activities)
        st.session_state.activity_history.append(activity)
        if len(st.session_state.activity_history) > 50:
            st.session_state.activity_history.pop(0)
