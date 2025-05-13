import streamlit as st
import datetime
import uuid

# User database for authentication
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "doctor": {"password": "doctor123", "role": "medical"},
    "nurse": {"password": "nurse123", "role": "medical"},
    "user": {"password": "password", "role": "user"},
    "test": {"password": "test", "role": "user"}
}

def initialize_session():
    """Initialize session state for persistence"""
    # Create a unique session ID if not exists
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Initialize session state for user management
    if 'session' not in st.session_state:
        st.session_state.session = {
            'logged_in': False,
            'username': None,
            'user_role': None,
            'login_time': None,
            'last_activity': None
        }
    
    # For backward compatibility
    if 'username' not in st.session_state:
        st.session_state.username = st.session_state.session.get('username')
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = st.session_state.session.get('logged_in', False)

def check_url_params():
    """Check URL parameters for auto-login"""
    try:
        # Check for username in URL parameters
        if 'username' in st.query_params and st.query_params['username']:
            username = st.query_params['username']
            # Auto-login from URL parameter
            login_user(username)
            return True
        
        # Check for session ID in URL parameters
        elif 'session_id' in st.query_params and st.query_params['session_id']:
            url_session_id = st.query_params['session_id']
            if 'session_id' in st.session_state and url_session_id == st.session_state.session_id:
                # Session is valid, make sure we're still logged in
                if st.session_state.session.get('username'):
                    return True
    except Exception as e:
        print(f"Error checking URL parameters: {e}")
    
    return False

def login_user(username, role=None):
    """Login a user and set session state"""
    # Set role if not provided
    if role is None:
        role = "admin" if username == "admin" else "user"
    
    # Update session state
    st.session_state.session['username'] = username
    st.session_state.session['logged_in'] = True
    st.session_state.session['user_role'] = role
    st.session_state.session['login_time'] = datetime.datetime.now()
    st.session_state.session['last_activity'] = datetime.datetime.now()
    
    # For backward compatibility
    st.session_state.username = username
    st.session_state.logged_in = True
    
    # Set URL parameters to maintain session on reload
    st.query_params['username'] = username
    st.query_params['session_id'] = st.session_state.session_id
    
    return True

def logout_user():
    """Logout a user and clear session state"""
    # Clear session state
    st.session_state.session['logged_in'] = False
    st.session_state.session['username'] = None
    
    # For backward compatibility
    st.session_state.logged_in = False
    st.session_state.username = None
    
    # Clear URL parameters
    if 'username' in st.query_params:
        del st.query_params['username']
    if 'session_id' in st.query_params:
        del st.query_params['session_id']
    
    return True

def authenticate_user(username, password):
    """Authenticate a user against the user database"""
    if username in USERS:
        # Check if USERS[username] is a dictionary (new format) or a string (old format)
        if isinstance(USERS[username], dict) and USERS[username]["password"] == password:
            return login_user(username, USERS[username]["role"])
        elif isinstance(USERS[username], str) and USERS[username] == password:
            return login_user(username)
    
    return False
