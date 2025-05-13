import streamlit as st
from api_client import register_user

def display_login_form():
    """Display an enhanced login form in the sidebar"""

    # Add a professional login header
    st.markdown("""
    <div style="text-align:center; margin-bottom:15px;">
        <h3 style="color:#1E88E5;">User Login</h3>
        <p style="font-size:0.9em; color:#555;">Please enter your credentials</p>
    </div>
    """, unsafe_allow_html=True)

    # Create the login form with improved styling
    with st.form("login_form"):
        # Add icon to username field
        st.markdown('<p style="margin-bottom:5px;"><strong>👤 Username</strong></p>', unsafe_allow_html=True)
        username = st.text_input("", placeholder="Enter your username", label_visibility="collapsed")

        # Add icon to password field
        st.markdown('<p style="margin-bottom:5px;"><strong>🔒 Password</strong></p>', unsafe_allow_html=True)
        password = st.text_input("", type="password", placeholder="Enter your password", label_visibility="collapsed")

        # Add some space
        st.write("")

        # Create a more attractive submit button
        submit = st.form_submit_button("Sign In", use_container_width=True)

        return username, password, submit

def display_registration_form():
    """Display the registration form"""
    try:
        # Add debug information
        print("Displaying registration form...")

        # Add a professional registration header
        st.markdown("""
        <div style="text-align:center; margin-bottom:15px;">
            <h3 style="color:#4CAF50;">Create Your Account</h3>
            <p style="font-size:0.9em; color:#555;">Join our health prediction platform</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("registration_form"):
            # Personal information section
            st.markdown('<p style="color:#4CAF50; font-weight:bold; margin-bottom:10px;">👤 Personal Information</p>', unsafe_allow_html=True)

            # Basic information in two columns
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name", placeholder="Enter your first name")
                username = st.text_input("Username", placeholder="Choose a username")

            with col2:
                last_name = st.text_input("Last Name", placeholder="Enter your last name")
                email = st.text_input("Email", placeholder="Enter your email address")

            # Security section
            st.markdown('<p style="color:#4CAF50; font-weight:bold; margin-top:15px; margin-bottom:10px;">🔒 Security</p>', unsafe_allow_html=True)

            # Password fields in two columns
            pwd_col1, pwd_col2 = st.columns(2)
            with pwd_col1:
                password = st.text_input("Password", type="password", placeholder="Create a strong password")
            with pwd_col2:
                password2 = st.text_input("Confirm Password", type="password", placeholder="Repeat your password")

            # Role selection (default to user)
            role = "user"

            # Add some space
            st.write("")

            # Terms and conditions with better styling
            st.markdown('<div style="background-color:#f8f9fa; padding:10px; border-radius:5px;">', unsafe_allow_html=True)
            terms = st.checkbox("I agree to the Terms and Conditions and Privacy Policy")
            st.markdown('</div>', unsafe_allow_html=True)

            # Add some space
            st.write("")

            # Submit button with better styling
            submit = st.form_submit_button("Create Account", use_container_width=True)

            if submit:
                # Add debug information
                print(f"Registration form submitted with: {username}, {email}, {first_name}, {last_name}")

                # Validate inputs
                if not (first_name and last_name and username and email and password):
                    st.error("Please fill in all required fields.")
                    return False, {"error": "Please fill in all required fields."}

                if password != password2:
                    st.error("Passwords do not match.")
                    return False, {"error": "Passwords do not match."}

                if not terms:
                    st.error("You must agree to the Terms and Conditions.")
                    return False, {"error": "You must agree to the Terms and Conditions."}

                # Register the user
                try:
                    print("Calling register_user function...")
                    success, response = register_user(
                        username=username,
                        email=email,
                        password=password,
                        first_name=first_name,
                        last_name=last_name,
                        role=role
                    )

                    print(f"Registration result: success={success}, response={response}")

                    if success:
                        st.success("Registration successful! You can now log in.")
                        return True, response
                    else:
                        error_msg = "Registration failed."
                        if isinstance(response, dict) and 'error' in response:
                            error_msg = response['error']
                        st.error(error_msg)
                        return False, response
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    print(f"Error during registration: {e}")
                    print(error_trace)
                    st.error(f"Registration error: {e}")
                    return False, {"error": str(e), "traceback": error_trace}

            return False, None
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error displaying registration form: {e}")
        print(error_trace)
        st.error(f"Error displaying registration form: {e}")
        st.code(error_trace)
        return False, {"error": str(e), "traceback": error_trace}
