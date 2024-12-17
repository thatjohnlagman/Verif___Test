import streamlit as st
from streamlit_navigation_bar import st_navbar

# Define the pages
def home():
    st.title('Home')
    st.write('Welcome to the homepage!')

def documentation():
    st.title('Documentation')
    st.write('Here is the documentation.')

def examples():
    st.title('Examples')
    st.write('Explore the examples here.')

def community():
    st.title('Community')
    st.write('Join our community discussions.')

def about():
    st.title('About')
    st.write('Learn more about us.')


# Sidebar navigation
page = ["Home", "Documentation", "Examples", "Community", "About"]

page1 = st_navbar(page)

st.write(page1)

# Display the corresponding page content
if page == "Home":
    home()
elif page == "Documentation":
    documentation()
elif page == "Examples":
    examples()
elif page == "Community":
    community()
else:
    about()
