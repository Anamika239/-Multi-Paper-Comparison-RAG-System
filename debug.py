import streamlit as st

st.set_page_config(page_title="Debug View", layout="wide")

st.title("🔍 Theme Debug Viewer")

# Check what Streamlit thinks the theme is
st.write("## Streamlit Theme Detection")
st.write(f"Current theme: `{st.get_option('theme.base')}`")

# Let user manually switch for testing
st.write("## Manual Theme Override (for testing)")
col1, col2 = st.columns(2)
with col1:
    if st.button("Force Light Theme Test"):
        st.markdown("""
        <style>
        /* Force everything white */
        .stApp, .main, .block-container, 
        div[data-testid="stVerticalBlock"],
        div[data-testid="column"],
        .element-container {
            background: #ffffff !important;
        }
        
        /* Force all text black */
        * {
            color: #000000 !important;
        }
        
        /* Sidebar light gray */
        section[data-testid="stSidebar"] {
            background: #f8fafc !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.success("Light theme forced - check if it works")

with col2:
    if st.button("Force Dark Theme Test"):
        st.markdown("""
        <style>
        /* Force everything dark */
        .stApp, .main, .block-container, 
        div[data-testid="stVerticalBlock"],
        div[data-testid="column"],
        .element-container {
            background: #0f172a !important;
        }
        
        /* Force all text white */
        * {
            color: #ffffff !important;
        }
        
        /* Sidebar dark */
        section[data-testid="stSidebar"] {
            background: #1e293b !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.success("Dark theme forced - check if it works")

st.write("## Current CSS Inspection")
st.write("Open browser DevTools (F12) and check what CSS is actually being applied")

# Show a sample component to test
st.write("## Test Components")
st.text_input("Test Input", placeholder="Type here...")
st.multiselect("Test Multi-select", ["Option 1", "Option 2", "Option 3"])
st.button("Test Button")

st.write("## Browser Info")
import streamlit.web.bootstrap as bootstrap
st.write(f"Browser color scheme: `{st.get_option('browser.gatherUsageStats')}`")
