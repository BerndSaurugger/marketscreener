# market_screener_app.py
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Market Tools",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"  # or "auto", not "collapsed"
)

# Custom CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #f8fafc;
    }
    
    .header {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        padding: 2rem 1rem;
        border-radius: 0 0 20px 20px;
    }
    
    .tool-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        height: 100%;
        border: 1px solid #e2e8f0;
    }
    
    .tool-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #0ea5e9;
    }
    
    /* Custom button styling */
    .stButton>button {
        background: #0ea5e9 !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background: #0284c7 !important;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="header">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    st.title("📊 Trading Tools")
    st.markdown("""
    <h3 style='color:black; font-weight:600;'>
        Personal stock analysis for our group
    </h3>
    <p style='font-size:1.1rem;'>
        Simple tools to screen markets and track our watchlists
    </p>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Start Exploring"):
        st.session_state.show_screener = True

with col2:
    # Placeholder for chart image
    st.image(Image.new('RGB', (600, 300), color='#7dd3fc'), 
             caption="Analysis Dashboard")
st.markdown('</div>', unsafe_allow_html=True)

# Main Tools Section
st.markdown("<br><br>", unsafe_allow_html=True)
st.header("Our Analysis Tools", anchor=False)
st.markdown("---")

tools = [
    {
        "icon": "🔍",
        "title": "Stock Screener",
        "desc": "Filter stocks by your custom criteria"
    },
    {
        "icon": "📈",
        "title": "Technical Charts",
        "desc": "Interactive charts with drawing tools"
    },
    {
        "icon": "📋",
        "title": "Watchlist",
        "desc": "Our shared watchlist of interesting stocks"
    }
]

cols = st.columns(3)
for idx, tool in enumerate(tools):
    with cols[idx]:
        container = st.container()
        with container:
            st.markdown(f'<div class="feature-icon">{tool["icon"]}</div>', unsafe_allow_html=True)
            st.subheader(tool["title"])
            st.write(tool["desc"])
            
            # Use a unique key for each button
            if st.button("Open Tool", key=f"btn_{idx}"):
                st.session_state.active_tool = tool["title"]

# How We Use It Section
st.markdown("<br><br>", unsafe_allow_html=True)
st.header("How We Use This", anchor=False)
st.markdown("---")

how_we_use = """
1. **Weekly watchlist updates** - Every Sunday we update our shared watchlist
2. **Technical scans** - Run momentum/volume scans before market open
3. **Chart sharing** - Share interesting chart setups in our group chat
4. **Sector tracking** - Monitor specific industry groups we're watching
"""

st.markdown(how_we_use)

# Data Sources Section
st.markdown("<br><br>", unsafe_allow_html=True)
st.header("Our Data Sources", anchor=False)
st.markdown("---")

sources = [
    {"name": "Yahoo Finance", "purpose": "Free market data"},
    {"name": "Trading View", "purpose": "Fundamental data"},
    {"name": "xxx", "purpose": "xxx"},
    {"name": "xxx", "purpose": "xxx"}
]

source_cols = st.columns(4)
for idx, source in enumerate(sources):
    with source_cols[idx]:
        st.info(f"**{source['name']}**  \n{source['purpose']}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1rem; color:#64748b;">
    <p>Personal trading tools • Updated weekly • For our group only</p>
</div>
""", unsafe_allow_html=True)

# Tool activation message
if "active_tool" in st.session_state:
    st.info(f"Opening {st.session_state.active_tool}... (This is a placeholder)")
    st.session_state.pop("active_tool")