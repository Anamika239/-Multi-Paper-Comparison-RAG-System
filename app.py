import streamlit as st
import pandas as pd
import plotly.express as px
from document_processor import DocumentProcessor
from rag_engine import MultiPaperRAG
import os
import time

st.set_page_config(
    page_title="Multi Research Paper Comparison",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== SIDEBAR – with remove buttons (no emojis) =====
with st.sidebar:
    st.markdown("# Paper Management")
    if not os.path.exists("papers"):
        os.makedirs("papers")
        st.info("Created 'papers' folder")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        for f in uploaded_files:
            with open(os.path.join("papers", f.name), "wb") as fp:
                fp.write(f.getbuffer())
            st.success(f"Uploaded: {f.name}")

    if st.button("Load/Reload All Papers", use_container_width=True):
        if 'rag' not in st.session_state:
            st.session_state.rag = MultiPaperRAG()
            st.session_state.processor = DocumentProcessor()
        papers = st.session_state.processor.process_papers_folder("papers")
        if papers:
            prog = st.progress(0)
            for i, (name, text, meta) in enumerate(papers):
                st.session_state.rag.add_paper(text, name, meta)
                prog.progress((i + 1) / len(papers))
                time.sleep(0.05)
            st.success(f"Loaded {len(papers)} papers")
        else:
            st.warning("No PDFs found")

    st.markdown("### Loaded Papers")
    if 'rag' in st.session_state:
        papers = st.session_state.rag.get_all_papers()
        if papers:
            for paper in papers:
                col1, col2 = st.columns([5, 1])
                with col1:
                    display = paper if len(paper) <= 28 else paper[:25] + "..."
                    st.markdown(f"📄 {display}")  # document icon kept for clarity
                with col2:
                    if st.button("X", key=f"del_{paper}", help="Remove this paper"):
                        if st.session_state.rag.remove_paper(paper):
                            st.success(f"Removed {paper}")
                            st.rerun()
                        else:
                            st.error(f"Failed to remove {paper}")
            # Clear All button
            if st.button("Clear All", use_container_width=True):
                for paper in papers:
                    st.session_state.rag.remove_paper(paper)
                st.success("All papers removed")
                st.rerun()
        else:
            st.info("No papers loaded")
    else:
        st.info("No papers loaded")

# ===== DARK THEME CSS – with bigger tabs and animated header =====
st.markdown("""
<style>
    .stApp { background: #0f172a; }
    section[data-testid="stSidebar"] { background: #1e293b; }
    .stTextInput input, .stTextArea textarea,
    .stMultiSelect div[data-baseweb="select"] {
        background: #1e293b !important;
        color: white !important;
        border: 1px solid #334155 !important;
    }
    .stFileUploader > div {
        background: #1e293b !important;
        border: 2px dashed #334155;
    }
    .stButton button {
        background: linear-gradient(135deg, #4361ee, #4cc9f0);
        color: white;
        border: none;
    }
    h1, h2, h3, p, li, label, .stMarkdown {
        color: white !important;
    }
    .dataframe {
        background: #1e293b;
        color: white;
        border: 1px solid #334155;
    }
    .stAlert {
        background: #1e293b;
        color: white;
        border-left: 4px solid #4361ee;
    }
    /* Tabs – bigger, no emojis */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 50px;
        padding: 0.5rem;
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #a0a0a0 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 50px !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: white !important;
        background: rgba(255,255,255,0.05) !important;
    }
    .stTabs [aria-selected="true"] {
        background: #4361ee !important;
        color: white !important;
    }
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #4361ee, #4cc9f0);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
    }
    /* Hide Streamlit clutter but keep sidebar toggle */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stApp header { background: transparent; }
    button[kind="header"] { display: block !important; }

    /* Animated header */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .animated-header {
        background: linear-gradient(270deg, #4361ee, #4cc9f0, #a78bfa, #4361ee);
        background-size: 300% 300%;
        animation: gradientShift 10s ease infinite;
        padding: 3rem 2rem;
        border-radius: 30px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .animated-header h1 {
        color: white !important;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    .animated-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== ANIMATED HEADER =====
st.markdown("""
<div class="animated-header">
    <h1>Multi Research Paper Comparison</h1>
    <p>Compare, analyze, and understand multiple research papers with AI‑powered semantic search</p>
</div>
""", unsafe_allow_html=True)

# ===== TABS (no emojis) =====
tab1, tab2, tab3, tab4 = st.tabs(["Compare", "Statistics", "Search", "Themes"])

# ---------- TAB 1: Compare ----------
with tab1:
    if 'rag' in st.session_state:
        col1, col2 = st.columns([2, 1])
        with col1:
            q = st.text_area("Query", placeholder="Compare methodologies...")
        with col2:
            papers = st.session_state.rag.get_all_papers()
            sel = st.multiselect("Papers", papers, default=papers[:2] if len(papers) >= 2 else papers)
        if st.button("Compare", use_container_width=True) and q and sel:
            comp = st.session_state.rag.compare_papers(q, sel)
            st.dataframe(pd.DataFrame([
                {'Paper': p[:30], 'Relevant': v['relevant_chunks'],
                 'Score': f"{1 - v['avg_relevance']:.2%}" if v['avg_relevance'] < 1 else "N/A"}
                for p, v in comp['papers'].items()
            ]))
            cols = st.columns(len(sel))
            for i, (p, v) in enumerate(comp['papers'].items()):
                with cols[i]:
                    st.write(f"**{p[:20]}...**")
                    for j, passage in enumerate(v['top_passages'][:2]):
                        with st.expander(f"Passage {j+1}"):
                            st.write(passage[:300] + "..." if len(passage) > 300 else passage)
    else:
        st.info("Load papers first")

# ---------- TAB 2: Statistics ----------
with tab2:
    if 'rag' in st.session_state:
        papers = st.session_state.rag.get_all_papers()
        stats = [st.session_state.rag.get_paper_summary_stats(p) for p in papers]
        stats = [s for s in stats if s]
        if stats:
            df = pd.DataFrame(stats)
            c1, c2, c3 = st.columns(3)
            c1.metric("Papers", len(df))
            c2.metric("Avg chunks", int(df['num_chunks'].mean()))
            c3.metric("Total words", f"{int(df['total_words'].sum()):,}")

            fig = px.bar(df, x='paper', y='num_chunks',
                         title="Chunks per Paper",
                         color_discrete_sequence=['#4361ee'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df)
        else:
            st.info("No statistics available")
    else:
        st.info("Load papers first")

# ---------- TAB 3: Semantic Search ----------
with tab3:
    if 'rag' in st.session_state:
        q = st.text_input("Search query", placeholder="attention mechanism...")
        if q:
            results = st.session_state.rag.retrieve_similar_chunks(q, 10)
            if results:
                st.success(f"Found {len(results)} passages")
                for r in results:
                    paper = r['metadata'].get('paper_name', '')
                    rel = 1 - r['distance']
                    with st.expander(f"{paper[:40]} – {rel:.1%}"):
                        st.write(r['text'])
            else:
                st.info("No results")
    else:
        st.info("Load papers first")

# ---------- TAB 4: Common Themes ----------
with tab4:
    if 'rag' in st.session_state:
        papers = st.session_state.rag.get_all_papers()
        sel = st.multiselect("Select papers", papers, default=papers)
        if sel and st.button("Find Themes", use_container_width=True):
            themes = st.session_state.rag.find_common_themes(sel, 12)
            cols = st.columns(3)
            for i, th in enumerate(themes):
                cols[i % 3].markdown(
                    f"<div style='background:#1e293b; color:#4361ee; padding:0.75rem; "
                    f"border-radius:8px; text-align:center;'>{th}</div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("Load papers first")

# ===== FOOTER =====
st.markdown(
    '<div class="footer"><p>Multi Research Paper Comparison System | Powered by Sentence Transformers + ChromaDB</p></div>',
    unsafe_allow_html=True
)
