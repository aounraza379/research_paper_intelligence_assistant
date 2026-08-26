from html import escape

import streamlit as st

from rag import (
    PaperIndex,
    build_paper_index,
    generate_answer,
    load_embedding_model,
)


st.set_page_config(
    page_title="Research Paper Intelligence Assistant",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#17211f; --muted:#687571; --paper:#f7f6f1; --cream:#eeece4; --accent:#e3633f; --teal:#176b65; --line:#d9d8ce; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
[data-testid="stSidebar"] { background: #e5e4da; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
.brand { font-family:'Space Grotesk', sans-serif; font-size:1.15rem; font-weight:700; letter-spacing:0; color:var(--teal); }
.eyebrow { color:var(--accent); font-size:.74rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; margin-bottom:.7rem; }
.hero { padding: 3.6rem 0 2rem; max-width: 880px; }
.hero h1 { font-family:'Space Grotesk', sans-serif; font-size:clamp(2.35rem, 5vw, 4.8rem); line-height:.98; letter-spacing:0; margin:0; max-width:800px; }
.hero p { color:var(--muted); font-size:1.08rem; line-height:1.6; max-width:680px; margin-top:1.35rem; }
.status { background:var(--cream); border:1px solid var(--line); border-left:4px solid var(--teal); padding:1rem 1.15rem; margin: .5rem 0 1.8rem; }
.status strong { color:var(--teal); }
.source { background:#fff; border:1px solid var(--line); border-radius:6px; padding:1rem 1.15rem; margin:.7rem 0; }
.source-meta { color:var(--teal); font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; margin-bottom:.45rem; }
.source-text { color:#3d4945; line-height:1.55; font-size:.94rem; }
section[data-testid="stChatMessage"] { background:transparent; }
.stButton > button { border-radius:4px; border:1px solid var(--teal); color:var(--teal); font-weight:600; }
.stButton > button[kind="primary"] { background:var(--accent); border-color:var(--accent); color:white; }
div[data-testid="stMetric"] { background:var(--cream); border:1px solid var(--line); padding: .7rem; }
.small-note { color:var(--muted); font-size:.82rem; line-height:1.45; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the local embedding model...")
def get_model():
    return load_embedding_model()


def initialize_state() -> None:
    st.session_state.setdefault("paper", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_upload", None)


def render_sources(sources) -> None:
    st.markdown("#### Evidence retrieved from the paper")
    for source in sources:
        page_label = f"Page {source.page}" if source.page else "Bundled demo source"
        st.markdown(
            f"""
<div class="source">
<div class="source-meta">Source {source.rank} &nbsp; | &nbsp; {escape(page_label)} &nbsp; | &nbsp; distance {source.distance:.3f}</div>
<div class="source-text">{escape(source.text)}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def main() -> None:
    initialize_state()
    model = get_model()

    with st.sidebar:
        st.markdown('<div class="brand">RESEARCH PAPER<br>INTELLIGENCE ASSISTANT</div>', unsafe_allow_html=True)
        st.markdown("### 1. Add your paper")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file and st.button("Process paper", type="primary", use_container_width=True):
            with st.spinner("Extracting, chunking, and indexing..."):
                try:
                    st.session_state.paper = build_paper_index(
                        uploaded_file.getvalue(), uploaded_file.name, model
                    )
                    st.session_state.messages = []
                    st.success("Paper indexed and ready.")
                except ValueError as error:
                    st.error(str(error))

        if st.session_state.paper:
            st.markdown("---")
            st.markdown(f"**Paper ready**  \n{st.session_state.paper.title}")
            st.caption(f"{len(st.session_state.paper.chunks)} searchable passages")
        else:
            st.info("Upload a text-based PDF, then click Process paper.")
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown(
            "1. Upload a research paper PDF.\n2. Click **Process paper**.\n3. Ask a question below.\n4. Open the evidence to verify the answer."
        )
        st.markdown("**Grounding policy**")
        st.markdown(
            '<div class="small-note">Answers are generated only from retrieved passages. Unsupported questions are surfaced instead of guessed.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="hero"><div class="eyebrow">Evidence-first literature review</div><h1>Ask your paper.<br>See the evidence.</h1><p>Turn a dense research PDF into a focused conversation. Every answer is grounded in passages you can inspect.</p></div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.paper:
        st.warning("Upload a PDF in the sidebar to activate your research workspace.")
        return

    paper: PaperIndex = st.session_state.paper
    st.markdown(
        f'<div class="status"><strong>Paper ready:</strong> {paper.title}</div>',
        unsafe_allow_html=True,
    )
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Indexed passages", len(paper.chunks))
    metric_two.metric("Embedding model", "MiniLM")
    metric_three.metric("Answer mode", "Grounded")

    st.markdown("### 2. Ask questions about your paper")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Inspect retrieved evidence"):
                    render_sources(message["sources"])

    question = st.chat_input("Ask about the paper's methods, findings, or limitations...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the paper and checking the evidence..."):
                try:
                    answer, sources = generate_answer(paper, question, model)
                    st.markdown(answer)
                    with st.expander("Inspect retrieved evidence", expanded=True):
                        render_sources(sources)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except RuntimeError as error:
                    st.error(str(error))
                    st.info("Add GROQ_API_KEY under Streamlit Cloud App settings > Secrets.")
                except Exception as error:
                    error_name = type(error).__name__
                    if error_name == "NotFoundError":
                        st.error(
                            "The configured Groq model is unavailable. Set GROQ_MODEL to "
                            "llama-3.1-8b-instant in Streamlit Cloud Secrets, then reboot the app."
                        )
                    else:
                        st.error(
                            "Groq could not generate an answer right now. Check the API key, "
                            "model setting, and app logs, then try again."
                        )


if __name__ == "__main__":
    main()
