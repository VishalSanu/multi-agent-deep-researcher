"""Streamlit UI for the Multi-Agent Deep Researcher."""
import os
import sys
import asyncio
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore")

# Link local virtualenv if running outside it
project_root = Path(__file__).resolve().parent
venv_site_packages = project_root / ".venv" / "Lib" / "site-packages"
if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
    sys.path.insert(0, str(venv_site_packages))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv, set_key

from deep_researcher.models.schemas import ResearchBrief, SourceDocument
from deep_researcher.orchestrator import DeepResearcherOrchestrator
from deep_researcher.history_manager import save_session, list_sessions, load_session

# Load existing environment
ENV_PATH = project_root / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Deep Researcher",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E88E5;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1E88E5;
    }
    .conflict-card {
        background-color: #fff9e6;
        border: 1px solid #ffe082;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .source-badge-primary {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .source-badge-secondary {
        background-color: #f5f5f5;
        color: #616161;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def test_gemini_connection(api_key: str) -> tuple[bool, str]:
    """Validates Gemini API key credentials with a lightweight ping."""
    if not api_key:
        return False, "API key cannot be empty."
    try:
        from google import genai
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Say 'OK'",
        )
        if response and response.text:
            return True, "Successfully connected to Gemini API!"
        return False, "Received empty response from API."
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


# ---------------- SIDEBAR: SETTINGS & HISTORY ----------------
with st.sidebar:
    st.markdown("### 🔬 Deep Researcher Studio")
    st.markdown("---")

    # API Configuration Section
    st.markdown("#### 🔑 API Credentials")
    env_gemini = os.getenv("GEMINI_API_KEY", "")
    env_tavily = os.getenv("TAVILY_API_KEY", "")

    gemini_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_gemini,
        type="password",
        help="Required. Get from Google AI Studio (aistudio.google.com).",
    )
    tavily_key_input = st.text_input(
        "Tavily Search API Key",
        value=env_tavily,
        type="password",
        help="Optional. If omitted, DuckDuckGo search is used automatically.",
    )

    col_test, col_save = st.columns(2)
    with col_test:
        if st.button("🔌 Test Key", use_container_width=True):
            with st.spinner("Pinging API..."):
                ok, msg = test_gemini_connection(gemini_key_input)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    with col_save:
        if st.button("💾 Save to .env", use_container_width=True):
            if not ENV_PATH.exists():
                ENV_PATH.touch()
            set_key(str(ENV_PATH), "GEMINI_API_KEY", gemini_key_input)
            if tavily_key_input:
                set_key(str(ENV_PATH), "TAVILY_API_KEY", tavily_key_input)
            os.environ["GEMINI_API_KEY"] = gemini_key_input
            if tavily_key_input:
                os.environ["TAVILY_API_KEY"] = tavily_key_input
            st.success("Saved to .env!")

    # Set in os.environ for current execution
    if gemini_key_input:
        os.environ["GEMINI_API_KEY"] = gemini_key_input
    if tavily_key_input:
        os.environ["TAVILY_API_KEY"] = tavily_key_input

    st.markdown("---")
    st.markdown("#### ⚙️ Research Parameters")
    max_sources = st.slider("Max Sources to Scrape", min_value=3, max_value=20, value=10, step=1)
    max_search_loops = st.slider("Max Re-Search Loops", min_value=0, max_value=4, value=2, step=1)

    with st.expander("🤖 Model Selection"):
        model_fast = st.selectbox(
            "Fast / Extraction Model",
            options=["gemini-3.8-flash"],
            index=0,
            help="Used for query planning, text chunking, and raw claim extraction.",
        )
        model_pro = st.selectbox(
            "Reasoning / Synthesis Model",
            options=["gemini-3.1-pro-preview", "gemini-3.8-flash"],
            index=0,
            help="Used for conflict arbitration, deep reasoning, and final narrative synthesis.",
        )

    st.markdown("---")
    st.markdown("#### 📂 Research History")
    past_sessions = list_sessions()

    if st.button("➕ New Research", use_container_width=True):
        st.session_state["current_brief"] = None
        st.rerun()

    if past_sessions:
        session_options = {s["display_label"]: s["json_path"] for s in past_sessions}
        selected_label = st.selectbox(
            "Past Reports",
            options=list(session_options.keys()),
            index=None,
            placeholder="Select a past brief to load...",
        )
        if selected_label and st.button("📖 Load Selected Brief", use_container_width=True):
            json_file = session_options[selected_label]
            loaded_data = load_session(json_file)
            if loaded_data:
                st.session_state["current_brief"] = loaded_data
                st.success("Loaded from history!")
                st.rerun()
    else:
        st.caption("No past research runs yet.")


# ---------------- MAIN CONTENT AREA ----------------
st.markdown('<div class="main-title">Multi-Agent Deep Researcher 🔬</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A coordinated team of specialized agents: '
    '<b>Searcher</b> (Scrapes primary sources) → <b>Fact-Checker</b> (Audits atomic claims & contradictions) → '
    '<b>Writer</b> (Compiles publication-grade cited brief)</div>',
    unsafe_allow_html=True,
)

# Initialize Session State
if "current_brief" not in st.session_state:
    st.session_state["current_brief"] = None

# Callback to safely set topic suggestions before widget instantiation
def set_topic_suggestion(topic: str):
    st.session_state["topic_input"] = topic


# Topic Input Section
col_input, col_btn = st.columns([4, 1])
with col_input:
    sample_topic = "Impact of Solid-State Batteries on Grid-Scale Storage vs Lithium-Iron-Phosphate"
    topic_query = st.text_input(
        "Enter Complex Research Topic or Hypothesis",
        placeholder=f"e.g., {sample_topic}",
        key="topic_input",
    )

with col_btn:
    st.write("")  # Spacing
    st.write("")
    start_clicked = st.button("🚀 Start Deep Research", type="primary", use_container_width=True)

# Sample Topic Chips
st.caption("Quick suggestions:")
chip_cols = st.columns(3)
with chip_cols[0]:
    st.button(
        "🔋 Solid-State vs LFP Storage",
        use_container_width=True,
        on_click=set_topic_suggestion,
        args=("Solid-State Batteries vs Lithium-Iron-Phosphate for Grid-Scale Storage",),
    )
with chip_cols[1]:
    st.button(
        "🧬 mRNA Cancer Vaccines Efficacy",
        use_container_width=True,
        on_click=set_topic_suggestion,
        args=("Efficacy of mRNA Cancer Vaccines: Clinical Trial Outcomes vs Immune Tolerance",),
    )
with chip_cols[2]:
    st.button(
        "⚛️ Quantum Error Correction",
        use_container_width=True,
        on_click=set_topic_suggestion,
        args=("Fault-Tolerant Quantum Computing: Neutral Atoms vs Superconducting Qubits",),
    )

st.markdown("---")


# ---------------- EXECUTION RUNNER ----------------
if start_clicked:
    current_topic = st.session_state.get("topic_input", "").strip()
    if not current_topic:
        st.warning("Please enter a research topic first.")
    elif not os.getenv("GEMINI_API_KEY"):
        st.error("Please provide your Google Gemini API Key in the sidebar.")
    else:
        # Create Live Multi-Stage Stepper
        status_box = st.status("🔬 Deep Research Team Initialized...", expanded=True)
        log_messages = []

        def ui_status_callback(msg: str):
            clean_msg = (
                msg.replace("[bold blue]", "**")
                .replace("[/bold blue]", "**")
                .replace("[bold yellow]", "**")
                .replace("[/bold yellow]", "**")
                .replace("[cyan]", "")
                .replace("[/cyan]", "")
                .replace("[green]", "")
                .replace("[/green]", "")
                .replace("[yellow]", "")
                .replace("[/yellow]", "")
                .replace("[magenta]", "")
                .replace("[/magenta]", "")
                .replace("[dim]", "*")
                .replace("[/dim]", "*")
                .replace("[bold green]", "**")
                .replace("[/bold green]", "**")
            )
            log_messages.append(clean_msg)
            status_box.markdown(f"**Current Action:** {clean_msg}")

        orchestrator = DeepResearcherOrchestrator(
            search_model=model_fast,
            fact_check_extraction_model=model_fast,
            fact_check_reasoning_model=model_pro,
            writer_model=model_pro,
        )

        try:
            status_box.update(label="🚀 Phase 1: Searcher Agent planning queries...", state="running")
            # Run the async pipeline
            brief_obj = asyncio.run(
                orchestrator.run_research(
                    topic=current_topic,
                    max_sources=max_sources,
                    max_search_loops=max_search_loops,
                    on_status=ui_status_callback,
                )
            )

            # Persist to local history
            saved_paths = save_session(brief_obj)
            st.session_state["current_brief"] = brief_obj.model_dump()

            status_box.update(
                label=f"✔ Research Complete! Brief generated and archived to history.",
                state="complete",
                expanded=False,
            )
            st.toast("Research Brief successfully compiled!", icon="🎉")
            st.rerun()

        except Exception as e:
            status_box.update(label=f"❌ Error during research: {str(e)}", state="error")
            st.error(f"Execution Error: {str(e)}")


# ---------------- DASHBOARD RESULTS VIEW ----------------
brief_data = st.session_state.get("current_brief")

if brief_data:
    st.markdown(f"## 📑 {brief_data.get('title', 'Research Brief')}")

    # Top Metric Banner
    bib_list = brief_data.get("bibliography", [])
    claims_list = brief_data.get("claims", [])
    conflicts_list = brief_data.get("conflicts", [])
    primary_count = sum(1 for b in bib_list if b.get("is_primary"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Consulted Sources", len(bib_list))
    m2.metric("Primary / Official Sources", f"{primary_count} / {len(bib_list)}")
    m3.metric("Atomic Claims Verified", len(claims_list))
    m4.metric("Contradictions Arbitrated", len(conflicts_list))

    # Download Actions
    col_dl_md, col_dl_json, _ = st.columns([1.5, 1.5, 5])
    with col_dl_md:
        st.download_button(
            label="📥 Download Brief (.md)",
            data=brief_data.get("full_markdown", ""),
            file_name=f"research_brief_{brief_data.get('topic', 'report')[:30].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_dl_json:
        import json
        st.download_button(
            label="📊 Download Data (.json)",
            data=json.dumps(brief_data, indent=2, ensure_ascii=False),
            file_name=f"research_data_{brief_data.get('topic', 'report')[:30].replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")

    # Tabs Layout
    tab_brief, tab_conflicts, tab_sources, tab_claims = st.tabs(
        [
            "📄 Executive Brief",
            "⚖️ Conflicting Claims & Nuances",
            "📚 Source Bibliography",
            "🔍 Claim Audit Matrix",
        ]
    )

    # Tab 1: Executive Brief
    with tab_brief:
        st.markdown("### Executive Summary")
        st.info(brief_data.get("executive_summary", "No executive summary available."))

        st.markdown("### Evidence Synthesis")
        sections = brief_data.get("sections", [])
        if sections:
            for sec in sections:
                st.markdown(f"#### {sec.get('heading', '')}")
                st.markdown(sec.get("content", ""))
                st.write("")
        else:
            # Fallback to full markdown
            st.markdown(brief_data.get("full_markdown", ""))

    # Tab 2: Conflicting Claims & Nuances
    with tab_conflicts:
        st.markdown("### ⚖️ Empirical Contradictions & Nuance Analysis")
        if conflicts_list:
            st.markdown(
                "The Fact-Checker cross-referenced claims across sources to identify and arbitrate direct disputes:"
            )
            for i, conf in enumerate(conflicts_list):
                claim_a = conf.get("claim_a", {})
                claim_b = conf.get("claim_b", {})
                with st.container():
                    st.markdown(
                        f"""
                        <div class="conflict-card">
                            <h4 style="margin-top:0; color:#b78103;">⚡ Focus: {conf.get('topic_focus', f'Dispute #{i+1}')}</h4>
                            <div style="display:flex; gap:16px; margin-bottom:12px;">
                                <div style="flex:1; background:#ffffff; padding:10px; border-radius:6px; border:1px solid #ddd;">
                                    <b>Perspective A:</b> {claim_a.get('text', '')}<br>
                                    <small><i>Source: {claim_a.get('source_title', claim_a.get('source_domain', ''))}</i></small><br>
                                    <blockquote style="font-size:0.85rem; color:#555; margin:6px 0;">"{claim_a.get('quote', '')[:140]}..."</blockquote>
                                </div>
                                <div style="flex:1; background:#ffffff; padding:10px; border-radius:6px; border:1px solid #ddd;">
                                    <b>Perspective B:</b> {claim_b.get('text', '')}<br>
                                    <small><i>Source: {claim_b.get('source_title', claim_b.get('source_domain', ''))}</i></small><br>
                                    <blockquote style="font-size:0.85rem; color:#555; margin:6px 0;">"{claim_b.get('quote', '')[:140]}..."</blockquote>
                                </div>
                            </div>
                            <b>Synthesis / Resolution:</b> {conf.get('resolved_synthesis') or conf.get('nature_of_conflict', '')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.success("✔ No critical empirical contradictions detected across the analyzed sources.")

    # Tab 3: Source Bibliography
    with tab_sources:
        st.markdown("### 📚 Source Bibliography & Authority Ratings")
        if bib_list:
            for b in bib_list:
                cite_no = b.get("citation_number", 1)
                is_prim = b.get("is_primary", False)
                badge_class = "source-badge-primary" if is_prim else "source-badge-secondary"
                badge_text = "PRIMARY / OFFICIAL" if is_prim else "SECONDARY"

                st.markdown(
                    f"""
                    <div style="padding:10px 0; border-bottom:1px solid #eee;">
                        <b>[{cite_no}]</b> <a href="{b.get('url', '#')}" target="_blank" style="font-weight:600; text-decoration:none;">{b.get('title', 'Source')}</a>
                        <span class="{badge_class}" style="margin-left:8px;">{badge_text}</span>
                        <div style="font-size:0.85rem; color:#666; margin-top:3px;">
                            Domain: <code>{b.get('domain', '')}</code>
                        </div>
                        <div style="font-size:0.85rem; color:#444; margin-top:4px; font-style:italic;">
                            &ldquo;{b.get('key_excerpt', '')}&rdquo;
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No bibliography records found.")

    # Tab 4: Claim Audit Matrix
    with tab_claims:
        st.markdown("### 🔍 Granular Atomic Claims Audit Trail")
        if claims_list:
            claims_table_data = []
            for c in claims_list:
                claims_table_data.append({
                    "ID": c.get("id"),
                    "Subtopic": c.get("subtopic"),
                    "Stance": c.get("stance"),
                    "Claim Assertion": c.get("text"),
                    "Verbatim Quote": f'"{c.get("quote", "")[:120]}..."',
                    "Source Domain": c.get("source_domain"),
                    "Primary": "Yes" if c.get("is_primary_source") else "No",
                    "Credibility": f"{c.get('credibility_score', 0.7):.2f}",
                })
            df_claims = pd.DataFrame(claims_table_data)
            st.dataframe(df_claims, use_container_width=True, hide_index=True)
        else:
            st.caption("No claim records found.")
