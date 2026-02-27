"""
MedAgent 360 · Streamlit Dashboard
Unified web UI for all three modules.
Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="MedAgent 360",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/caduceus.png", width=80)
st.sidebar.title("MedAgent 360")
st.sidebar.caption("Autonomous Healthcare AI Agent")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔬 Lab Report", "💊 Prescription", "📞 Follow-up Agent", "🚨 Alerts"],
)

language = st.sidebar.selectbox("Language / భాష / भाषा", ["English", "Telugu", "Hindi"])

st.sidebar.divider()

# ── Pages ──────────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    st.title("🏥 MedAgent 360")
    st.subheader("Autonomous Healthcare AI Agent for Rural India")
    st.markdown("""
    > *650 million Indians can't understand their medical reports. MedAgent 360 changes that.*

    ---

    ### What this agent does:
    - **🔬 Module A** — Reads your blood test PDF, flags abnormal values, explains results in your language
    - **💊 Module B** — Scans your prescription photo, identifies medicines, gives voice instructions
    - **📞 Module C** — Sends WhatsApp check-ins, monitors symptoms, alerts your doctor if critical

    ### How to use:
    Select a module from the sidebar to get started.
    """)

    col1, col2, col3 = st.columns(3)
    col2.metric("Languages", "3", "EN / Telugu / Hindi")
    col3.metric("Build Time", "24 hrs", "Hackathon Sprint")


elif page == "🔬 Lab Report":
    st.title("🔬 Lab Report Intelligence")
    st.caption("Upload your blood test PDF and get an AI-powered explanation in your language.")

    uploaded_file = st.file_uploader("Upload Lab Report PDF", type=["pdf"])

    if uploaded_file:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"📄 File uploaded: {uploaded_file.name}")
        with col2:
            analyze_btn = st.button("🔍 Analyze Report", type="primary", use_container_width=True)

        if analyze_btn:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            with st.spinner("🤖 Analysing your report with AI..."):
                try:
                    from lab_report.rag_pipeline import run_full_pipeline
                    result = run_full_pipeline(tmp_path, language=language)

                    # Patient info
                    info = result["patient_info"]
                    st.subheader(f"Patient: {info['name']} | Age: {info['age']} | Date: {info['date']}")

                    # Stats
                    stats = result["stats"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Tests", stats["total"])
                    c2.metric("✅ Normal", stats["normal"])
                    c3.metric("⚠️ Abnormal", stats["abnormal"], delta_color="inverse")

                    # Results table
                    st.subheader("📊 Test Results")
                    import pandas as pd
                    df = pd.DataFrame(result["classified_values"])
                    
                    if not df.empty:
                        display_cols = ["test", "value", "unit", "status", "risk_icon", "benchmark_min", "benchmark_max"]
                        available = [c for c in display_cols if c in df.columns]

                        def color_status(val):
                            colors = {"CRITICAL": "background-color: #ff4444; color: white",
                                      "HIGH": "background-color: #ff9900; color: white",
                                      "LOW": "background-color: #ffcc00",
                                      "NORMAL": "background-color: #00cc44; color: white"}
                            return colors.get(val, "")

                        styled = df[available].style.applymap(color_status, subset=["status"])
                        st.dataframe(styled, use_container_width=True)
                    else:
                        st.warning("No tabular lab values were detected in this document. The AI summary below evaluates the raw text instead.")

                    # AI Summary
                    st.subheader(f"🤖 AI Summary ({language})")
                    st.info(result["summary"])

                    # Critical flags
                    if result["critical_flags"]:
                        st.error(f"🚨 CRITICAL values detected: {', '.join(result['critical_flags'])} — Please see a doctor immediately!")

                    # Audio
                    st.subheader("🔊 Listen to Summary")
                    with st.spinner("Generating audio..."):
                        from lab_report.voice import generate_audio
                        audio_path = generate_audio(result["summary"], language)
                        with open(audio_path, "rb") as f:
                            st.audio(f.read(), format="audio/mp3")
                        os.unlink(audio_path)

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

elif page == "💊 Prescription":
    st.title("💊 Prescription Parser")
    st.info("Module B — Coming soon (Dev 2)")

elif page == "📞 Follow-up Agent":
    st.title("📞 Autonomous Follow-up Agent")
    st.info("Module C — Coming soon (Dev 3)")

elif page == "🚨 Alerts":
    st.title("🚨 Doctor Alerts")
    st.info("Alert dashboard — Available after Module C is connected.")
