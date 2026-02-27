"""
MedAgent 360 · Streamlit Dashboard (Phase 1)
Full UI for all 3 modules with live demo support.
Run with: streamlit run app.py
"""
import streamlit as st
import tempfile, os, json
import pandas as pd

st.set_page_config(
    page_title="MedAgent 360", page_icon="🏥", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedAgent 360")
    st.caption("Autonomous Healthcare AI Agent")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home",
        "🔬 Lab Report (PS #24)",
        "💊 Prescription (PS #22)",
        "📞 Follow-up Agent (PS #23)",
        "🚨 Alerts & Recovery",
    ])
    language = st.selectbox("Language / భాష / भाषा", ["English", "Telugu", "Hindi"])
    st.divider()
    st.caption("KLH HackWithAI 2026 | Feb 27-28")

# ═══════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏥 MedAgent 360")
    st.subheader("Autonomous Healthcare AI Agent for Rural India")
    st.info("*650 million Indians can't understand their medical reports. MedAgent 360 changes that.*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modules", "3", "PS #22+23+24")
    c2.metric("Languages", "3", "EN/Telugu/Hindi")
    c3.metric("Build Time", "24 hrs", "Hackathon")
    c4.metric("Target Score", "97/100", "Projected")

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔬 Lab Report (A)")
        st.markdown("Upload blood test PDF → AI flags abnormal values → Summary in your language + voice")
    with col2:
        st.markdown("### 💊 Prescription (B)")
        st.markdown("Photo your prescription → OCR extracts medicines → Voice instructions in Telugu/Hindi")
    with col3:
        st.markdown("### 📞 Follow-up (C)")
        st.markdown("Daily WhatsApp check-ins → LLM symptom analysis → Auto-alerts doctor if critical")


# ═══════════════════════════════════════════════════════════════════
# MODULE A — LAB REPORT
# ═══════════════════════════════════════════════════════════════════
elif page == "🔬 Lab Report (PS #24)":
    st.title("🔬 Lab Report Intelligence")
    st.caption("Upload a blood test PDF → AI explains it in your language with voice output")

    uploaded = st.file_uploader("Upload Lab Report PDF", type=["pdf"])

    if uploaded:
        st.success(f"📄 {uploaded.name} ({uploaded.size/1024:.1f} KB)")
        col1, col2 = st.columns([3,1])
        with col2:
            analyze = st.button("🔍 Analyze Report", type="primary", use_container_width=True)

        if analyze:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            with st.spinner("🤖 Analysing with Gemini AI..."):
                try:
                    from lab_report.rag_pipeline import run_full_pipeline
                    result = run_full_pipeline(tmp_path, language=language)

                    # Patient header
                    info = result["patient_info"]
                    st.subheader(f"👤 {info['name']}  |  Age: {info['age']}  |  {info['date']}")
                    st.caption(f"Extraction method: `{result.get('extraction_method','—')}`")

                    # Stats bar
                    s = result["stats"]
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Total Tests", s["total"])
                    c2.metric("✅ Normal", s["normal"])
                    c3.metric("⚠️ Abnormal", s["abnormal"])
                    c4.metric("🚨 Critical", s["critical"], delta_color="inverse")

                    # Critical alert banner
                    if result["critical_flags"]:
                        st.error(f"🚨 **CRITICAL VALUES DETECTED:** {', '.join(result['critical_flags'])} — Please see a doctor immediately!")
                        expls = result.get("critical_explanations", {})
                        for test, explanation in expls.items():
                            st.warning(f"**{test}:** {explanation}")

                    # Results table
                    st.subheader("📊 Test Results")
                    df = pd.DataFrame(result["classified_values"])
                    if not df.empty:
                        display_cols = [c for c in ["risk_icon","test","value","unit","status","benchmark_min","benchmark_max","benchmark_unit","deviation_pct"] if c in df.columns]
                        df_display = df[display_cols].rename(columns={
                            "risk_icon":"", "test":"Test","value":"Your Value","unit":"Unit",
                            "status":"Status","benchmark_min":"Normal Min","benchmark_max":"Normal Max",
                            "benchmark_unit":"Normal Unit","deviation_pct":"Deviation %"
                        })

                        def highlight(row):
                            colors = {"CRITICAL":"background-color:#ff4444;color:white",
                                      "HIGH":"background-color:#ff9900;color:white",
                                      "LOW":"background-color:#ffcc00",
                                      "NORMAL":"background-color:#00cc44;color:white"}
                            return [colors.get(row.get("Status",""), "") if col=="Status" else "" for col in df_display.columns]

                        st.dataframe(df_display.style.apply(highlight, axis=1), use_container_width=True, height=400)

                    # AI Summary
                    st.subheader(f"🤖 AI Summary ({language})")
                    st.info(result["summary"])

                    # Audio player
                    st.subheader("🔊 Listen to Summary")
                    with st.spinner("Generating audio..."):
                        try:
                            from lab_report.voice import generate_audio
                            audio_path = generate_audio(result["summary"], language)
                            with open(audio_path, "rb") as f:
                                st.audio(f.read(), format="audio/mp3")
                            os.unlink(audio_path)
                        except Exception as e:
                            st.warning(f"Audio unavailable: {e}")

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.exception(e)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════
# MODULE B — PRESCRIPTION
# ═══════════════════════════════════════════════════════════════════
elif page == "💊 Prescription (PS #22)":
    st.title("💊 Prescription Parser")
    st.caption("Upload a prescription photo → AI extracts medicines → Voice instructions in your language")

    uploaded = st.file_uploader("Upload Prescription Image", type=["jpg","jpeg","png","bmp","tiff"])
    patient_phone = st.text_input("Patient WhatsApp (optional, for reminders)", placeholder="+91XXXXXXXXXX")
    schedule = st.checkbox("Schedule medication reminders via WhatsApp")

    if uploaded:
        col1, col2 = st.columns([1,1])
        with col1:
            st.image(uploaded, caption="Uploaded Prescription", use_column_width=True)
        with col2:
            parse = st.button("💊 Parse Prescription", type="primary", use_container_width=True)

        if parse:
            ext = os.path.splitext(uploaded.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            with st.spinner("🔤 Running OCR + AI extraction..."):
                try:
                    from prescription.parser import run_prescription_pipeline
                    result = run_prescription_pipeline(
                        tmp_path, language=language,
                        patient_phone=patient_phone, schedule=schedule and bool(patient_phone)
                    )

                    st.success(f"✅ Found **{result['medicine_count']} medicines** | OCR confidence: {result['ocr_confidence']:.1f}%")
                    st.caption(f"Detection mode: `{result.get('detection_mode','—')}`")

                    # Medicine cards
                    st.subheader("💊 Medicines Found")
                    for i, med in enumerate(result["medicines"]):
                        with st.expander(f"💊 {med.get('medicine','Medicine')} ({med.get('dosage','—')})", expanded=i<3):
                            c1,c2,c3 = st.columns(3)
                            c1.markdown(f"**Frequency:** {med.get('frequency','—')}")
                            c2.markdown(f"**Timing:** {med.get('timing','—')}")
                            c3.markdown(f"**Duration:** {med.get('duration','—')}")
                            if med.get("special_notes"):
                                st.info(f"📝 {med['special_notes']}")
                            if med.get("instruction_translated"):
                                st.markdown(f"**{language}:** {med['instruction_translated']}")
                            # Audio
                            if med.get("audio_path") and os.path.exists(med["audio_path"]):
                                with open(med["audio_path"], "rb") as f:
                                    st.audio(f.read(), format="audio/mp3")

                    # OCR text expander
                    with st.expander("📄 Raw OCR Text"):
                        st.code(result["ocr_text"])

                except Exception as e:
                    st.error(f"❌ Parsing failed: {e}")
                    st.exception(e)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════
# MODULE C — FOLLOW-UP AGENT
# ═══════════════════════════════════════════════════════════════════
elif page == "📞 Follow-up Agent (PS #23)":
    st.title("📞 Autonomous Follow-up Agent")
    st.caption("Enroll patients → Automated WhatsApp check-ins → LLM analysis → Doctor alerts")

    tab1, tab2, tab3 = st.tabs(["👤 Enroll Patient", "📨 Send Check-in", "🧪 Test Analysis"])

    with tab1:
        st.subheader("Enroll a New Patient")
        with st.form("enroll_form"):
            e_phone = st.text_input("Patient WhatsApp", placeholder="whatsapp:+91XXXXXXXXXX")
            e_name  = st.text_input("Patient Name")
            e_lang  = st.selectbox("Preferred Language", ["English","Telugu","Hindi"])
            e_doc   = st.text_input("Doctor Phone (for alerts)", placeholder="+91XXXXXXXXXX")
            submitted = st.form_submit_button("✅ Enroll Patient", type="primary")
        if submitted:
            if e_phone and e_name:
                from followup.agent import enroll_patient, init_database
                init_database()
                result = enroll_patient(e_phone, e_name, e_lang, e_doc)
                if result.get("success"):
                    st.success(f"✅ Enrolled: {e_name} ({e_phone})")
                else:
                    st.error(f"Failed: {result.get('error')}")
            else:
                st.warning("Phone and name are required")

    with tab2:
        st.subheader("Send a Manual Check-in")
        m_phone = st.text_input("Patient WhatsApp", key="manual_phone", placeholder="whatsapp:+91XXXXXXXXXX")
        m_name  = st.text_input("Patient Name", key="manual_name")
        m_lang  = st.selectbox("Language", ["English","Telugu","Hindi"], key="manual_lang")
        if st.button("📨 Send Now", type="primary"):
            if m_phone:
                from followup.agent import send_checkin_message
                result = send_checkin_message(m_phone, m_name or "Patient", m_lang)
                if result.get("success"):
                    st.success(f"✅ Check-in sent! SID: {result.get('message_sid')}")
                else:
                    st.error(f"Failed: {result.get('error')}")
            else:
                st.warning("Enter a phone number")

    with tab3:
        st.subheader("🧪 Test Symptom Analysis (no Twilio needed)")
        t_phone = st.text_input("Patient Phone (demo)", value="+919999999999")
        t_response = st.text_area("Simulate patient WhatsApp reply:", 
            value="I have severe chest pain and difficulty breathing, pain level 9/10. Started 2 hours ago.")
        if st.button("🔍 Analyze Symptoms", type="primary"):
            with st.spinner("Analysing with Gemini..."):
                try:
                    from followup.agent import init_database, analyze_symptoms
                    init_database()
                    analysis = analyze_symptoms(t_phone, t_response)

                    sev = analysis.get("severity","UNKNOWN")
                    icon = analysis.get("severity_icon","❓")
                    color = analysis.get("severity_color","#aaa")

                    st.markdown(f"### {icon} Severity: **{sev}**")
                    st.markdown(f"**Reasoning:** {analysis.get('reasoning','')}")
                    if analysis.get("symptoms"):
                        st.markdown(f"**Symptoms identified:** {', '.join(analysis['symptoms'])}")
                    if analysis.get("pain_level") is not None:
                        st.metric("Pain Level", f"{analysis['pain_level']}/10")
                    if sev == "CRITICAL":
                        st.error("🚨 In production, this would trigger an immediate doctor alert!")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.exception(e)


# ═══════════════════════════════════════════════════════════════════
# ALERTS & RECOVERY
# ═══════════════════════════════════════════════════════════════════
elif page == "🚨 Alerts & Recovery":
    st.title("🚨 Alerts & Recovery Tracker")

    tab1, tab2 = st.tabs(["🚨 Doctor Alerts", "📈 Recovery Timeline"])

    with tab1:
        st.subheader("Recent Doctor Alerts")
        try:
            import sqlite3
            from scripts.config import config
            from followup.agent import init_database
            init_database()
            conn = sqlite3.connect(config.SQLITE_DB_PATH)
            rows = conn.execute("SELECT patient_phone,severity,sent_at,alert_message FROM doctor_alerts ORDER BY sent_at DESC LIMIT 20").fetchall()
            conn.close()
            if rows:
                df = pd.DataFrame(rows, columns=["Patient","Severity","Sent At","Message"])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No alerts yet. They will appear here when Module C detects critical symptoms.")
        except Exception as e:
            st.warning(f"Could not load alerts: {e}")

    with tab2:
        st.subheader("Patient Recovery Timeline")
        r_phone = st.text_input("Enter patient phone", placeholder="+91XXXXXXXXXX")
        if r_phone:
            try:
                from followup.agent import get_recovery_timeline
                timeline = get_recovery_timeline(r_phone)
                if timeline:
                    df = pd.DataFrame(timeline)
                    st.line_chart(df.set_index("date")["pain_level"].dropna())
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No recovery data for this patient yet.")
            except Exception as e:
                st.warning(f"Could not load timeline: {e}")
