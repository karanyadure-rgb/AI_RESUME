import streamlit as st
import json
from pathlib import Path

# --- IMPORT PIPELINE FUNCTIONS DIRECTLY ---
from jd_extractor import extract_jd_data
from resume_extractor import extract_resume_data
from scorer import score as score_match
from suggester import generate_suggestions
from resume_generator import resume_generator, _validate

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Resume ",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
    [data-testid="stHeader"]::before {
        content: "KY";
        position: absolute;
        left: 60px; /* Positions it right next to the sidebar toggle > */
        top: 16px;
        font-size: 24px;
        font-weight: 900;
        color: inherit;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def run_step(step_description: str, progress_placeholder, func, *args, **kwargs):
    progress_placeholder.info(f"⏳ **Step:** {step_description}")
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"❌ **Error during: {step_description}**")
        with st.expander("View Error Details"):
            st.code(str(e), language="text")
        st.stop()


# --- HEADER ---
st.title("📄 AI Resume Tailor")
st.markdown("Match your resume against a job description, or build a new resume from scratch.")
st.divider()

tab_match, tab_generate = st.tabs(["🔍 Match Against a Job", "📝 Build a New Resume"])

# =====================================================================
# TAB 1: MATCH AN EXISTING RESUME AGAINST A JOB DESCRIPTION
# =====================================================================
with tab_match:
    st.subheader("Compare your actual experience against a Job Description")
    st.markdown("Identify gaps and get honest, data-backed bullet rewrites.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Your Resume")
        resume_text = st.text_area(
            "Paste your raw resume text here:",
            height=300,
            placeholder="Paste your skills, summary, and experience bullets here...",
            key="match_resume_text",
        )

    with col2:
        st.subheader("2. Job Description")
        jd_text = st.text_area(
            "Paste the Job Description here:",
            height=300,
            placeholder="Paste the target role's responsibilities and requirements here...",
            key="match_jd_text",
        )

    analyze_button = st.button("Run AI Analysis 🚀", type="primary", use_container_width=True)

    if analyze_button:
        if not resume_text.strip() or not jd_text.strip():
            st.warning("⚠️ Please paste both your resume and the job description to continue.")
            st.stop()

        st.divider()
        st.subheader("⚙️ Processing Pipeline")
        progress_ui = st.empty()

        embedding_model = run_step("Loading embedding model", progress_ui, get_embedding_model)

        jd_data = run_step("Extracting Job Requirements", progress_ui, extract_jd_data, jd_text)
        resume_data = run_step("Parsing Resume Data", progress_ui, extract_resume_data, resume_text)
        match_result = run_step(
            "Calculating Match Scores", progress_ui, score_match, resume_data, jd_data, embedding_model
        )
        suggestions = run_step(
            "Generating Honest Rewrites", progress_ui, generate_suggestions, resume_data, jd_data, match_result
        )

        progress_ui.success("✅ Analysis Complete!")

        Path("resume_data.json").write_text(json.dumps(resume_data, indent=2), encoding="utf-8")
        Path("jd_data.json").write_text(json.dumps(jd_data, indent=2), encoding="utf-8")
        Path("match_report.json").write_text(json.dumps(match_result, indent=2), encoding="utf-8")
        Path("suggestions.json").write_text(json.dumps(suggestions, indent=2), encoding="utf-8")

        st.divider()

        st.header(f"📊 Match Report: {suggestions.get('job_title', 'Target Role')}")

        overall_score = match_result.get("final_score", 0) * 100
        score_col, metric_col1, metric_col2 = st.columns([2, 1, 1])

        with score_col:
            st.metric(label="Overall Match Score", value=f"{overall_score:.0f}%")
            st.progress(overall_score / 100)

        with metric_col1:
            matched_reqs = len(match_result.get('covered_requirements', []))
            unmatched_reqs = len(match_result.get('unmatched_requirements', []))
            total_reqs = matched_reqs + unmatched_reqs
            st.metric("Requirements Met", f"{matched_reqs} / {total_reqs}")

        with metric_col2:
            matched_kws = len(match_result.get('matched_keywords', []))
            missing_kws = len(match_result.get('missing_keywords', []))
            total_kws = matched_kws + missing_kws
            st.metric("Keywords Found", f"{matched_kws} / {total_kws}")

        st.divider()

        st.subheader("💡 Smart Resume Rewrites")
        st.markdown("Based *only* on your actual experience, here is how you can better target this role.")

        result_tab1, result_tab2 = st.tabs(["🎯 Requirement Gaps", "🔑 Missing Keywords"])

        with result_tab1:
            req_suggestions = suggestions.get("requirement_suggestions", [])
            if not req_suggestions:
                st.info("Great job! You meet all listed requirements.")
            else:
                for item in req_suggestions:
                    with st.expander(f"Gap: {item['requirement']}"):
                        if item.get("has_real_match"):
                            st.success("**Honest Match Found in Resume!**")
                            st.markdown(f"**Suggested Rewrite:** \n> {item.get('suggested_bullet')}")
                            st.caption(f"*AI Reasoning:* {item.get('reasoning')}")
                        else:
                            st.error("**No Honest Match Found.**")
                            st.caption(f"*AI Reasoning:* {item.get('reasoning')}")
                            st.markdown("*Advice: Do not add a bullet for this unless you have actual experience.*")

        with result_tab2:
            kw_suggestions = suggestions.get("keyword_suggestions", [])
            if not kw_suggestions:
                st.info("Great job! All required keywords were found.")
            else:
                for item in kw_suggestions:
                    with st.expander(f"Missing Keyword: {item['keyword']}"):
                        if item.get("genuinely_has_skill"):
                            st.success("**Skill exists under different phrasing.**")
                            st.markdown(f"> {item.get('note')}")
                        else:
                            st.error("**Genuine Gap.**")
                            st.markdown(f"> {item.get('note')}")

# =====================================================================
# TAB 2: BUILD A NEW RESUME FROM RAW DETAILS
# =====================================================================
with tab_generate:
    st.subheader("Build a resume from scratch")
    st.markdown("Fill in your raw details below — the AI turns them into a polished, ready-to-submit resume.")

    with st.form("resume_generator_form"):
        gen_col1, gen_col2 = st.columns(2)
        with gen_col1:
            name = st.text_input("Full Name")
            college = st.text_input("College Name")
            course = st.text_input("Course (e.g., Bachelor of Information Technology)")
            cgpa = st.text_input("CGPA / Marks (e.g., 8.5 CGPA or 85%)")
        with gen_col2:
            skills = st.text_area("Skills (comma-separated, e.g., Python, Java, AWS, Gen AI)", height=100)
            job_target = st.text_input("Target Job Role (e.g., Data Engineer in Pune)")
            
        experience = st.text_area(
            "Experience / Projects (e.g., Online Voting System using Java Swing)", height=100
        )
        certifications = st.text_area(
            "Certifications (optional, e.g., AWS Certified Developer, Meta Frontend)", height=80
        )
        achievements = st.text_area(
            "Achievements (optional, e.g., 1st Rank in College Hackathon, Published IEEE Paper)", height=80
        )
        
        generate_clicked = st.form_submit_button("Generate Resume ✨", type="primary", use_container_width=True)

    if generate_clicked:
        data = {
            "name": name.strip(),
            "college": college.strip(),
            "course": course.strip(),
            "cgpa": cgpa.strip(),
            "skills": skills.strip(),
            "experience": experience.strip(),
            "certifications": certifications.strip(),
            "achievements": achievements.strip(),
            "job_target": job_target.strip(),
        }

        try:
            _validate(data)
        except ValueError as e:
            st.warning(f"⚠️ {e}")
            st.stop()

        with st.spinner("Generating your resume..."):
            try:
                generated_resume = resume_generator(data)
            except Exception as e:
                st.error("❌ Resume generation failed.")
                with st.expander("View Error Details"):
                    st.code(str(e), language="text")
                st.stop()

        st.success("✅ Resume generated!")
        st.text_area("Your generated resume", value=generated_resume, height=400)

        safe_name = (data["name"].replace(" ", "_") or "candidate")
        st.download_button(
            "⬇️ Download as .txt",
            data=generated_resume,
            file_name=f"{safe_name}_resume.txt",
            mime="text/plain",
            use_container_width=True,
        )