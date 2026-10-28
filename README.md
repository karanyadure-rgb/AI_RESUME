# 📄 AI Resume Tailor & Matcher

An AI-powered toolkit designed to generate polished resumes from raw data, analyze them against target job descriptions, and provide honest, data-backed rewrites. Powered by local semantic embeddings and the Groq API for blazing-fast LLM inference.

## ✨ Features
*   **Intelligent Resume Generation:** Automatically generate a professionally formatted resume from raw, unformatted details.
*   **Structured Data Extraction:** Converts free-form resume text and job descriptions (JDs) into strictly formatted JSON for accurate comparison.
*   **Advanced ATS Scoring:** Calculates a hybrid match score using deterministic keyword matching and RAG-style semantic similarity.
*   **Honest AI Rewrites:** Suggests intelligent resume bullet rewrites to better match JD requirements, strictly utilizing *only* the candidate's actual provided experience.
*   **Interactive UI:** Fully featured Streamlit web interface with dedicated workflows for matching against jobs or building new resumes.

---

## 🛠️ Tech Stack
*   **Language:** Python 3.10+
*   **Frontend:** Streamlit
*   **LLM Provider:** Groq API (`openai/gpt-oss-20b`)
*   **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`) for local, cost-free semantic scoring

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/ai-resume-tailor.git](https://github.com/yourusername/ai-resume-tailor.git)
cd ai-resume-tailor
2. Install dependencies
Ensure you have the required packages installed:

Bash
pip install openai python-dotenv streamlit sentence-transformers
3. Environment Variables
This project relies on the Groq API for LLM calls. Create a .env file in the root directory and add your API key:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
💻 Usage
Streamlit Web Interface (Recommended)
The easiest way to use the toolkit is via the Streamlit application:

Bash
streamlit run app.py
Tab 1: Match Against a Job: Paste your raw resume and a target job description. The app will calculate your ATS score, identify missing keywords, and suggest honest bullet rewrites.

Tab 2: Build a New Resume: Fill in a form with your details (e.g., target role, skills, experience) to generate a professional .txt resume.

CLI Tools
You can also run the modular scripts individually via the command line:

Extract Resume Data: Converts a .txt resume into resume_data.json.

Bash
python resume_extractor.py
Extract JD Data: Converts JD text into jd_data.json.

Bash
python jd_extractor.py
Score Match: Compares the extracted JSON files and generates match_report.json.

Bash
python scorer.py
Generate Suggestions: Creates actionable rewrites and outputs to suggestions.json.

Bash
python suggester.py
Generate Resume: Quickly build a resume from terminal prompts.

Bash
python resume_generator.py
📂 Project Architecture
app.py: The main Streamlit frontend that integrates all underlying pipelines into a unified UI.

resume_generator.py: Prompts llama-3.1-8b-instant to create a 3-line professional summary, action-verb experience bullets, and structured skill/education sections from raw user input.

resume_extractor.py & jd_extractor.py: Utility scripts configured with strict JSON system prompts to extract structured lists of skills, experience, and qualifications without conversational text.

scorer.py: Flattens resume data into chunks and uses pure-Python cosine similarity against JD requirements to generate keyword and semantic scores.

suggester.py: Analyzes the gaps identified by the scorer and uses the LLM to verify if missing skills are genuinely absent or just phrased differently.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📝 License
This project is licensed under the MIT License.
