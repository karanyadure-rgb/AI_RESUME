from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
) 

REQUIRED_FIELDS = ["name", "college", "course", "skills", "experience", "job_target"]

def _validate(data: dict):
    missing = [f for f in REQUIRED_FIELDS if not data.get(f)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

def resume_generator(data: dict):
    prompt = f"""You are a professional resume writer with 10+ years of experience.
    Create a polished resume from the raw details below.
    Improve the language and use professional English grammar.

    Name: {data['name']}
    College: {data['college']}
    Course: {data['course']}
    CGPA / Marks: {data.get('cgpa', 'N/A')}
    Skills: {data['skills']}
    Experience: {data['experience']}
    Certifications: {data.get('certifications', 'N/A')}
    Achievements: {data.get('achievements', 'N/A')}
    Target Role: {data['job_target']}

    Create these sections:
    - Professional Summary (3 lines)
    - Key Skills (bullet points)
    - Education (include College, Course, and Marks/CGPA if provided)
    - Experience & Projects (with action verbs)
    - Certifications (bullet points, if provided)
    - Achievements (bullet points, if provided)
    
    Format it cleanly and make it ready to submit.
    """
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200, # Increased token limit to allow for new sections
        temperature=0.7
    )
    return response.choices[0].message.content
    
if __name__ == "__main__":
    print("--- AI RESUME BUILDER ---")
    print("Enter Your Details:")
    
    data = {
        'name': input("Enter Full Name: "),
        'college': input("Enter College Name: "),
        'course': input("Enter Course Name (e.g., Bachelor of Information Technology): "),
        'cgpa': input("Enter Marks / CGPA (e.g., 8.5 CGPA or 85%): "),
        'skills': input("Enter Skills (e.g., Python, Java, AWS, Gen AI): "),
        'experience': input("Enter Your Experience or Projects (e.g., Online Voting System using Java Swing): "),
        'certifications': input("Enter Certifications (e.g., AWS Certified Developer, Meta Frontend): "),
        'achievements': input("Enter Achievements (e.g., 1st Rank in Hackathon, Published Paper): "),
        'job_target': input("Enter Target Job Role (e.g., Data Engineer in Pune): ")
    }

    try:
        _validate(data)
        print("Processing resume_generator... ", end="", flush=True)
        resume = resume_generator(data)
        print("Done.")
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    save_dir = "resume"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    safe_name = data["name"].replace(" ", "_")
    file_path = os.path.join(save_dir, f"{safe_name}_resume.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(resume)