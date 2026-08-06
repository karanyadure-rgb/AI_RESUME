from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def extract_jd_data(jd_text):
    # FIXED: Changed "resume" to "job description"
    prompt = f"""Extract structured data from the job description below.
    
    Return ONLY valid JSON (no markdown, no explanation, no code fences) in this exact shape:
    {{
      "job_title": "string",
      "required_skills": ["skill1", "skill2", ...],
      "responsibilities": ["responsibility1", "responsibility2", ...],
      "qualifications": ["qualification1", "qualification2", ...]
    }}
    
    JOB description:
    \"\"\"
    {jd_text}
    \"\"\"
    """
    try:
        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            # ADDED: System prompt to strictly enforce JSON behavior
            messages=[
                {"role": "system", "content": "You are a precise data extraction API that outputs strict JSON without any conversational text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0,
            response_format={"type": "json_object"}
        )
        jd_raw = response.choices[0].message.content.strip()
    except Exception as e:
            raise RuntimeError(f"JD_extraction failed: {e}") from e
    
    # IMPROVED: Safety net to catch both "```" and "```json"
    if jd_raw.startswith("```"):
        jd_raw = jd_raw.strip("`")
        if jd_raw.lower().startswith("json"):
            jd_raw = jd_raw[4:].strip()
        jd_raw = jd_raw.split("\n", 1)[1] if "\n" in jd_raw else jd_raw

    try:
        return json.loads(jd_raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {jd_raw[:300]}") from e

def load_document(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found!")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    if not text.strip():
        print("Error: Document is empty!")
        return None
        
    print("Processing extract_jd_data... ", end="", flush=True)
    result = extract_jd_data(text)
    print("Done.")
    return result

def paste_text() -> dict | None:
    print("Paste the job description below. Press Enter on an empty line when done:")
    lines = []
    while True:
        line = input()
        if line == '':
            break
        lines.append(line)

    text = "\n".join(lines).strip()

    if not text.strip():
        print("Error: You did not paste a job description.")
        return None
        
    print("Processing extract_jd_data... ", end="", flush=True)
    result = extract_jd_data(text)
    print("Done.")
    return result

if __name__ == "__main__":
    print("--- JOB DESCRIPTION EXTRACTOR ---")
    while True:
        print("\nOptions:")
        print("1. Enter a document file path")
        print("2. Paste the job description text directly")
        print("3. Exit")
        choice_raw = input("Enter your choice (1, 2, or 3): ")
        
        if not choice_raw.isdigit():
            print("Please enter a valid number (1, 2, or 3).")
            continue        

        choice = int(choice_raw)
        data = None

        if choice == 1:
            file_name = input("Enter the file path: ")
            data = load_document(file_name)

        elif choice == 2:
            data = paste_text()

        elif choice == 3:
            print("Exiting...")
            break

        else:
            print("Enter a valid choice number (1, 2, or 3).")

        if data:
            with open("jd_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)