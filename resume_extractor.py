import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "llama-3.1-8b-instant"

def extract_resume_data(resume_text: str) -> dict:
    """
    Turns free-form resume text into structured JSON:
    summary, skills, experience_bullets, education.
    This is what the matcher/scorer will operate on later.
    """
    prompt = f"""Extract structured data from the resume below.

    Return ONLY valid JSON (no markdown, no explanation, no code fences) in this exact shape:
    {{
      "summary": "string",
      "skills": ["skill1", "skill2", ...],
      "experience_bullets": ["bullet1", "bullet2", ...],
      "education": ["entry1", "entry2", ...]
    }}

    Resume:
    \"\"\"
    {resume_text}
    \"\"\"
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                # Added the system prompt to enforce strict JSON behavior
                {"role": "system", "content": "You are a precise data extraction API that outputs strict JSON without any conversational text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise RuntimeError(f"Resume extraction failed: {e}") from e

    raw = response.choices[0].message.content.strip()

    # Safety net in case the model wraps output in code fences anyway
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {raw[:300]}") from e


if __name__ == "__main__":
    print("--- RESUME EXTRACTOR ---")
    path = input("Path to a resume .txt file to extract: ")
    
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found!")
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        try:
            print("Processing extract_resume_data... ", end="", flush=True)
            data = extract_resume_data(text)
            print("Done.")
            
            output_file = "resume_data.json"
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, indent=2)
                
        except Exception as e:
            print(f"An error occurred: {e}")