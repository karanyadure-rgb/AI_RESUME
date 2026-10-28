import json
import os

from llm_client import get_client, MODEL


def extract_resume_data(resume_text: str) -> dict:
    """Extract structured information from a resume using Groq."""

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty.")

    client = get_client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured information from resumes. "
                        "Return only a valid JSON object. "
                        "Do not include Markdown or explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
    Extract information from the resume below.

    Return exactly these four keys:
    - summary: a short string
    - skills: an array of strings
    - experience_bullets: an array of concise strings
    - education: an array of strings

    Rules:
    1. Use only information stated in the resume.
    2. Use an empty string or empty array when information is unavailable.
    3. Keep the summary short and each item concise.
    4. Do not repeat the entire resume.
    5. Return one complete JSON object.

    Required JSON structure:
    {{
        "summary": "Short professional summary",
        "skills": ["Python", "SQL"],
        "experience_bullets": ["Developed a Python application"],
        "education": ["Diploma in Computer Engineering"]
    }}

    Resume:
    <resume>
    {resume_text}
    </resume>
    """,
                },
            ],
            temperature=0,
            max_completion_tokens=4096,
            response_format={"type": "json_object"},
            reasoning_effort="low",
        )

        raw = response.choices[0].message.content

        if not raw:
            raise ValueError(
                "The AI returned an empty response. Try a shorter resume."
            )

        resume_data = json.loads(raw)

        required_keys = [
            "summary",
            "skills",
            "experience_bullets",
            "education",
        ]

        for key in required_keys:
            if key not in resume_data:
                raise ValueError(
                    f"Resume extraction is missing the '{key}' field."
                )

        if not isinstance(resume_data["summary"], str):
            raise ValueError("'summary' must be a string.")

        for key in ["skills", "experience_bullets", "education"]:
            if not isinstance(resume_data[key], list):
                raise ValueError(f"'{key}' must be a list.")

        return resume_data

    except json.JSONDecodeError as exc:
        raise ValueError(
            "The AI returned invalid JSON. Try again or use a shorter resume."
        ) from exc

    except Exception as exc:
        # Preserve the original error so the app can display its details.
        raise RuntimeError(f"Resume extraction failed: {exc}") from exc


if __name__ == "__main__":
    print("--- RESUME EXTRACTOR ---")
    path = input("Path to a resume .txt file: ").strip().strip('"')

    if not os.path.isfile(path):
        print(f"Error: File '{path}' not found!")
    else:
        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            print("Extracting resume information...")
            data = extract_resume_data(text)

            output_file = "resume_data.json"

            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)

            print(f"Success! Results saved to {output_file}")
            print(json.dumps(data, indent=2, ensure_ascii=False))

        except Exception as exc:
            print(f"Error: {exc}")

