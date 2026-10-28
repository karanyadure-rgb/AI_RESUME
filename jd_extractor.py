
import json
import os

from llm_client import get_client, MODEL


def extract_jd_data(jd_text: str) -> dict:
    """Extract structured information from a job description."""

    if not jd_text or not jd_text.strip():
        raise ValueError("Job description is empty.")

    client = get_client()

    prompt = f"""
Extract information from the job description below.

Return one valid JSON object with exactly these keys:
- job_title: string
- required_skills: array of strings
- responsibilities: array of strings
- qualifications: array of strings

Use an empty string or empty array when information is unavailable.
Do not invent information. Return JSON only.

Example:
{{
    "job_title": "Python Developer",
    "required_skills": ["Python", "SQL"],
    "responsibilities": ["Develop applications"],
    "qualifications": ["Relevant degree or equivalent experience"]
}}

JOB DESCRIPTION:
<job_description>
{jd_text}
</job_description>
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract job description data accurately. "
                        "Return a valid JSON object only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content

        if not raw or not raw.strip():
            raise ValueError("The model returned an empty response.")

        data = json.loads(raw)

        required_keys = [
            "job_title",
            "required_skills",
            "responsibilities",
            "qualifications",
        ]

        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing JSON field: {key}")

        if not isinstance(data["job_title"], str):
            raise ValueError("job_title must be a string.")

        for key in required_keys[1:]:
            if not isinstance(data[key], list):
                raise ValueError(f"{key} must be a list.")

        return data

    except Exception as e:
        raise RuntimeError(f"JD extraction failed: {e}") from e


def load_document(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found!")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print("Error: Document is empty!")
        return None

    print("Processing job description...", flush=True)
    result = extract_jd_data(text)
    print("Done.")
    return result


def paste_text() -> dict | None:
    print(
        "Paste the job description below. "
        "Press Enter on an empty line when done:"
    )

    lines = []

    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    text = "\n".join(lines).strip()

    if not text:
        print("Error: You did not paste a job description.")
        return None

    print("Processing job description...", flush=True)
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

        choice_raw = input("Enter your choice (1, 2, or 3): ").strip()

        if not choice_raw.isdigit():
            print("Please enter a valid number (1, 2, or 3).")
            continue

        choice = int(choice_raw)
        data = None

        try:
            if choice == 1:
                file_name = input("Enter the file path: ").strip()
                data = load_document(file_name)

            elif choice == 2:
                data = paste_text()

            elif choice == 3:
                print("Exiting...")
                break

            else:
                print("Enter a valid choice number (1, 2, or 3).")
                continue

            if data:
                with open("jd_data.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print("Saved extracted data to jd_data.json")

        except Exception as e:
            print(f"Error: {e}")
