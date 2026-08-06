import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

MODEL = "llama-3.1-8b-instant"


def _call_model(prompt: str, max_tokens: int) -> dict:
    """Shared call + parse logic, same pattern as jd_extractor.py /
    resume_extractor.py: JSON mode, temperature 0, strip stray fences."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise RuntimeError(f"Suggestion call failed: {e}") from e

    raw = response.choices[0].message.content.strip()
    
    # Strip markdown code fences if the model accidentally includes them
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {raw[:300]}") from e


def _resume_context_blob(resume_data: dict) -> str:
    """Everything real about the candidate, handed to the model as its
    ONLY source of truth — this is what keeps it from inventing things."""
    parts = []
    if resume_data.get("summary"):
        parts.append(f"Summary: {resume_data['summary']}")
    if resume_data.get("skills"):
        parts.append("Skills: " + ", ".join(resume_data["skills"]))
    if resume_data.get("experience_bullets"):
        parts.append("Experience bullets:\n- " + "\n- ".join(resume_data["experience_bullets"]))
    if resume_data.get("education"):
        parts.append("Education: " + ", ".join(resume_data["education"]))
    return "\n\n".join(parts)


def suggest_for_requirement(requirement: str, resume_context: str) -> dict:
    """For one unmatched JD requirement: is there real resume experience
    that could be honestly reworded to address it?"""
    prompt = f"""You are helping rewrite resume bullets to better match a job requirement.

RULES (do not break these):
- Only use experience that actually appears in the candidate's resume below.
- Never invent tools, skills, metrics, or experience the candidate doesn't have.
- If nothing in the resume honestly relates to this requirement, say so — do not fabricate a bullet.

Candidate's actual resume content:
\"\"\"
{resume_context}
\"\"\"

Job requirement not currently well-matched:
\"\"\"
{requirement}
\"\"\"

Return ONLY valid JSON (no markdown, no code fences) in this exact shape:
{{
  "has_real_match": true or false,
  "suggested_bullet": "a rewritten/reframed resume bullet using ONLY real resume content, or empty string if has_real_match is false",
  "reasoning": "one sentence explaining the connection, or why there's no honest match"
}}
"""
    return _call_model(prompt, max_tokens=400)


def suggest_for_keyword(keyword: str, resume_context: str) -> dict:
    """For a missing required skill: is it genuinely absent, or just
    phrased differently somewhere in the resume?"""
    prompt = f"""A job requires the skill "{keyword}". Check the candidate's actual resume
below and decide honestly.

RULES:
- Only say yes if the resume genuinely shows this skill (even under different wording).
- If the resume does not show it, say so plainly — do not suggest adding a skill they don't have.

Candidate's actual resume content:
\"\"\"
{resume_context}
\"\"\"

Return ONLY valid JSON (no markdown, no code fences):
{{
  "genuinely_has_skill": true or false,
  "note": "one sentence: either where it's evidenced and how to phrase it, or that it's a real gap worth learning"
}}
"""
    return _call_model(prompt, max_tokens=200)


def generate_suggestions(resume_data: dict, jd_data: dict, match_result: dict) -> dict:
    resume_context = _resume_context_blob(resume_data)

    requirement_suggestions = []
    # FIX: Properly indented the block inside this loop
    for req in match_result.get("unmatched_requirements", []):
        result = suggest_for_requirement(req, resume_context)
        requirement_suggestions.append({"requirement": req, **result})
        time.sleep(0.5) # Protects against rate limits

    keyword_suggestions = []
    # FIX: Properly indented the block and added sleep here as well
    for kw in match_result.get("missing_keywords", []):
        result = suggest_for_keyword(kw, resume_context)
        keyword_suggestions.append({"keyword": kw, **result})
        time.sleep(0.5) # Protects against rate limits

    return {
        "job_title": jd_data.get("job_title", "Unknown role"),
        "requirement_suggestions": requirement_suggestions,
        "keyword_suggestions": keyword_suggestions,
    }


def _print_suggestions(result: dict):
    print(f"\n=== Suggestions for: {result['job_title']} ===\n")

    print("--- Unmatched requirements ---")
    if not result["requirement_suggestions"]:
        print("  None — nice.")
    for item in result["requirement_suggestions"]:
        print(f"\nRequirement: {item['requirement']}")
        if item.get("has_real_match"):
            print(f"  Suggested bullet: {item.get('suggested_bullet')}")
            print(f"  Why: {item.get('reasoning')}")
        else:
            print(f"  No honest match in resume. {item.get('reasoning')}")

    print("\n--- Missing keywords ---")
    if not result["keyword_suggestions"]:
        print("  None — nice.")
    for item in result["keyword_suggestions"]:
        print(f"\nKeyword: {item['keyword']}")
        if item.get("genuinely_has_skill"):
            print(f"  {item.get('note')}")
        else:
            print(f"  Real gap: {item.get('note')}")

    print()


if __name__ == "__main__":
    resume_path = Path("resume_data.json")
    jd_path = Path("jd_data.json")
    report_path = Path("match_report.json")

    missing_files = [p.name for p in (resume_path, jd_path, report_path) if not p.exists()]
    if missing_files:
        print(f"Missing file(s): {', '.join(missing_files)}")
        print("Run resume_extractor.py, jd_extractor.py, then scorer.py first.")
        raise SystemExit(1)

    resume_data = json.loads(resume_path.read_text())
    jd_data = json.loads(jd_path.read_text())
    match_result = json.loads(report_path.read_text())

    print("Processing generate_suggestions... ", end="", flush=True)
    suggestions = generate_suggestions(resume_data, jd_data, match_result)
    print("Done.")

    Path("suggestions.json").write_text(json.dumps(suggestions, indent=2))