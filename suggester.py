
"""Generate honest resume-improvement suggestions using Groq."""

import json
import time
from pathlib import Path

from llm_client import get_client, MODEL


def _call_model(prompt: str, max_tokens: int = 1200) -> dict:
    """Call Groq and validate its JSON response."""

    client = get_client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a resume analysis assistant. "
                        "Return a single valid JSON object only. "
                        "Keep every answer concise. Never invent "
                        "skills or experience."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
            reasoning_effort="low",
        )

        raw = response.choices[0].message.content

        if not raw:
            raise ValueError("The AI returned an empty suggestion.")

        result = json.loads(raw)

        if not isinstance(result, dict):
            raise ValueError("The AI response must be a JSON object.")

        return result

    except Exception as exc:
        raise RuntimeError(f"Suggestion call failed: {exc}") from exc


def _resume_context_blob(resume_data: dict) -> str:
    """Build a compact context using only the candidate's real resume."""

    parts = []

    summary = resume_data.get("summary", "")
    if isinstance(summary, str) and summary.strip():
        parts.append(f"Summary: {summary.strip()}")

    skills = resume_data.get("skills", [])
    if isinstance(skills, list) and skills:
        parts.append("Skills: " + ", ".join(str(x) for x in skills))
    elif isinstance(skills, str) and skills.strip():
        parts.append("Skills: " + skills.strip())

    experience = resume_data.get("experience_bullets", [])
    if isinstance(experience, list) and experience:
        parts.append(
            "Experience:\n- "
            + "\n- ".join(str(x) for x in experience[:15])
        )

    education = resume_data.get("education", [])
    if isinstance(education, list) and education:
        parts.append("Education: " + "; ".join(str(x) for x in education))
    elif isinstance(education, str) and education.strip():
        parts.append("Education: " + education.strip())

    context = "\n\n".join(parts)

    if not context:
        raise ValueError("No usable resume information was provided.")

    return context


def suggest_for_requirement(requirement: str, resume_context: str) -> dict:
    """Suggest a truthful rewrite for one unmatched requirement."""

    prompt = f"""
Review the candidate's resume against this job requirement.

RESUME:
{resume_context}

JOB REQUIREMENT:
{requirement}

Use only evidence from the resume. Never invent experience, tools,
skills, or measurements. If no genuine match exists, return false
and leave suggested_bullet empty.

Return this JSON structure:
{{
  "has_real_match": false,
  "suggested_bullet": "",
  "reasoning": "Brief explanation"
}}

Keep reasoning to one short sentence.
"""

    result = _call_model(prompt, max_tokens=1200)

    if not isinstance(result.get("has_real_match"), bool):
        raise ValueError("Invalid has_real_match value in AI response.")

    result.setdefault("suggested_bullet", "")
    result.setdefault("reasoning", "")

    return result


def suggest_for_keyword(keyword: str, resume_context: str) -> dict:
    """Check whether a missing keyword is already evidenced by the resume."""

    prompt = f"""
Determine whether the resume genuinely demonstrates this skill:
{keyword}

RESUME:
{resume_context}

Do not assume the candidate has a skill without evidence.
If it is absent, identify it as a learning gap.

Return this JSON structure:
{{
  "genuinely_has_skill": false,
  "note": "One short sentence explaining the evidence or gap"
}}
"""

    result = _call_model(prompt, max_tokens=800)

    if not isinstance(result.get("genuinely_has_skill"), bool):
        raise ValueError("Invalid genuinely_has_skill value in AI response.")

    result.setdefault("note", "")
    return result


def generate_suggestions(
    resume_data: dict,
    jd_data: dict,
    match_result: dict,
) -> dict:
    """Generate suggestions for unmatched requirements and missing skills."""

    resume_context = _resume_context_blob(resume_data)

    requirement_suggestions = []

    for requirement in match_result.get("unmatched_requirements", [])[:10]:
        result = suggest_for_requirement(str(requirement), resume_context)

        requirement_suggestions.append({
            "requirement": requirement,
            **result,
        })

        time.sleep(0.5)

    keyword_suggestions = []

    for keyword in match_result.get("missing_keywords", [])[:10]:
        result = suggest_for_keyword(str(keyword), resume_context)

        keyword_suggestions.append({
            "keyword": keyword,
            **result,
        })

        time.sleep(0.5)

    return {
        "job_title": jd_data.get("job_title", "Unknown role"),
        "requirement_suggestions": requirement_suggestions,
        "keyword_suggestions": keyword_suggestions,
    }


def _print_suggestions(result: dict) -> None:
    print(f"\n=== Suggestions for: {result['job_title']} ===")

    print("\n--- Unmatched requirements ---")
    for item in result["requirement_suggestions"]:
        print(f"\nRequirement: {item['requirement']}")

        if item.get("has_real_match"):
            print(f"Suggested bullet: {item.get('suggested_bullet', '')}")

        print(f"Reason: {item.get('reasoning', '')}")

    if not result["requirement_suggestions"]:
        print("No unmatched requirements.")

    print("\n--- Missing keywords ---")
    for item in result["keyword_suggestions"]:
        print(f"\nKeyword: {item['keyword']}")
        print(f"Analysis: {item.get('note', '')}")

    if not result["keyword_suggestions"]:
        print("No missing keywords.")


if __name__ == "__main__":
    resume_path = Path("resume_data.json")
    jd_path = Path("jd_data.json")
    report_path = Path("match_report.json")

    missing_files = [
        path.name
        for path in (resume_path, jd_path, report_path)
        if not path.exists()
    ]

    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
        print("Run resume extraction, JD extraction, and scoring first.")
        raise SystemExit(1)

    try:
        resume_data = json.loads(resume_path.read_text(encoding="utf-8"))
        jd_data = json.loads(jd_path.read_text(encoding="utf-8"))
        match_result = json.loads(report_path.read_text(encoding="utf-8"))

        print(f"Using model: {MODEL}")
        print("Generating suggestions...")

        suggestions = generate_suggestions(
            resume_data,
            jd_data,
            match_result,
        )

        output_path = Path("suggestions.json")
        output_path.write_text(
            json.dumps(suggestions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        _print_suggestions(suggestions)
        print(f"\nSuggestions saved to {output_path.resolve()}")

    except Exception as exc:
        print(f"\nSuggestion generation failed: {exc}")
