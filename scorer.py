"""
scorer.py — Step 3 of the AI Resume + Job Matcher project.

Pure-Python + local-embedding comparison between resume_data (from
resume_extractor.py) and jd_data (from jd_extractor.py). No LLM API
calls here — that happens later in the "suggestion" agent step.

Produces:
  - keyword_score           : deterministic overlap between JD required_skills
                               and resume skills/text (your ATS-style number)
  - semantic_score           : RAG-style coverage check — for each JD
                               requirement, does any resume chunk match well?
  - final_score              : weighted combo of the two above
  - missing_keywords         : required skills not found anywhere in the resume
  - unmatched_requirements   : JD responsibilities/qualifications with no
                               strong semantic match in the resume

Expected input schemas
-----------------------
jd_data (from jd_extractor.py):
    {
        "job_title": str,
        "required_skills": [str, ...],
        "responsibilities": [str, ...],
        "qualifications": [str, ...]
    }

resume_data (from resume_extractor.py):
    {
        "summary": "string",
        "skills": [str, ...],
        "experience_bullets": [str, ...],
        "education": [str, ...]
    }
"""

import json
import re
from pathlib import Path

# ---- Config: matches resume_extractor.py's actual output field names ----
RESUME_SKILLS_KEYS = ["skills"]
RESUME_TEXT_KEYS = ["experience_bullets", "summary", "education"]

SEMANTIC_THRESHOLD = 0.45  # cosine similarity cutoff for "this counts as covered"
WEIGHT_KEYWORD = 0.5
WEIGHT_SEMANTIC = 0.5


def _normalize(term: str) -> str:
    return re.sub(r"[^a-z0-9+.# ]", "", term.lower()).strip()


def _flatten_resume_text(resume_data: dict) -> list:
    """Pull every resume bullet/sentence into a flat list of strings so we
    have something to embed and compare against JD requirements."""
    chunks = []
    for key in RESUME_TEXT_KEYS:
        val = resume_data.get(key)
        if not val:
            continue
        if isinstance(val, str):
            chunks.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, str):
                            chunks.append(v)
                        elif isinstance(v, list):
                            chunks.extend([b for b in v if isinstance(b, str)])
    return [c for c in chunks if c.strip()]


def _get_resume_skills(resume_data: dict) -> list:
    for key in RESUME_SKILLS_KEYS:
        if resume_data.get(key):
            return resume_data[key]
    return []


def keyword_score(resume_data: dict, jd_data: dict):
    """Deterministic, explainable ATS-style score: does the exact skill
    term show up anywhere in the resume's skills list or body text?"""
    required = jd_data.get("required_skills", [])
    resume_skills = _get_resume_skills(resume_data)
    normalized_skills = [_normalize(s) for s in resume_skills]
    resume_text_blob = _normalize(" ".join(_flatten_resume_text(resume_data)))

    matched, missing = [], []
    for skill in required:
        norm_skill = _normalize(skill)
        if not norm_skill:
            continue
        if norm_skill in normalized_skills or norm_skill in resume_text_blob:
            matched.append(skill)
        else:
            missing.append(skill)

    score = len(matched) / len(required) if required else 0.0
    return round(score, 3), matched, missing


def _load_model():
    # Local, free embeddings — no API cost for this step.
    # pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(texts, model=None):
    """Encode a list of strings into plain Python list vectors (no numpy
    objects exposed outside this function — easy to cache as JSON later)."""
    if model is None:
        model = _load_model()
    embeddings = model.encode(texts)
    return embeddings.tolist()


def cosine_similarity(vec1, vec2):
    """Pure-Python cosine similarity — no numpy needed for the math."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = sum(a * a for a in vec1) ** 0.5
    norm_vec2 = sum(b * b for b in vec2) ** 0.5
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def semantic_score(resume_data: dict, jd_data: dict, model=None):
    """RAG-style gap detection: for each JD responsibility/qualification,
    retrieve the best-matching resume chunk and check if it clears the
    similarity threshold. This catches paraphrased matches that keyword
    overlap misses (e.g. JD says 'led cross-functional teams', resume
    says 'coordinated with design and backend engineers')."""
    requirements = jd_data.get("responsibilities", []) + jd_data.get("qualifications", [])
    resume_chunks = _flatten_resume_text(resume_data)

    if not requirements or not resume_chunks:
        return 0.0, [], [(r, 0.0) for r in requirements]

    if model is None:
        model = _load_model()

    req_vectors = get_embedding(requirements, model)
    chunk_vectors = get_embedding(resume_chunks, model)

    covered, gaps = [], []
    for req, req_vec in zip(requirements, req_vectors):
        best_score = -1.0
        best_chunk = None
        for chunk, chunk_vec in zip(resume_chunks, chunk_vectors):
            sim = cosine_similarity(req_vec, chunk_vec)
            if sim > best_score:
                best_score = sim
                best_chunk = chunk

        if best_score >= SEMANTIC_THRESHOLD:
            covered.append((req, best_score, best_chunk))
        else:
            gaps.append((req, best_score))

    score = len(covered) / len(requirements)
    return round(score, 3), covered, gaps


def score(resume_data: dict, jd_data: dict, model=None) -> dict:
    kw_score, matched_kw, missing_kw = keyword_score(resume_data, jd_data)
    sem_score, covered_reqs, gap_reqs = semantic_score(resume_data, jd_data, model=model)

    final = round(WEIGHT_KEYWORD * kw_score + WEIGHT_SEMANTIC * sem_score, 3)

    return {
        "job_title": jd_data.get("job_title", "Unknown role"),
        "keyword_score": kw_score,
        "semantic_score": sem_score,
        "final_score": final,
        "matched_keywords": matched_kw,
        "missing_keywords": missing_kw,
        "unmatched_requirements": [r for r, _ in gap_reqs],
        "covered_requirements": [r for r, _, _ in covered_reqs],
    }


def _print_report(result: dict):
    print(f"\n=== Match Report: {result['job_title']} ===")
    print(f"Keyword score  : {result['keyword_score'] * 100:.1f}%")
    print(f"Semantic score : {result['semantic_score'] * 100:.1f}%")
    print(f"Final score    : {result['final_score'] * 100:.1f}%\n")

    if result["missing_keywords"]:
        print("Missing keywords (only add these if honestly true):")
        for k in result["missing_keywords"]:
            print(f"  - {k}")
    else:
        print("No missing keywords found.")

    if result["unmatched_requirements"]:
        print("\nRequirements with no strong match in your resume:")
        for r in result["unmatched_requirements"]:
            print(f"  - {r}")

    print()


if __name__ == "__main__":
    resume_path = Path("resume_data.json")
    jd_path = Path("jd_data.json")

    if not resume_path.exists() or not jd_path.exists():
        print("Need resume_data.json and jd_data.json in this folder.")
        print("Run resume_extractor.py and jd_extractor.py first to generate them.")
        raise SystemExit(1)

    resume_data = json.loads(resume_path.read_text())
    jd_data = json.loads(jd_path.read_text())

    print("Processing score... ", end="", flush=True)
    result = score(resume_data, jd_data)
    print("Done.")

    Path("match_report.json").write_text(json.dumps(result, indent=2))