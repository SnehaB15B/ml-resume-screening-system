"""
nlp_utils.py
------------
Shared NLP building blocks used by both train_classifier.py and app.py:
  - IT category mapping (raw dataset labels -> engineering-student-focused roles)
  - reference skill list + per-role expected skills
  - resume section detection (rule-based, no ML)
  - text cleaning
  - keyword extraction (RAKE)
  - engineered numeric features (used in the TF-IDF vs TF-IDF+FeatureEngineering
    experiment in train_classifier.py, and shown live in the app)
"""

import re
from rake_nltk import Rake

# ============================================================
# CATEGORY MAPPING: raw dataset labels -> IT-focused roles
# ============================================================
# The raw UpdatedResumeDataSet.csv has 25 categories, many non-IT (HR, Sales,
# Arts, Advocate, Mechanical/Civil/Electrical Engineer, etc.) and several with
# very few samples (20-28). For an engineering-student-focused project we keep
# only the IT-relevant categories and relabel them to recognizable roles.
#
# IMPORTANT HONESTY NOTE: the raw dataset does NOT contain separate labels for
# Frontend vs Backend vs Full-Stack Developer, nor separate Data Analyst vs
# Data Scientist vs ML Engineer labels. Rather than invent a split the data
# doesn't support, categories are merged/renamed to the closest defensible IT
# role. This is documented in the README as a known dataset limitation.
CATEGORY_MAP = {
    "Java Developer": "Software Developer",
    "DotNet Developer": ".NET Developer",
    "Python Developer": "Python Developer",
    "Web Designing": "Web / Frontend Developer",
    "Data Science": "Data Scientist",
    "ETL Developer": "Data Analyst / ETL Developer",
    "DevOps Engineer": "Cloud / DevOps Engineer",
    "Testing": "QA / Test Engineer",
    "Automation Testing": "QA / Test Engineer",
    "Database": "Database Developer",
}

IT_CATEGORIES = sorted(set(CATEGORY_MAP.values()))

# ============================================================
# SKILLS REFERENCE
# ============================================================
ALL_SKILLS = [
    "Python", "Java", "C", "C++", "C#", ".NET", "JavaScript", "TypeScript", "Go", "Rust",
    "React", "Angular", "Vue", "Node.js", "Django", "Flask", "Spring", "Spring Boot", "Hibernate",
    "SQL", "MongoDB", "MySQL", "PostgreSQL", "Redis", "NoSQL",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "CI/CD",
    "Machine Learning", "Deep Learning", "AI", "NLP", "Data Science",
    "Pandas", "NumPy", "Scikit-learn", "Statistics",
    "Power BI", "Tableau", "Excel", "ETL",
    "REST API", "GraphQL", "Microservices",
    "HTML", "CSS", "Bootstrap", "Tailwind",
    "Data Structures", "Algorithms", "OOP",
    "Selenium", "JUnit", "TestNG", "Manual Testing", "Automation Testing",
    "Git", "Linux", "Agile", "Scrum",
    "Communication", "Leadership", "Problem Solving",
]

# Expected core skills per IT role — used for Career Fit Analysis (role fit %)
# and the resume-vs-role skill checklist. Kept short and curated (6-7 skills
# each) so the checklist stays readable in the UI.
ROLE_SKILLS = {
    "Software Developer": ["Java", "C++", "Data Structures", "Algorithms", "OOP", "Git", "Problem Solving"],
    ".NET Developer": ["C#", ".NET", "SQL", "REST API", "Azure", "Git"],
    "Python Developer": ["Python", "Django", "Flask", "SQL", "REST API", "Git"],
    "Web / Frontend Developer": ["HTML", "CSS", "JavaScript", "React", "Bootstrap", "Git"],
    "Data Scientist": ["Python", "Pandas", "NumPy", "Machine Learning", "Statistics", "Scikit-learn"],
    "Data Analyst / ETL Developer": ["SQL", "Excel", "Power BI", "Tableau", "ETL", "Python"],
    "Cloud / DevOps Engineer": ["AWS", "Docker", "Kubernetes", "Jenkins", "CI/CD", "Linux"],
    "QA / Test Engineer": ["Selenium", "Manual Testing", "Automation Testing", "Java", "SQL", "JUnit"],
    "Database Developer": ["SQL", "MySQL", "PostgreSQL", "MongoDB", "NoSQL", "ETL"],
}

# ============================================================
# RESUME SECTION DETECTION (rule-based, no ML)
# ============================================================
SECTION_KEYWORDS = {
    "Education": ["education", "academic", "qualification", "b.tech", "university", "college", "degree"],
    "Technical Skills": ["technical skills", "skills", "technologies"],
    "Projects": ["projects", "project"],
    "Experience / Internship": ["experience", "internship", "work experience", "intern"],
    "Certifications": ["certification", "certifications", "certificate"],
    "Achievements": ["achievement", "achievements", "award", "awards", "honor"],
}


def detect_sections(text: str) -> dict:
    """Rule-based detection of common resume sections by keyword presence.
    Returns {section_name: True/False}. Not ML — simple, explainable heuristic."""
    text_lower = text.lower()
    return {name: any(kw in text_lower for kw in kws) for name, kws in SECTION_KEYWORDS.items()}


# ============================================================
# TEXT CLEANING
# ============================================================
def clean_text(text: str) -> str:
    """Resume text cleaning: strip URLs, emails, non-alphabetic characters,
    collapse whitespace, lowercase. Used identically at train and inference time."""
    text = str(text)
    text = re.sub(r"http\S+\s*", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# ============================================================
# SKILL DETECTION
# ============================================================
def detect_skills(text: str) -> list:
    """Returns which reference skills appear in the given text, using word-
    boundary matching so short skill names (e.g. 'C', 'R', 'Go') don't false-
    positive match inside unrelated words (e.g. 'Science', 'Certificate')."""
    text_lower = text.lower()
    found = []
    for skill in ALL_SKILLS:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


# ============================================================
# KEYWORD EXTRACTION (RAKE)
# ============================================================
def extract_keywords(raw_text: str, top_n: int = 15) -> list:
    """RAKE keyword/phrase extraction. Needs the RAW (unclean) text since it
    relies on punctuation/stopword boundaries to find phrases."""
    try:
        rake = Rake()
        rake.extract_keywords_from_text(raw_text)
        return rake.get_ranked_phrases()[:top_n]
    except Exception:
        return []


# ============================================================
# ENGINEERED NUMERIC FEATURES (for the TF-IDF vs TF-IDF+FeatureEngineering
# experiment, and shown live in the app's preprocessing stats)
# ============================================================
def engineered_feature_vector(raw_text: str) -> dict:
    """A handful of simple, explainable numeric features derived from the
    raw resume text, used alongside TF-IDF in the feature-engineering
    experiment (see train_classifier.py)."""
    word_count = len(raw_text.split())
    skill_count = len(detect_skills(raw_text))
    section_count = sum(detect_sections(raw_text).values())
    try:
        rake = Rake()
        rake.extract_keywords_from_text(raw_text)
        keyword_count = len(rake.get_ranked_phrases())
    except Exception:
        keyword_count = 0
    return {
        "word_count": word_count,
        "skill_count": skill_count,
        "section_count": section_count,
        "keyword_count": keyword_count,
    }


ENGINEERED_FEATURE_NAMES = ["word_count", "skill_count", "section_count", "keyword_count"]
