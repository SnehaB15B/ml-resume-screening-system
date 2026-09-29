import io
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import docx
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import train_classifier as tc
from nlp_utils import (
    ALL_SKILLS,
    ROLE_SKILLS,
    IT_CATEGORIES,
    clean_text,
    detect_skills,
    detect_sections,
    extract_keywords,
)

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="ML-Based Resume Screening and Job Matching System", layout="wide")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_THIS_DIR, "model")
MODEL_FILES = {
    "model": os.path.join(MODEL_DIR, "best_model.joblib"),
    "vectorizer": os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"),
    "label_encoder": os.path.join(MODEL_DIR, "label_encoder.joblib"),
}


def download_nltk():
    for pkg, path in [("punkt", "tokenizers/punkt"),
                       ("punkt_tab", "tokenizers/punkt_tab"),
                       ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)


download_nltk()

# ============================================================
# LOAD TRAINED ML ARTIFACTS (self-healing: retrains if missing/incompatible)
# ============================================================
@st.cache_resource
def load_ml_artifacts():
    if not all(os.path.exists(p) for p in MODEL_FILES.values()):
        with st.spinner("No trained model found — training and comparing models now (~20-30s, one-time)..."):
            tc.train_and_save()
    try:
        model = joblib.load(MODEL_FILES["model"])
        vectorizer = joblib.load(MODEL_FILES["vectorizer"])
        label_encoder = joblib.load(MODEL_FILES["label_encoder"])
        _ = vectorizer.idf_  # sanity check: raises if pickled with incompatible sklearn version
    except Exception:
        with st.spinner("Existing model is incompatible with this environment — retraining..."):
            tc.train_and_save()
        model = joblib.load(MODEL_FILES["model"])
        vectorizer = joblib.load(MODEL_FILES["vectorizer"])
        label_encoder = joblib.load(MODEL_FILES["label_encoder"])

    best_model_name = "Unknown"
    name_path = os.path.join(MODEL_DIR, "best_model_name.txt")
    if os.path.exists(name_path):
        with open(name_path) as f:
            best_model_name = f.read().strip()

    def _read_csv(name):
        p = os.path.join(MODEL_DIR, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    comparison_df = _read_csv("model_comparison.csv")
    fe_df = _read_csv("feature_engineering_comparison.csv")
    category_counts_df = _read_csv("category_counts.csv")

    return model, vectorizer, label_encoder, best_model_name, comparison_df, fe_df, category_counts_df


try:
    (ml_model, tfidf_vectorizer, label_encoder, BEST_MODEL_NAME,
     comparison_df, fe_df, category_counts_df) = load_ml_artifacts()
    ML_READY = True
except Exception as e:
    st.error(f"Could not load or train the ML model: {e}")
    ML_READY = False

# Optional preset job descriptions for the Job Matching section
PRESET_JOBS = {
    "Custom (paste below)": "",
    "Data Scientist": "Python Pandas NumPy Machine Learning Statistics Scikit-learn Data Science",
    "Data Analyst / ETL Developer": "SQL Excel Power BI Tableau ETL Python",
    "Software Developer": "Java C++ Data Structures Algorithms OOP Git Problem Solving",
    "Web / Frontend Developer": "HTML CSS JavaScript React Bootstrap Git",
    "Cloud / DevOps Engineer": "AWS Docker Kubernetes Jenkins CI/CD Linux",
    "QA / Test Engineer": "Selenium Manual Testing Automation Testing Java SQL JUnit",
}

CHECK, CROSS = "✅", "❌"

# ============================================================
# TEXT EXTRACTION
# ============================================================
def extract_text(file):
    name = file.name.lower()
    content = file.getvalue()
    try:
        if name.endswith(".pdf"):
            doc = fitz.open(stream=content, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        elif name.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return content.decode("utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"Could not parse file: {e}")
        return ""


# ============================================================
# ML PREDICTION
# ============================================================
def predict_category(resume_text):
    cleaned = clean_text(resume_text)
    vec = tfidf_vectorizer.transform([cleaned])
    pred_label = ml_model.predict(vec)[0]
    category = label_encoder.inverse_transform([pred_label])[0]

    if hasattr(ml_model, "predict_proba"):
        proba = ml_model.predict_proba(vec)[0]
        confidence = float(proba[pred_label])
        prob_df = pd.DataFrame({
            "Category": label_encoder.classes_,
            "Probability": proba,
        }).sort_values("Probability", ascending=False).reset_index(drop=True)
        return category, confidence, prob_df
    return category, None, None


def job_similarity(resume_text, job_text):
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english").fit_transform([resume_text, job_text])
    return float(cosine_similarity(vec[0:1], vec[1:2])[0][0])


# ============================================================
# UI
# ============================================================
st.title("Machine Learning-Based Resume Screening and Job Matching System")
st.caption(
    "Built for engineering students applying to IT roles. "
)

if not ML_READY:
    st.stop()

with st.expander("ℹ️ About the 9 IT role categories used in this project"):
    st.write(
        "This project classifies resumes into 9 IT-focused roles, filtered and "
        "relabeled from the original dataset's 25 generic categories (which "
        "included non-IT roles like HR, Sales, and Civil Engineer). The dataset "
        "does not contain separate labels for Frontend vs Backend vs Full-Stack "
        "Developer, or for Data Analyst vs Data Scientist vs ML Engineer as three "
        "distinct categories — so these are represented by the closest available "
        "category rather than invented. See the README for the full mapping."
    )
    if category_counts_df is not None:
        st.dataframe(category_counts_df, hide_index=True)

tab_predict, tab_eval, tab_explain = st.tabs(
    ["Resume Analysis", "Model Evaluation", "Model Explainability"]
)

# ------------------------------------------------------------
# TAB 1: RESUME ANALYSIS
# ------------------------------------------------------------
with tab_predict:
    st.subheader(f"Active model: {BEST_MODEL_NAME}  (selected by cross-validated F1-score)")

    file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if file:
        raw_text = extract_text(file)

        st.subheader("Resume Preview")
        st.text_area("Extracted text (first 1000 characters)", raw_text[:1000], height=160)

        if len(raw_text.strip()) == 0:
            st.error("Could not read resume. Try another file.")
            st.stop()

        cleaned_text = clean_text(raw_text)
        keywords = extract_keywords(raw_text)
        detected = detect_skills(raw_text)
        sections = detect_sections(raw_text)

        # ---- Preprocessing statistics ----
        st.subheader("Preprocessing Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original word count", len(raw_text.split()))
        c2.metric("Cleaned word count", len(cleaned_text.split()))
        c3.metric("Extracted keywords", len(keywords))
        c4.metric("Detected skills", len(detected))

        # ---- Resume Section Analysis ----
        st.subheader("Resume Structure (Section Analysis)")
        st.caption("Rule-based detection of common resume sections by heading/keyword presence.")
        sec_cols = st.columns(len(sections))
        for col, (name, present) in zip(sec_cols, sections.items()):
            col.markdown(f"**{name}**")
            col.markdown(f"{CHECK if present else CROSS}")
        sections_found = sum(sections.values())
        st.caption(f"{sections_found} of {len(sections)} expected sections detected.")

        # ---- ML Prediction ----
        st.subheader("ML Prediction: Job Category")
        category, confidence, prob_df = predict_category(raw_text)
        if confidence is not None:
            st.success(f"Predicted role: **{category}** — confidence: {confidence:.1%}")
        else:
            st.success(f"Predicted role: **{category}** (this model does not output probabilities)")

        # ---- Top 3 Suitable Roles ----
        if prob_df is not None:
            st.subheader("Your Top IT Career Matches")
            top3 = prob_df.head(3).reset_index(drop=True)
            cols = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            for i, col in enumerate(cols):
                if i < len(top3):
                    col.metric(f"{medals[i]} {top3.loc[i, 'Category']}", f"{top3.loc[i, 'Probability']:.1%}")

            st.markdown("**Full probability breakdown**")
            fig, ax = plt.subplots(figsize=(7, 4))
            top_n_chart = prob_df.head(9)
            ax.barh(top_n_chart["Category"][::-1], top_n_chart["Probability"][::-1])
            ax.set_xlabel("Probability")
            st.pyplot(fig)

        # ---- Career Fit Analysis ----
        st.subheader("Career Fit Analysis")
        st.caption("Why this predicted role? Comparing your detected skills against the core skills typically expected for it.")
        role_skills = ROLE_SKILLS.get(category, [])
        if role_skills:
            matched_role_skills = [s for s in role_skills if s in detected]
            missing_role_skills = [s for s in role_skills if s not in detected]
            role_fit_pct = len(matched_role_skills) / len(role_skills)

            checklist_cols = st.columns(len(role_skills))
            for col, skill in zip(checklist_cols, role_skills):
                mark = CHECK if skill in detected else CROSS
                col.markdown(f"**{skill}**")
                col.markdown(mark)

            st.metric(f"Role Fit — {category}", f"{role_fit_pct:.0%}")
            if missing_role_skills:
                st.info(f"To strengthen your fit for **{category}**, consider learning: {', '.join(missing_role_skills)}")
        else:
            st.info("No curated skill profile available for this category.")

        # ---- Skills & keywords ----
        st.subheader("Detected Skills & Keywords")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**All detected skills (full reference list)**")
            st.write(detected if detected else "No skills from the reference list were detected.")
        with colB:
            st.markdown("**Extracted keywords (RAKE)**")
            st.write(keywords if keywords else "No keywords extracted.")

        # ---- Job Matching + Skill Gap Analysis ----
        st.subheader("Job Matching")
        preset_choice = st.selectbox("Choose a sample job description, or select Custom to paste your own",
                                      list(PRESET_JOBS.keys()))
        default_text = PRESET_JOBS[preset_choice]
        job_text = st.text_area("Job description", value=default_text, height=110)

        if job_text.strip():
            sim = job_similarity(cleaned_text, clean_text(job_text))
            jd_skills = detect_skills(job_text)
            matched_skills = sorted(set(detected) & set(jd_skills))
            missing_skills = sorted(set(jd_skills) - set(detected))
            skill_match_pct = (len(matched_skills) / len(jd_skills)) if jd_skills else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("TF-IDF cosine similarity", f"{sim:.1%}")
            m2.metric("Skill match", f"{skill_match_pct:.1%}")
            m3.metric("Skill Gap", f"{len(missing_skills)} skills")

            st.markdown(f"**For {preset_choice if preset_choice != 'Custom (paste below)' else 'this job description'}**")
            if jd_skills:
                gap_cols = st.columns(len(jd_skills))
                for col, skill in zip(gap_cols, jd_skills):
                    mark = CHECK if skill in matched_skills else CROSS
                    col.markdown(f"**{skill}**")
                    col.markdown(mark)
            else:
                st.info("No reference skills detected in this job description.")

            # ---- ATS-style score (transparent breakdown) ----
            st.subheader("Project-defined ATS-style Score")
            st.caption(
                "This is a transparent, project-defined scoring formula for "
                "demonstration purposes. It does not reproduce the proprietary "
                "scoring system of any commercial ATS software."
            )

            resume_feature_score = (len(detected) / len(ALL_SKILLS)) * 100
            skill_match_score = skill_match_pct * 100
            job_sim_score = sim * 100
            ml_confidence_score = (confidence * 100) if confidence is not None else 50.0

            weights = {
                "Job similarity (TF-IDF cosine)": (job_sim_score, 0.30),
                "Skill match (vs job description)": (skill_match_score, 0.30),
                "Resume feature score (skill richness)": (resume_feature_score, 0.20),
                "ML classifier confidence": (ml_confidence_score, 0.20),
            }
            final_score = sum(val * w for val, w in weights.values())

            breakdown_df = pd.DataFrame([
                {"Component": k, "Score": round(v, 1), "Weight": f"{int(w * 100)}%"}
                for k, (v, w) in weights.items()
            ])
            st.dataframe(breakdown_df, hide_index=True)

            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.barh(breakdown_df["Component"], breakdown_df["Score"])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score (0-100)")
            plt.tight_layout()
            st.pyplot(fig)

            st.metric("Final project-defined ATS-style score", f"{final_score:.1f} / 100")
            if final_score > 75:
                st.success("Strong match for this job description.")
            elif final_score > 50:
                st.warning("Moderate match — some gaps to address.")
            else:
                st.error("Weak match — significant gaps against this job description.")
        else:
            st.info("Enter or select a job description above to run job matching, skill gap analysis, and the ATS-style score.")

# ------------------------------------------------------------
# TAB 2: MODEL EVALUATION
# ------------------------------------------------------------
with tab_eval:
    st.subheader("Model Comparison")
    st.caption(
        "Four candidate models were trained on TF-IDF features and compared "
        "using 5-fold cross-validation (macro F1) on the training split, then "
        "evaluated once on a held-out test split. The model with the highest "
        "cross-validated F1 is selected, with a documented interpretability "
        "tie-break (see README) when scores are within 1 point of each other."
    )
    if comparison_df is not None:
        st.dataframe(comparison_df, hide_index=True)
        cmp_chart_path = os.path.join(MODEL_DIR, "model_comparison.png")
        if os.path.exists(cmp_chart_path):
            st.image(cmp_chart_path, caption="Cross-validation F1 vs test F1 across models")
    else:
        st.info("Model comparison data not found. Run train_classifier.py to regenerate it.")

    st.subheader("Feature Engineering Experiment: TF-IDF vs TF-IDF + Engineered Features")
    st.caption(
        "Does adding simple engineered numeric features (resume word count, "
        "skill count, detected-section count, keyword count) on top of TF-IDF "
        "improve macro F1 over TF-IDF alone?"
    )
    if fe_df is not None:
        st.dataframe(fe_df, hide_index=True)
        fe_chart_path = os.path.join(MODEL_DIR, "feature_engineering_comparison.png")
        if os.path.exists(fe_chart_path):
            st.image(fe_chart_path)
    else:
        st.info("Feature engineering comparison data not found. Run train_classifier.py to regenerate it.")

    st.subheader(f"Best Model Detail: {BEST_MODEL_NAME}")
    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.text(f.read())

    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption=f"Confusion matrix — {BEST_MODEL_NAME}")

    st.caption(
        "Note: with 9 categories and 469 total IT-role resumes, some categories "
        "still have as few as 28-40 examples, and several resumes within the same "
        "category are near-duplicate templates, which inflates cross-validation "
        "and test scores. Evaluating on genuinely independent, unseen resumes is "
        "recommended future work before drawing strong generalization conclusions."
    )

# ------------------------------------------------------------
# TAB 3: MODEL EXPLAINABILITY
# ------------------------------------------------------------
with tab_explain:
    st.subheader("Top TF-IDF Terms — Overall Corpus")
    st.caption("Terms with the highest mean TF-IDF weight across the training corpus (model-independent).")
    overall_path = os.path.join(MODEL_DIR, "top_terms_overall.csv")
    if os.path.exists(overall_path):
        overall_df = pd.read_csv(overall_path)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(overall_df["term"][::-1], overall_df["mean_tfidf_weight"][::-1])
        ax.set_xlabel("Mean TF-IDF weight")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader(f"Top Discriminative Terms per Category — {BEST_MODEL_NAME}")
    percat_path = os.path.join(MODEL_DIR, "top_terms_per_category.csv")
    global_path = os.path.join(MODEL_DIR, "global_feature_importance.csv")
    if os.path.exists(percat_path):
        percat_df = pd.read_csv(percat_path)
        categories = sorted(percat_df["category"].unique())
        chosen_cat = st.selectbox("Choose a category", categories)
        subset = percat_df[percat_df["category"] == chosen_cat].sort_values("weight", ascending=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(subset["term"][::-1], subset["weight"][::-1])
        ax.set_xlabel(f"{BEST_MODEL_NAME} weight for '{chosen_cat}'")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(
            "These are the model's learned linear weights (or log-probabilities) "
            "connecting each TF-IDF term to this category — i.e. which words push "
            "the prediction toward this category."
        )
    elif os.path.exists(global_path):
        st.info(
            f"{BEST_MODEL_NAME} only exposes GLOBAL feature importances, not "
            "per-category — showing the global view instead."
        )
        global_df = pd.read_csv(global_path)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(global_df["term"][::-1], global_df["importance"][::-1])
        ax.set_xlabel("Global feature importance")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No explainability data found. Run train_classifier.py to regenerate it.")

    st.subheader("Role Skill Profiles (used for Career Fit Analysis)")
    st.caption("Curated reference skill sets per role, used to compute the Role Fit % on the Resume Analysis tab.")
    role_skill_df = pd.DataFrame([
        {"Role": role, "Core Skills": ", ".join(skills)} for role, skills in ROLE_SKILLS.items()
    ])
    st.dataframe(role_skill_df, hide_index=True)
