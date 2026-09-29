# Machine Learning-Based Resume Screening and Job Matching System

Built for **engineering students applying to IT roles**. A pure Machine
Learning + NLP project — no generative AI, no chatbot, no LLM, no email
functionality. Every result is produced by a trained classical ML model or a
classical NLP/IR technique (TF-IDF, cosine similarity, rule-based section
detection), and every score is explainable and traceable back to its inputs.

## Pipeline Overview

```
                Resume (PDF / DOCX / TXT)
                          |
                          v
                  Text Extraction
                (PyMuPDF / python-docx)
                          |
                          v
                Text Preprocessing
        (lowercase, strip URLs/emails/punctuation)
                          |
                          v
               TF-IDF Vectorization
              (unigrams + bigrams, 5000 features)
                          |
                          v
              + Feature Engineering
   (word count, skill count, section count, keyword count)
                          |
                          v
              Train / Test Split (stratified)
                          |
                          v
                     ML Models
          +----------+----------+----------+
    Logistic      Naive      Linear      Random
    Regression    Bayes      SVM         Forest
          +----------+----------+----------+
                          |
                          v
        5-Fold Cross-Validation + Test Evaluation
      (Accuracy, Precision, Recall, F1, Confusion Matrix)
                          |
                          v
          Best Model Selected (CV F1, interpretability tie-break)
                          |
                          v
              IT Job Role Prediction
           (+ confidence, top-3 role matches)
                          |
                          v
         Career Fit Analysis (role skill checklist)
                          |
                          v
       Job Description Matching (TF-IDF cosine similarity)
                          |
                          v
                Skill Gap Analysis
                          |
                          v
         Final Project-defined ATS-style Score
```

## Project Scope

1. Resume text extraction (PDF / DOCX / TXT)
2. Text preprocessing
3. Resume section analysis (rule-based: Education, Skills, Projects, etc.)
4. Keyword extraction (RAKE)
5. Skill extraction (reference-list, word-boundary matching)
6. TF-IDF feature extraction
7. Engineered numeric features (word count, skill count, section count, keyword count)
8. Resume-to-job-description similarity (cosine similarity)
9. Supervised ML classification of resume -> IT job role
10. Comparison of 4 ML models (Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest)
11. Train/test split + 5-fold stratified cross-validation
12. TF-IDF vs TF-IDF+Engineered-Features experiment
13. Evaluation: accuracy, precision, recall, F1-score, confusion matrix
14. Prediction probability/confidence + Top-3 role matches
15. Feature importance / model coefficients (explainability)
16. Career Fit Analysis (role skill checklist + Role Fit %)
17. Skill Gap Analysis (job-description skill checklist)
18. Transparent, project-defined ATS-style scoring
19. Streamlit dashboard with visualizations
20. Model serialization with joblib

## Why only 9 categories, not 25 (or exactly the 10 you might expect)

The raw dataset (`UpdatedResumeDataSet.csv`, 962 resumes) ships with 25
generic categories, many non-IT (HR, Sales, Arts, Advocate, Mechanical/Civil/
Electrical Engineer, Health and Fitness, etc.) and several with very few
samples (20-28). For an engineering-student / IT-role focused project, only
the IT-relevant categories were kept and relabeled:

| New Role Label | Original Category | Resumes |
|---|---|---|
| Software Developer | Java Developer | 84 |
| QA / Test Engineer | Testing + Automation Testing | 96 |
| Cloud / DevOps Engineer | DevOps Engineer | 55 |
| Python Developer | Python Developer | 48 |
| Web / Frontend Developer | Web Designing | 45 |
| Data Scientist | Data Science | 40 |
| Data Analyst / ETL Developer | ETL Developer | 40 |
| Database Developer | Database | 33 |
| .NET Developer | DotNet Developer | 28 |

**Total: 469 resumes across 9 categories** (up from an average of ~10 per
class in the excluded categories, this gives 28-96 per class — noticeably
better balanced than the original 25-category split, where several
categories had only 20-25 samples total).

**Honesty note:** the raw dataset does not contain separate labels for
Frontend vs Backend vs Full-Stack Developer, nor for Data Analyst vs Data
Scientist vs ML Engineer as three distinct categories. Rather than invent a
split the data doesn't support, categories were mapped to the closest
defensible IT role (e.g. "Web Designing" -> "Web / Frontend Developer",
"Data Science" -> "Data Scientist" covering both data science and ML
engineering skill profiles). This mapping is in `nlp_utils.CATEGORY_MAP` and
is a good, honest answer if asked about it in a viva: *"why these 9 and not
your original 10?"* -> *"because the dataset's actual labels only support
this split; inventing finer-grained labels the data can't back up would be
dishonest, so this is the most defensible mapping."*

## Model Training & Selection

**Features:** TF-IDF, unigrams + bigrams, top 5000 by frequency, English
stopwords removed. Fit only on the training split to avoid data leakage.

**Models compared** (identical TF-IDF features):
| Model | Why it's included |
|---|---|
| Logistic Regression | Strong linear baseline; gives per-class coefficients for explainability |
| Multinomial Naive Bayes | Classic text-classification baseline, fast, probabilistic |
| Linear SVM | Often the strongest linear model for high-dimensional sparse text |
| Random Forest | Non-linear ensemble; tests whether non-linearity helps at all here |

**Selection rule:** each model is scored with 5-fold **stratified**
cross-validation (macro F1) on the training split. The model with the
highest mean CV F1 is normally selected — **with one documented
tie-break**: if multiple models are within 1 percentage point of the top CV
F1 (i.e. statistically indistinguishable given the cross-validation
variance), the more interpretable model is preferred, in this priority
order: Logistic Regression > Linear SVM > Multinomial Naive Bayes > Random
Forest. This matters because Random Forest exposes only *global* feature
importances, not per-category ones, which would break the Model
Explainability tab. This is a disclosed, deliberate design choice — a good
viva talking point (*"why not just always pick the single highest score?"*).

The winning model is then evaluated once more on a held-out test split for a
final, unbiased estimate — the test set is never used to choose between
models.

## TF-IDF vs TF-IDF + Engineered Features Experiment

A second experiment tests whether four simple, explainable numeric features
— resume word count, detected skill count, detected section count, and RAKE
keyword count — add value on top of pure TF-IDF text features. Engineered
features are min-max scaled and concatenated (`scipy.sparse.hstack`) onto
the TF-IDF matrix, then Logistic Regression and Linear SVM are retrained and
re-evaluated with the same CV setup. Results (from an actual run — your
numbers may vary slightly by machine/random seed but should be close):

| Feature Set + Model | CV F1 (mean) |
|---|---|
| TF-IDF only + Logistic Regression | 0.9834 |
| TF-IDF only + Linear SVM | 0.9834 |
| TF-IDF + Engineered Features + Logistic Regression | 0.9861 |
| TF-IDF + Engineered Features + Linear SVM | 0.9861 |

A small but consistent improvement from adding engineered features — a good
viva answer to *"why did you use feature engineering in addition to
TF-IDF?"*: even simple, cheap-to-compute structural/resume-quality signals
add complementary information TF-IDF alone doesn't capture (e.g. how
complete/well-structured the resume is), on top of its pure word-content
signal.

## Career Fit Analysis & Skill Gap Analysis (no LLM required)

Both are built entirely from the existing ML prediction + rule-based skill
matching — no generative AI involved:

- **Career Fit Analysis**: after predicting a role, the app compares the
  resume's detected skills against a curated reference skill list for that
  role (`nlp_utils.ROLE_SKILLS`), shows a ✅/❌ checklist, and computes a
  Role Fit % = matched / total core skills for that role.
- **Skill Gap Analysis**: when a job description is entered, the same
  matched/missing logic runs against that job description's detected
  skills instead of the role's fixed profile, and reports "Skill Gap: N
  skills" directly.

## Resume Section Analysis (rule-based NLP, no ML)

Detects the presence of six common resume sections — Education, Technical
Skills, Projects, Experience/Internship, Certifications, Achievements — by
keyword/heading matching (`nlp_utils.detect_sections`). This is explicitly
rule-based, not a trained model; it's included because it's a simple,
explainable NLP technique that adds real value to a resume-screening tool.

## Honest limitation (worth stating in your report/viva)
Cross-validation and test accuracy come out very high (~98-99%) on this
dataset. Part of the reason is that several resumes within the same category
are near-duplicate templates, so train and held-out test/validation folds
overlap heavily in style — this inflates the numbers relative to what you'd
see on genuinely independent, unseen resumes. Recommended future work:
evaluate on resumes never seen at all during model development (your own,
classmates', or scraped from elsewhere) for a more realistic generalization
estimate.

## Project Structure
```
.
├── app.py                          # Streamlit dashboard (3 tabs)
├── train_classifier.py             # Preprocessing, TF-IDF, multi-model training, evaluation, explainability
├── nlp_utils.py                    # Shared: category mapping, skills, section detection, cleaning, keywords
├── UpdatedResumeDataSet.csv        # Raw training data (962 resumes, 25 categories)
├── model/                          # Created automatically on first run
│   ├── best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── label_encoder.joblib
│   ├── best_model_name.txt
│   ├── category_counts.csv
│   ├── model_comparison.csv / .png
│   ├── feature_engineering_comparison.csv / .png
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── top_terms_overall.csv
│   └── top_terms_per_category.csv   (or global_feature_importance.csv)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

There is **no pre-trained model shipped in this folder on purpose.** The
first time you run the app, it automatically trains and compares all 4
models plus the feature-engineering experiment (about 20-30 seconds) and
saves the best one into `model/`. Training locally — rather than shipping
someone else's `.joblib` files — avoids `NotFittedError` / pickle
version-incompatibility crashes across different machines.

If you ever see a model-loading error anyway (e.g. after upgrading
scikit-learn), delete the `model/` folder and rerun the app, or run
`python train_classifier.py` manually.

## App Walkthrough

**Tab 1 — Resume Analysis**
Upload a resume → preview + preprocessing stats → resume section analysis →
ML-predicted role with confidence → Top-3 IT career matches → Career Fit
Analysis (skill checklist + Role Fit %) → detected skills & keywords → job
description matching (TF-IDF cosine similarity) → Skill Gap Analysis →
transparent, weighted ATS-style score breakdown.

**Tab 2 — Model Evaluation**
4-model comparison table + chart, feature-engineering experiment table +
chart, the selected best model's classification report and confusion
matrix.

**Tab 3 — Model Explainability**
Top TF-IDF terms overall (corpus-wide), top discriminative terms per
category from the best model's learned weights (or global feature
importance if the model doesn't support per-category weights), and the
curated role skill profiles used for Career Fit Analysis.

## Interview / viva talking points
Dataset (25 -> 9 IT categories, and why) → preprocessing → resume section
detection → TF-IDF feature engineering → engineered numeric features →
multi-model training → cross-validation-based model selection with a
documented interpretability tie-break → held-out test evaluation →
confusion matrix / precision / recall / F1 → prediction with confidence and
top-3 roles → career fit and skill gap analysis via rule-based skill
matching (no LLM) → resume-job similarity via TF-IDF cosine similarity →
transparent weighted ATS-style scoring → model explainability via linear
coefficients / log-probabilities / global feature importances.
