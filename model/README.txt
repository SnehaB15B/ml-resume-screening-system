This folder intentionally does NOT contain best_model.joblib, tfidf_vectorizer.joblib,
or label_encoder.joblib.

Those are generated automatically the first time you run `streamlit run app.py`
(or manually via `python train_classifier.py`), using YOUR installed scikit-learn
version. Shipping someone else's .joblib files can cause NotFittedError crashes
if your sklearn version differs.

The CSV/TXT/PNG files here are from a verified training run kept so you have
them immediately for your report. They'll be regenerated with your own machine's
results (should be nearly identical) the first time you run the app.
