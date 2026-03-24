# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-24

### Added
- Data quality reporting module with 8 validation checks and weighted quality score
- Preprocessing pipeline with ordinal encoding, log transforms, SMOTE, and StandardScaler
- LightGBM classifier with GridSearchCV hyperparameter tuning
- SHAP-based explainability with waterfall charts and plain-English factor descriptions
- Streamlit app with prediction interface and data quality dashboard
- Terminal-styled landing page with boot sequence, CLI, and command palette
- Comprehensive test suite for data quality and preprocessing modules
- SQL-style validation query documentation
- Makefile and shell script for reproducible pipeline execution
