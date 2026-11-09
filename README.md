# Student Performance ML Lab

A reproducible research demonstration using **real historical student data**, three regression models, model evaluation, permutation feature importance, and a Streamlit dashboard.

## Windows quick start

1. Extract this ZIP into your existing `Desktop\StudentPerformanceML` folder, replacing `train_model.py` and `app.py`.
2. In Command Prompt:

```cmd
cd %USERPROFILE%\Desktop\StudentPerformanceML
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

The training script downloads the original UCI Mathematics CSV automatically (internet needed once). It saves the dataset in `data/student-mat.csv` and generated model/evaluation files in `models/`.

**Important:** Delete or rename the older synthetic `data/student_data.csv` if desired; this version does not use it. Never replace `student-mat.csv` with synthetic data.

## Data and academic limitations

Source: Cortez, P. (2008), *Student Performance*, UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. Licence: CC BY 4.0.

The dataset is from two Portuguese secondary schools. Grades are on a 0–20 scale. G1/G2 are earlier period grades; G3 is the final grade. Using G1/G2 makes this a **late-year** prediction task. A fully independent final evaluation would require model selection with cross-validation on the training portion, then a final untouched test set. Do not interpret feature importance as causality or use predictions for consequential student decisions.

## Files

- `train_model.py` — downloads real data, trains models, evaluates and saves artifacts.
- `app.py` — interactive prediction, model evaluation, dataset exploration and methodology.
- `requirements.txt` — required Python packages.
