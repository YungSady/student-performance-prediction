
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Project directories
ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "student_model.pkl"
INFO_FILE = ROOT / "models" / "model_info.pkl"
RESULTS_FILE = ROOT / "models" / "model_results.csv"
DATA_FILE = ROOT / "data" / "student-mat.csv"


# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Interface styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    h1 {
        font-size: 2.1rem !important;
        font-weight: 650 !important;
    }

    h2 {
        font-size: 1.5rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
    }

    div[data-testid="stMetric"] {
        padding: 16px;
        border: 1px solid #dce3eb;
        border-radius: 10px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }

    .section-description {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Load trained model
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    information = joblib.load(INFO_FILE)
    return model, information


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_FILE, sep=";")


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_FILE)


if not MODEL_FILE.exists() or not INFO_FILE.exists():
    st.error(
        "Model files were not found. "
        "Run python train_model.py before opening the application."
    )
    st.stop()


try:
    model, info = load_model()
    features = list(info["features"])
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# Main heading
st.title("Student Academic Performance Prediction")

st.markdown(
    '<p class="section-description">'
    'Machine learning analysis of academic performance indicators'
    '</p>',
    unsafe_allow_html=True
)


# Navigation
st.sidebar.title("Student Performance")

page = st.sidebar.radio(
    "Navigation",
    [
        "Grade Prediction",
        "Model Evaluation",
        "Dataset Exploration",
        "Research Notes"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Python | Scikit-learn | Streamlit"
)


# --------------------------------------------------
# GRADE PREDICTION
# --------------------------------------------------

if page == "Grade Prediction":

    st.header("Final Grade Prediction")

    st.write(
        "Enter the student's academic information to estimate "
        "their final Mathematics grade."
    )

    st.divider()

    with st.form("prediction_form"):

        left, right = st.columns(2, gap="large")

        with left:

            st.subheader("Academic Background")

            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                format_func=lambda value: {
                    1: "Less than 2 hours",
                    2: "2–5 hours",
                    3: "5–10 hours",
                    4: "More than 10 hours"
                }[value],
                index=1
            )

            failures = st.number_input(
                "Previous Class Failures",
                min_value=0,
                max_value=4,
                value=0,
                step=1
            )

            absences = st.number_input(
                "Number of School Absences",
                min_value=0,
                max_value=93,
                value=4,
                step=1
            )

        with right:

            st.subheader("Previous Examination Results")

            g1 = st.slider(
                "First Period Grade (G1)",
                min_value=0,
                max_value=20,
                value=14
            )

            g2 = st.slider(
                "Second Period Grade (G2)",
                min_value=0,
                max_value=20,
                value=15
            )

            st.caption(
                "Examination grades are measured on a "
                "scale from 0 to 20."
            )

        submitted = st.form_submit_button(
            "Generate Prediction",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_values = {
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "G1": g1,
            "G2": g2
        }

        missing_features = [
            feature for feature in features
            if feature not in input_values
        ]

        if missing_features:
            st.error(
                "The model requires additional inputs: "
                + ", ".join(missing_features)
            )
            st.stop()

        input_data = pd.DataFrame([
            {
                feature: input_values[feature]
                for feature in features
            }
        ])

        try:
            prediction = float(model.predict(input_data)[0])
            prediction = max(0.0, min(20.0, prediction))

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.stop()

        st.divider()

        st.subheader("Prediction Result")

        result_col, details_col = st.columns([1, 2])

        with result_col:

            st.metric(
                label="Estimated Final Grade",
                value=f"{prediction:.2f} / 20"
            )

        with details_col:

            st.write(
                "The estimated grade is based on the "
                "student's previous examination results, "
                "study-time category, academic history "
                "and recorded absences."
            )

            st.caption(
                "This estimate is not a guaranteed "
                "examination outcome."
            )

        st.subheader("Prediction Summary")

        summary = pd.DataFrame({
            "Academic Indicator": [
                "Weekly Study Time",
                "Previous Failures",
                "School Absences",
                "First Period Grade",
                "Second Period Grade",
                "Predicted Final Grade"
            ],
            "Value": [
                {
                    1: "Less than 2 hours",
                    2: "2–5 hours",
                    3: "5–10 hours",
                    4: "More than 10 hours"
                }[studytime],
                str(failures),
                str(absences),
                str(g1),
                str(g2),
                f"{prediction:.2f}"
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        export_data = input_data.copy()
        export_data["predicted_G3"] = round(prediction, 2)

        st.download_button(
            label="Export Prediction",
            data=export_data.to_csv(index=False),
            file_name="student_prediction.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.header("Model Performance Evaluation")

    st.write(
        "Comparison of the regression models using "
        "held-out test data."
    )

    st.divider()

    if RESULTS_FILE.exists():

        results = load_results()

        required_columns = {"Model", "MAE", "RMSE", "R2"}

        if not required_columns.issubset(results.columns):
            st.error(
                "The evaluation file does not contain "
                "the expected columns."
            )
            st.stop()

        st.subheader("Evaluation Metrics")

        st.dataframe(
            results.round(3),
            use_container_width=True,
            hide_index=True
        )

        best_model_name = info["model_name"]

        st.metric(
            "Selected Model",
            best_model_name
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("Mean Absolute Error")

            st.bar_chart(
                results.set_index("Model")["MAE"]
            )

            st.caption(
                "Lower MAE indicates smaller average "
                "absolute prediction errors."
            )

        with right:

            st.subheader("Coefficient of Determination")

            st.bar_chart(
                results.set_index("Model")["R2"]
            )

            st.caption(
                "Higher R² indicates better explanatory "
                "performance on the evaluated data."
            )

        st.subheader("Root Mean Squared Error")

        st.bar_chart(
            results.set_index("Model")["RMSE"]
        )

        st.divider()

        st.subheader("Feature Importance")

        if hasattr(model, "feature_importances_"):

            importance = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            })

            importance = importance.sort_values(
                "Importance",
                ascending=False
            )

            st.bar_chart(
                importance.set_index("Feature")["Importance"]
            )

            st.dataframe(
                importance.round(4),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Feature importance describes how the "
                "selected model uses its inputs. "
                "It does not establish causal relationships."
            )

        elif hasattr(model, "coef_"):

            coefficients = pd.DataFrame({
                "Feature": features,
                "Coefficient": model.coef_
            })

            st.dataframe(
                coefficients.round(4),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Feature importance is not available "
                "for the selected model."
            )

        st.caption(
            "These results are based on a single train/test "
            "split. Model selection was also performed using "
            "the reported test results, so the estimates "
            "should be treated as preliminary."
        )

    else:

        st.warning(
            "Evaluation results are unavailable. "
            "Run train_model.py to generate them."
        )


# --------------------------------------------------
# DATASET EXPLORATION
# --------------------------------------------------

elif page == "Dataset Exploration":

    st.header("Dataset Exploration")

    st.write(
        "Descriptive analysis of the student records "
        "used in model development."
    )

    st.divider()

    if not DATA_FILE.exists():

        st.error(
            "Dataset not found at data/student-mat.csv."
        )
        st.stop()

    df = load_dataset()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Student Records",
        len(df)
    )

    col2.metric(
        "Dataset Variables",
        len(df.columns)
    )

    col3.metric(
        "Mean Final Grade",
        f"{df['G3'].mean():.2f} / 20"
    )

    st.divider()

    st.subheader("Final Grade Distribution")

    grade_distribution = (
        df["G3"]
        .value_counts()
        .sort_index()
        .rename_axis("Final Grade")
        .rename("Number of Students")
    )

    st.bar_chart(grade_distribution)

    st.subheader("Academic Performance Indicators")

    selected_columns = [
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2",
        "G3"
    ]

    correlations = df[selected_columns].corr()

    st.dataframe(
        correlations.round(3),
        use_container_width=True
    )

    st.caption(
        "Correlations describe statistical associations "
        "within this dataset and should not be interpreted "
        "as evidence of causation."
    )

    st.divider()

    st.subheader("Student Records")

    display_columns = list(
        dict.fromkeys(features + ["G3"])
    )

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Export Selected Dataset",
        data=df[display_columns].to_csv(index=False),
        file_name="student_research_data.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# RESEARCH NOTES
# --------------------------------------------------

elif page == "Research Notes":

    st.header("Research Methodology")

    st.subheader("Research Objective")

    st.write(
        "To examine the predictive performance of "
        "selected regression algorithms in estimating "
        "students' final Mathematics grades using "
        "previous academic results and related indicators."
    )

    st.subheader("Data Source")

    st.write(
        "The analysis uses the Student Performance "
        "dataset from the UCI Machine Learning Repository. "
        "The Mathematics dataset contains 395 records "
        "from Portuguese secondary schools."
    )

    st.markdown(
        "[UCI Student Performance Dataset]"
        "(https://archive.ics.uci.edu/dataset/320/student+performance)"
    )

    st.subheader("Predictor Variables")

    predictors = pd.DataFrame({
        "Variable": [
            "studytime",
            "failures",
            "absences",
            "G1",
            "G2"
        ],
        "Description": [
            "Weekly study-time category",
            "Number of previous class failures",
            "Number of school absences",
            "First-period Mathematics grade",
            "Second-period Mathematics grade"
        ]
    })

    st.dataframe(
        predictors,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "The target variable is G3, the final "
        "Mathematics grade measured from 0 to 20."
    )

    st.subheader("Machine Learning Algorithms")

    st.markdown("""
    Three regression algorithms were evaluated:

    - Linear Regression
    - Decision Tree Regression
    - Random Forest Regression
    """)

    st.subheader("Evaluation Criteria")

    st.markdown("""
    Model performance was assessed using:

    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Coefficient of Determination (R²)
    """)

    st.subheader("Study Limitations")

    st.write(
        "The dataset represents a limited educational "
        "context and may not reflect student populations "
        "in other countries or institutions."
    )

    st.write(
        "The inclusion of G1 and G2 means that the "
        "model estimates final grades after earlier "
        "assessment results are available."
    )

    st.write(
        "The current evaluation uses a single train/test "
        "split, with model selection based on test-set "
        "performance. Further cross-validation and "
        "independent testing are required before drawing "
        "strong conclusions about generalisation."
    )

    st.write(
        "Predictions are intended for methodological "
        "demonstration and should not be used as the "
        "sole basis for educational decisions."
    )


# Footer
st.divider()

st.caption(
    "Student Academic Performance Prediction"
)
