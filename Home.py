import os
import streamlit as st
import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ─── PATH SETUP ────────────────────────────────────────────────────────────────
# Ensure we load data relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "IntelliStay_Synthetic_Survey_Data.csv")

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="IntelliStay AI Dashboard", layout="wide", page_icon="🏨")
st.title("🏨 IntelliStay AI – Guest Analytics Dashboard")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

# ─── SIDEBAR NAVIGATION ───────────────────────────────────────────────────────
page = st.sidebar.radio("Navigate to", [
    "1. Data Visualization",
    "2. Classification",
    "3. Clustering",
    "4. Association Rules",
    "5. Regression Insights"
])

# ─── 1. DATA VISUALIZATION ────────────────────────────────────────────────────
if page == "1. Data Visualization":
    st.header("📊 Data Visualization")
    st.dataframe(df.head(), use_container_width=True)

    # Insight 1: Age Distribution
    st.subheader("1. Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], kde=True, ax=ax)
    st.pyplot(fig)

    # Insight 2: Income vs. Spend
    st.subheader("2. Income vs. Spend per Night")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["Annual_Income_USD"], y=df["Average_Spend_Per_Night"],
                    hue=df["Gender"], ax=ax)
    st.pyplot(fig)

    # Insight 3: Preferred Hotel Type
    st.subheader("3. Preferred Hotel Type")
    fig, ax = plt.subplots()
    df["Preferred_Hotel_Type"].value_counts().plot(kind="barh", ax=ax)
    st.pyplot(fig)

    # Insight 4: Age by Willingness
    st.subheader("4. Age by Willingness to Stay")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Willing_To_Stay_At_IntelliStay"],
                y=df["Age"], ax=ax)
    st.pyplot(fig)

    # Insight 5: Mobile Check-In by Occupation
    st.subheader("5. Mobile Check-In Usage by Occupation")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Mobile_Checkin_Usage"],
                  hue=df["Occupation"], ax=ax)
    st.pyplot(fig)

    # Insight 6: Comfort vs. Spend
    st.subheader("6. Comfort with Smart Features vs. Spend")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Comfort_With_Smart_Features"],
                y=df["Average_Spend_Per_Night"], ax=ax)
    st.pyplot(fig)

    # Insight 7: Marital Status & Share Preferences
    st.subheader("7. Marital Status & Willingness to Share Preferences")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Marital_Status"],
                  hue=df["Willing_To_Share_Preferences"], ax=ax)
    st.pyplot(fig)

    # Insight 8: Top Challenges
    st.subheader("8. Top Reported Challenges")
    chal_list = list(chain(*df["Common_Challenges"].str.split(", ")))
    top_chal = pd.Series(chal_list).value_counts().head(10)
    fig, ax = plt.subplots()
    top_chal.plot(kind="barh", ax=ax)
    st.pyplot(fig)

    # Insight 9: AI Preference Score
    st.subheader("9. AI Preference Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["AI_Preference_Score"], kde=True, ax=ax)
    st.pyplot(fig)

    # Insight 10: Most Valued Aspects
    st.subheader("10. Most Valued Aspects")
    val_list = list(chain(*df["Valued_Aspects"].str.split(", ")))
    top_vals = pd.Series(val_list).value_counts().head(10)
    fig, ax = plt.subplots()
    top_vals.plot(kind="barh", ax=ax)
    st.pyplot(fig)


# ─── 2. CLASSIFICATION ────────────────────────────────────────────────────────
elif page == "2. Classification":
    st.header("🤖 Classification Models")

    target = "Willing_To_Stay_At_IntelliStay"
    df_clf = df.dropna(subset=[target]).copy()
    le_target = LabelEncoder()
    df_clf[target] = le_target.fit_transform(df_clf[target])

    X = df_clf.drop(columns=[target])
    y = df_clf[target]

    encoders = {}
    for col in X.select_dtypes(include="object"):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    # Performance Comparison
    st.subheader("Model Performance")
    perf = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        perf.append({
            "Model": name,
            "Train Acc": m.score(X_train, y_train),
            "Test Acc": m.score(X_test, y_test),
            "Precision": rep["1"]["precision"],
            "Recall": rep["1"]["recall"],
            "F1-Score": rep["1"]["f1-score"]
        })
    st.dataframe(pd.DataFrame(perf).round(3))

    # Confusion Matrix
    sel = st.selectbox("Select model for Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, models[sel].predict(X_test))
    st.subheader(f"Confusion Matrix: {sel}")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curves
    st.subheader("ROC Curve Comparison")
    fig, ax = plt.subplots()
    for name, m in models.items():
        if hasattr(m, "predict_proba"):
            y_score = m.fit(X_train, y_train).predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Upload & Predict
    st.subheader("Upload New Data for Prediction")
    uploaded = st.file_uploader("CSV (same features, no target)", type="csv")
    if uploaded:
        new_df = pd.read_csv(uploaded)
        for col, enc in encoders.items():
            new_df[col] = enc.transform(new_df[col].astype(str).fillna("Unknown"))
        preds = models[sel].predict(new_df)
        new_df["Predicted_Willingness"] = le_target.inverse_transform(preds)
        st.dataframe(new_df)
        buf = io.BytesIO()
        new_df.to_csv(buf, index=False)
        st.download_button("📥 Download Predictions", buf.getvalue(),
                           file_name="intellistay_predictions.csv")


# ─── 3. CLUSTERING ─────────────────────────────────────────────────────────────
elif page == "3. Clustering":
    st.header("🧬 K-Means Clustering")

    # Prepare numeric data for clustering
    num_df = df.select_dtypes(include=["int64", "float64"]).copy()
    scaler = StandardScaler()
    Xc = scaler.fit_transform(num_df)

    # Elbow Chart
    st.subheader("Elbow Chart")
    distortions = []
    ks = range(2, 11)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42).fit(Xc)
        distortions.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(ks, distortions, "o-")
    ax.set_xlabel("Number of Clusters"); ax.set_ylabel("Inertia")
    st.pyplot(fig)

    # Cluster Slider
    n_clusters = st.slider("Select number of clusters", 2, 10, 4)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(Xc)
    df["Cluster"] = km.labels_

    # Persona Table
    st.subheader("Cluster Personae (Mean Values)")
    persona = df.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(persona)

    # Download Labeled Data
    buf2 = io.BytesIO()
    df.to_csv(buf2, index=False)
    st.download_button("📥 Download Clustered Data", buf2.getvalue(),
                       file_name="intellistay_clustered.csv")


# ─── 4. ASSOCIATION RULES ───────────────────────────────────────────────────────
elif page == "4. Association Rules":
    st.header("🔗 Association Rule Mining")

    cols = st.multiselect(
        "Select columns for Apriori",
        ["Valued_Aspects", "Smart_Features_Desired", "Common_Challenges"],
        default=["Valued_Aspects", "Smart_Features_Desired"]
    )
    min_sup = st.slider("Min Support", 0.01, 0.5, 0.05)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3)

    # One-hot encode the selected multi-value columns
    basket = pd.get_dummies(df[cols].str.get_dummies(sep=", "))
    freq_items = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    top_rules = rules.sort_values("confidence", ascending=False).head(10)

    st.subheader("Top 10 Association Rules")
    st.dataframe(top_rules[["antecedents", "consequents", "support", "confidence", "lift"]])


# ─── 5. REGRESSION INSIGHTS ────────────────────────────────────────────────────
elif page == "5. Regression Insights":
    st.header("📈 Regression Models & Insights")

    # Prepare regression data
    reg_df = df.dropna(subset=["Average_Spend_Per_Night", "Annual_Income_USD"])
    Xr = reg_df[["Annual_Income_USD"]]
    yr = reg_df["Average_Spend_Per_Night"]

    models_r = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=5)
    }

    # R² Scores
    st.subheader("Model R² Scores")
    scores = []
    for name, mr in models_r.items():
        mr.fit(Xr, yr)
        scores.append({"Model": name, "R²": mr.score(Xr, yr)})
    st.table(pd.DataFrame(scores))

    # Spend vs Income Chart
    st.subheader("Predicted Spend vs. Income")
    fig, ax = plt.subplots()
    ax.scatter(reg_df["Annual_Income_USD"], reg_df["Average_Spend_Per_Night"],
               label="Actual", alpha=0.3)
    xs = np.linspace(reg_df["Annual_Income_USD"].min(),
                     reg_df["Annual_Income_USD"].max(), 100).reshape(-1, 1)
    for name, mr in models_r.items():
        ys = mr.predict(xs)
        ax.plot(xs, ys, label=name)
    ax.set_xlabel("Annual Income (USD)"); ax.set_ylabel("Avg Spend/Night")
    ax.legend()
    st.pyplot(fig)

    # Business Insights
    st.subheader("Business Insights & Interpretation")
    st.markdown("""
    1. **Linear Model**: Explains base correlation between income and spend.
    2. **Ridge & Lasso**: Regularization smooths predictions, reducing overfitting in extreme incomes.
    3. **Decision Tree**: Captures non-linear spend thresholds to inform tiered pricing.
    4. High-income guests show diminishing marginal spend increases.
    5. These models combined aid dynamic pricing strategies for maximizing revenue.
    """)

