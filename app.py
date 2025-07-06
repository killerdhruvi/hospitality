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

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="IntelliStay AI Dashboard", layout="wide", page_icon="🏨")
st.title("🏨 IntelliStay AI – Guest Analytics Dashboard")

# ─── DEBUG: SHOW DIRECTORY CONTENTS ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.write("**Debug: Base directory**:", BASE_DIR)
st.write("**Debug: Files in base**:", os.listdir(BASE_DIR))

# ─── LOCATE CSV ────────────────────────────────────────────────────────────────
# 1) Try CSV at repo root
root_csv = os.path.join(BASE_DIR, "IntelliStay_Synthetic_Survey_Data.csv")
# 2) Then try CSV in data/
data_csv = os.path.join(BASE_DIR, "data", "IntelliStay_Synthetic_Survey_Data.csv")

if os.path.exists(root_csv):
    DATA_PATH = root_csv
    st.success("Loading data from root CSV")
elif os.path.exists(data_csv):
    DATA_PATH = data_csv
    st.success("Loading data from data/ subfolder")
else:
    st.error(
        "❌ Could not find 'IntelliStay_Synthetic_Survey_Data.csv'.\n\n"
        "Please ensure it's either:\n"
        "• In the repo root (where app.py lives), named IntelliStay_Synthetic_Survey_Data.csv\n"
        "• Or inside a `data/` folder under that name."
    )
    st.stop()

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

# ─── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", [
    "1. Data Visualization",
    "2. Classification",
    "3. Clustering",
    "4. Association Rules",
    "5. Regression Insights"
])

# ─── TAB 1: DATA VISUALIZATION ─────────────────────────────────────────────────
if page == "1. Data Visualization":
    st.header("📊 Data Visualization")
    st.dataframe(df.head(), use_container_width=True)

    # Insight 1
    st.subheader("1. Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], kde=True, ax=ax)
    st.pyplot(fig)

    # Insight 2
    st.subheader("2. Income vs. Spend per Night")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df["Annual_Income_USD"],
        y=df["Average_Spend_Per_Night"],
        hue=df["Gender"],
        ax=ax
    )
    st.pyplot(fig)

    # Insight 3
    st.subheader("3. Preferred Hotel Type")
    fig, ax = plt.subplots()
    df["Preferred_Hotel_Type"].value_counts().plot(kind="barh", ax=ax)
    st.pyplot(fig)

    # Insight 4
    st.subheader("4. Age by Willingness to Stay")
    fig, ax = plt.subplots()
    sns.boxplot(
        x=df["Willing_To_Stay_At_IntelliStay"],
        y=df["Age"],
        ax=ax
    )
    st.pyplot(fig)

    # Insight 5
    st.subheader("5. Mobile Check-In Usage by Occupation")
    fig, ax = plt.subplots()
    sns.countplot(
        x=df["Mobile_Checkin_Usage"],
        hue=df["Occupation"],
        ax=ax
    )
    st.pyplot(fig)

    # Insight 6
    st.subheader("6. Comfort with Smart Features vs. Spend")
    fig, ax = plt.subplots()
    sns.boxplot(
        x=df["Comfort_With_Smart_Features"],
        y=df["Average_Spend_Per_Night"],
        ax=ax
    )
    st.pyplot(fig)

    # Insight 7
    st.subheader("7. Marital Status & Willingness to Share Preferences")
    fig, ax = plt.subplots()
    sns.countplot(
        x=df["Marital_Status"],
        hue=df["Willing_To_Share_Preferences"],
        ax=ax
    )
    st.pyplot(fig)

    # Insight 8
    st.subheader("8. Top Reported Challenges")
    chal_list = list(chain(*df["Common_Challenges"].str.split(", ")))
    top_chal = pd.Series(chal_list).value_counts().head(10)
    fig, ax = plt.subplots()
    top_chal.plot(kind="barh", ax=ax)
    st.pyplot(fig)

    # Insight 9
    st.subheader("9. AI Preference Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["AI_Preference_Score"], kde=True, ax=ax)
    st.pyplot(fig)

    # Insight 10
    st.subheader("10. Most Valued Aspects")
    val_list = list(chain(*df["Valued_Aspects"].str.split(", ")))
    top_vals = pd.Series(val_list).value_counts().head(10)
    fig, ax = plt.subplots()
    top_vals.plot(kind="barh", ax=ax)
    st.pyplot(fig)

# ─── TAB 2: CLASSIFICATION ─────────────────────────────────────────────────────
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

    # Performance
    st.subheader("Model Performance Comparison")
    rows = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rows.append({
            "Model": name,
            "Train Acc": m.score(X_train, y_train),
            "Test Acc": m.score(X_test, y_test),
            "Precision": rep["1"]["precision"],
            "Recall": rep["1"]["recall"],
            "F1-Score": rep["1"]["f1-score"]
        })
    st.dataframe(pd.DataFrame(rows).round(3))

    # Confusion
    sel = st.selectbox("Select model for Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, models[sel].predict(X_test))
    st.subheader(f"Confusion Matrix: {sel}")
    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le_target.classes_, yticklabels=le_target.classes_,
        ax=ax
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC
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

    # Upload & Download
    st.subheader("Upload New Data for Prediction")
    up = st.file_uploader("CSV with same features (no target)", type="csv")
    if up:
        new_df = pd.read_csv(up)
        for col, enc in encoders.items():
            new_df[col] = enc.transform(new_df[col].astype(str).fillna("Unknown"))
        preds = models[sel].predict(new_df)
        new_df["Predicted_Willingness"] = le_target.inverse_transform(preds)
        st.dataframe(new_df)
        buf = io.BytesIO()
        new_df.to_csv(buf, index=False)
        st.download_button("📥 Download Predictions", buf.getvalue(), file_name="predictions.csv")

# ─── TAB 3: CLUSTERING ─────────────────────────────────────────────────────────
elif page == "3. Clustering":
    st.header("🧬 K-Means Clustering")

    num_df = df.select_dtypes(include=["int64", "float64"]).copy()
    scaler = StandardScaler()
    Xc = scaler.fit_transform(num_df)

    # Elbow
    st.subheader("Elbow Chart")
    inertias = []
    ks = range(2, 11)
    for k in ks:
        inertias.append(KMeans(n_clusters=k, random_state=42).fit(Xc).inertia_)
    fig, ax = plt.subplots()
    ax.plot(ks, inertias, "o-")
    ax.set_xlabel("Clusters"); ax.set_ylabel("Inertia")
    st.pyplot(fig)

    # Slider & Persona
    n_clusters = st.slider("Number of Clusters", 2, 10, 4)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(Xc)
    df["Cluster"] = km.labels_
    st.subheader("Cluster Personas (Mean Values)")
    persona = df.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(persona)

    # Download
    buf2 = io.BytesIO()
    df.to_csv(buf2, index=False)
    st.download_button("📥 Download Clustered Data", buf2.getvalue(), file_name="clustered_data.csv")

# ─── TAB 4: ASSOCIATION RULES ──────────────────────────────────────────────────
elif page == "4. Association Rules":
    st.header("🔗 Association Rule Mining")

    cols = st.multiselect("Select columns", [
        "Valued_Aspects", "Smart_Features_Desired", "Common_Challenges"
    ], default=["Valued_Aspects", "Smart_Features_Desired"])
    min_sup = st.slider("Min Support", 0.01, 0.5, 0.05)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3)

    basket = pd.get_dummies(df[cols].str.get_dummies(sep=", "))
    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    top10 = rules.sort_values("confidence", ascending=False).head(10)

    st.subheader("Top 10 Rules")
    st.dataframe(top10[["antecedents", "consequents", "support", "confidence", "lift"]])

# ─── TAB 5: REGRESSION INSIGHTS ────────────────────────────────────────────────
elif page == "5. Regression Insights":
    st.header("📈 Regression Models & Insights")

    reg_df = df.dropna(subset=["Annual_Income_USD", "Average_Spend_Per_Night"])
    Xr = reg_df[["Annual_Income_USD"]]
    yr = reg_df["Average_Spend_Per_Night"]

    models_r = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=5)
    }

    # R²
    st.subheader("Model R² Scores")
    scores = []
    for name, mr in models_r.items():
        mr.fit(Xr, yr)
        scores.append({"Model": name, "R²": mr.score(Xr, yr)})
    st.table(pd.DataFrame(scores))

    # Plot
    st.subheader("Spend vs Income Predictions")
    fig, ax = plt.subplots()
    ax.scatter(reg_df["Annual_Income_USD"], reg_df["Average_Spend_Per_Night"], alpha=0.3, label="Actual")
    xs = np.linspace(reg_df["Annual_Income_USD"].min(), reg_df["Annual_Income_USD"].max(), 100).reshape(-1, 1)
    for name, mr in models_r.items():
        ax.plot(xs, mr.predict(xs), label=name)
    ax.set_xlabel("Income (USD)"); ax.set_ylabel("Spend/Night"); ax.legend()
    st.pyplot(fig)

    # Insights
    st.subheader("Business Insights")
    st.markdown("""
    1. **Linear**: Base correlation between income and spend.  
    2. **Ridge/Lasso**: Regularization for robust predictions.  
    3. **Decision Tree**: Captures non-linear spending tiers.  
    4. High-income guests show diminishing spend increases.  
    5. Apply these insights for dynamic pricing strategies.
    """)
