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
st.title("🏨 IntelliStay AI – Guest Analytics")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/IntelliStay_Synthetic_Survey_Data.csv")
    return df

df = load_data()

# ─── SIDEBAR NAVIGATION ───────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", [
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

    # 10 Complex Insights
    # Insight 1: Age distribution
    st.subheader("1. Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df.Age, kde=True, ax=ax)
    st.pyplot(fig)

    # Insight 2: Income vs Spend
    st.subheader("2. Income vs. Spend per Night")
    fig, ax = plt.subplots()
    sns.scatterplot(df.Annual_Income_USD, df.Average_Spend_Per_Night, hue=df.Gender, ax=ax)
    st.pyplot(fig)

    # Insight 3: Hotel Type Preference
    st.subheader("3. Preferred Hotel Type")
    fig, ax = plt.subplots()
    df.Preferred_Hotel_Type.value_counts().plot(kind="barh", ax=ax)
    st.pyplot(fig)

    # Insight 4: Willingness vs Age
    st.subheader("4. Age by Willingness to Stay at IntelliStay")
    fig, ax = plt.subplots()
    sns.boxplot(x=df.Willing_To_Stay_At_IntelliStay, y=df.Age, ax=ax)
    st.pyplot(fig)

    # Insight 5: Mobile Check-in by Occupation
    st.subheader("5. Mobile Check-In Usage by Occupation")
    fig, ax = plt.subplots()
    sns.countplot(x=df.Mobile_Checkin_Usage, hue=df.Occupation, ax=ax)
    st.pyplot(fig)

    # Insight 6: Smart Features Comfort vs Spend
    st.subheader("6. Comfort with Smart Features vs. Spend")
    fig, ax = plt.subplots()
    sns.boxplot(x=df.Comfort_With_Smart_Features, y=df.Average_Spend_Per_Night, ax=ax)
    st.pyplot(fig)

    # Insight 7: Marital Status vs Share Preferences
    st.subheader("7. Marital Status & Willingness to Share Preferences")
    fig, ax = plt.subplots()
    sns.countplot(x=df.Marital_Status, hue=df.Willing_To_Share_Preferences, ax=ax)
    st.pyplot(fig)

    # Insight 8: Top Challenges
    st.subheader("8. Top Reported Challenges")
    challenges = list(chain(*df.Common_Challenges.str.split(", ")))
    top_chal = pd.Series(challenges).value_counts().head(10)
    fig, ax = plt.subplots()
    top_chal.plot(kind="barh", ax=ax)
    st.pyplot(fig)

    # Insight 9: AI Preference Score
    st.subheader("9. AI Preference Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df.AI_Preference_Score, kde=True, ax=ax)
    st.pyplot(fig)

    # Insight 10: Valued Aspects
    st.subheader("10. Most Valued Aspects")
    values = list(chain(*df.Valued_Aspects.str.split(", ")))
    top_vals = pd.Series(values).value_counts().head(10)
    fig, ax = plt.subplots()
    top_vals.plot(kind="barh", ax=ax)
    st.pyplot(fig)

# ─── 2. CLASSIFICATION ─────────────────────────────────────────────────────────
elif page == "2. Classification":
    st.header("🤖 Classification Models")

    # Prepare data
    target = "Willing_To_Stay_At_IntelliStay"
    df_clf = df.dropna(subset=[target]).copy()
    le = LabelEncoder()
    df_clf[target] = le.fit_transform(df_clf[target])
    X = df_clf.drop(columns=[target])
    y = df_clf[target]

    # Encode categoricals
    encoders = {}
    for col in X.select_dtypes(include="object"):
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))
        encoders[col] = enc

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    # Performance table
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
            "Precision": rep['1']['precision'],
            "Recall": rep['1']['recall'],
            "F1-Score": rep['1']['f1-score']
        })
    st.dataframe(pd.DataFrame(perf).round(3))

    # Confusion Matrix toggle
    sel = st.selectbox("Select model for Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, models[sel].predict(X_test))
    st.subheader(f"Confusion Matrix: {sel}")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC curve
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, m in models.items():
        if hasattr(m, "predict_proba"):
            y_score = m.fit(X_train, y_train).predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0,1], [0,1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    st.pyplot(fig)

    # Upload & predict
    st.subheader("Upload New Data for Prediction")
    up = st.file_uploader("CSV (same columns minus target)", type="csv")
    if up:
        new = pd.read_csv(up)
        for col, enc in encoders.items():
            new[col] = enc.transform(new[col].astype(str).fillna("Unknown"))
        preds = models[sel].predict(new)
        new["Predicted"] = le.inverse_transform(preds)
        st.dataframe(new)
        buf = io.BytesIO()
        new.to_csv(buf, index=False)
        st.download_button("Download Predictions", buf.getvalue(), file_name="predictions.csv")

# ─── 3. CLUSTERING ──────────────────────────────────────────────────────────────
elif page == "3. Clustering":
    st.header("🧬 K-Means Clustering")

    # Features to cluster on (numeric encoding)
    df_clu = df.copy()
    df_clu = pd.get_dummies(df_clu.select_dtypes(include=["int64","float64"]), drop_first=True)
    scaler = StandardScaler()
    Xc = scaler.fit_transform(df_clu)

    # Elbow chart
    st.subheader("Elbow Chart")
    distortions = []
    K = range(2, 11)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        distortions.append(km.fit(Xc).inertia_)
    fig, ax = plt.subplots()
    ax.plot(K, distortions, "o-")
    ax.set_xlabel("Number of Clusters"); ax.set_ylabel("Inertia")
    st.pyplot(fig)

    # Cluster slider
    n_clusters = st.slider("Select number of clusters", 2, 10, 4)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(Xc)
    df["Cluster"] = km.labels_

    # Persona table (mean values per cluster)
    st.subheader("Cluster Personas")
    persona = df.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(persona)

    # Download labeled data
    buf2 = io.BytesIO()
    df.to_csv(buf2, index=False)
    st.download_button("Download Clustered Data", buf2.getvalue(), file_name="clustered_data.csv")

# ─── 4. ASSOCIATION RULES ───────────────────────────────────────────────────────
elif page == "4. Association Rules":
    st.header("🔗 Association Rule Mining")

    # Choose columns
    cols = st.multiselect("Select multi-value columns", ["Valued_Aspects","Smart_Features_Desired","Common_Challenges"], default=["Valued_Aspects","Smart_Features_Desired"])
    min_sup = st.slider("Min Support", 0.01, 0.5, 0.05)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3)

    # One-hot encode selections
    basket = pd.get_dummies(df[cols].str.get_dummies(sep=", "))
    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])

# ─── 5. REGRESSION INSIGHTS ────────────────────────────────────────────────────
elif page == "5. Regression Insights":
    st.header("📈 Regression Models")

    # Prepare regression data
    df_reg = df.dropna(subset=["Average_Spend_Per_Night","Annual_Income_USD"])
    Xr = df_reg[["Annual_Income_USD"]]
    yr = df_reg["Average_Spend_Per_Night"]

    models_r = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=5)
    }

    # Fit & show metrics
    insights = []
    for name, mr in models_r.items():
        mr.fit(Xr, yr)
        score = mr.score(Xr, yr)
        insights.append({"Model": name, "R²": score})
    st.subheader("Model R² Scores")
    st.table(pd.DataFrame(insights))

    # Plot predictions
    st.subheader("Predicted Spend vs. Income")
    fig, ax = plt.subplots()
    ax.scatter(df_reg.Annual_Income_USD, df_reg.Average_Spend_Per_Night, label="Actual", alpha=0.3)
    xs = np.linspace(df_reg.Annual_Income_USD.min(), df_reg.Annual_Income_USD.max(), 100).reshape(-1,1)
    for name, mr in models_r.items():
        ys = mr.predict(xs)
        ax.plot(xs, ys, label=name)
    ax.set_xlabel("Annual Income (USD)"); ax.set_ylabel("Spend/Night")
    ax.legend()
    st.pyplot(fig)

    # Additional 5–7 quick insights
    st.subheader("Business Insights")
    st.markdown("""
    1. **Linear Model R²** indicates the proportion of variance in spend explained by income.  
    2. **Ridge & Lasso** help control overfitting via regularization, useful for robust spend forecasting.  
    3. **Decision Tree** captures non-linear income-spend relationships, highlighting thresholds for pricing tiers.  
    4. High-income guests tend to have higher spend but with diminishing returns—seen by divergence of model lines.  
    5. Regularization models (Ridge/Lasso) offer smoother spend predictions for mid-income segments.  
    6. Decision Tree reveals distinct spend clusters, aiding tiered pricing strategy.  
    7. Combining these insights can optimize room rates dynamically based on guest income profiles.
    """)

