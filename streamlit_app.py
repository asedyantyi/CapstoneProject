import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Page Setup ---
st.set_page_config(page_title="⚽ La Liga Match Predictor", layout="wide", page_icon="⚽")

# --- Background Style ---
page_bg = """
<style>
.stApp {
  background: linear-gradient(135deg, #004d00 0%, #003300 100%);
  background-image: url('https://upload.wikimedia.org/wikipedia/commons/e/e8/Soccer_Field_Overlay_Example.png');
  background-size: cover;
  background-repeat: no-repeat;
  background-position: center;
  color: #e6ffe6;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3 {
  font-weight: 900;
  text-shadow: 2px 2px 4px #000000;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; font-size:3rem;'>⚽ La Liga Match Predictor (Random Forest)</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
st.sidebar.image("La_liga.png", width=130)
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose Page", ["Upload Dataset", "Visualize", "Predict Match"])

# --- Load Dataset ---
DATA_PATH = "SP1_All_Seasons.csv"
df_cleaned = None
try:
    df = pd.read_csv(DATA_PATH)
    if 'AR' in df.columns:
        ar_index = df.columns.get_loc('AR')
        df_cleaned = df.iloc[:, :ar_index + 1]
    else:
        df_cleaned = df
except Exception as e:
    st.error(f"Error loading data:\n{e}")

# --- Train Random Forest Model ---
if df_cleaned is not None and 'FTR' in df_cleaned.columns:
    df_model = df_cleaned.dropna(subset=['FTR']).copy()
    label_encoders = {}
    for col in ['HomeTeam', 'AwayTeam', 'FTR']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    feature_cols = [c for c in df_model.columns if c not in ['FTR', 'Date']]
    X = df_model[feature_cols].select_dtypes(include=[np.number])
    y = df_model['FTR']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
else:
    rf, accuracy, df_model = None, None, None

# --- Page: Upload Dataset ---
if page == "Upload Dataset":
    st.subheader("Uploaded Dataset Preview")
    if df_cleaned is not None:
        st.success("✅ La Liga dataset loaded successfully!")
        st.dataframe(df_cleaned.head(10))
        st.write(f"Model accuracy (Random Forest): **{accuracy:.2f}**")
    else:
        st.info("Please upload the La Liga dataset (SP1_All_Seasons.csv).")

# --- Page: Visualize ---
elif page == "Visualize":
    st.subheader("La Liga Match Results Overview")
    if df_cleaned is not None and 'FTR' in df_cleaned.columns:
        label_mapping = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
        result_counts = df_cleaned['FTR'].value_counts().reindex(['H','D','A']).fillna(0).astype(int)
        labels = [label_mapping.get(k, k) for k in result_counts.index]

        col1, col2 = st.columns(2)
        palette = ['#2ecc71', '#95a5a6', '#e67e22']
        sns.set_style("darkgrid")

        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x=labels, y=result_counts.values, palette=palette, ax=ax)
            ax.set_title("Match Outcomes (Bar Chart)")
            ax.set_ylabel("Number of Matches")
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.pie(result_counts.values, labels=labels, autopct='%1.1f%%', startangle=90,
                    colors=palette, textprops={'color':'#f0f0f0', 'weight':'bold'})
            ax2.set_title("Outcome Distribution (Pie Chart)")
            st.pyplot(fig2)
    else:
        st.warning("Column 'FTR' not found in dataset.")

# --- Page: Predict Match ---
elif page == "Predict Match":
    st.subheader("Predict Match Outcome (Random Forest)")
    if df_cleaned is not None and rf is not None:
        teams = sorted(df_cleaned['HomeTeam'].unique())
        team1 = st.selectbox("Select Home Team", teams)
        team2 = st.selectbox("Select Away Team", teams)

        if team1 == team2:
            st.warning("Please select two different teams.")
        else:
            # --- FIX DATE FORMAT ---
            if 'Date' in df_cleaned.columns:
                df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], errors='coerce').dt.date

            # --- HEAD TO HEAD FIXTURES ---
            st.markdown(f"### 🔁 Head-to-Head Results: {team1} vs {team2} (Last 5 Seasons)")
            h2h = df_cleaned[
                ((df_cleaned['HomeTeam'] == team1) & (df_cleaned['AwayTeam'] == team2)) |
                ((df_cleaned['HomeTeam'] == team2) & (df_cleaned['AwayTeam'] == team1))
            ].sort_values(by='Date', ascending=False).head(10)

            if not h2h.empty:
                st.dataframe(h2h[['Date', 'HomeTeam', 'AwayTeam', 'FTR']])
            else:
                st.info("No recorded matches found between these teams in the last 5 seasons.")

            # --- PREDICT MATCH ---
            sample = df_model[
                (df_model['HomeTeam'] == label_encoders['HomeTeam'].transform([team1])[0]) &
                (df_model['AwayTeam'] == label_encoders['AwayTeam'].transform([team2])[0])
            ]

            if sample.empty:
                st.warning("No data found for this specific match-up.")
            else:
                sample = sample.drop(columns=['FTR'])
                pred_encoded = rf.predict(sample.select_dtypes(include=[np.number]))[0]
                pred_label = label_encoders['FTR'].inverse_transform([pred_encoded])[0]
                result_map = {'H': f"{team1} Win", 'D': "Draw", 'A': f"{team2} Win"}
                st.success(f"🎯 Predicted Outcome: **{result_map.get(pred_label, pred_label)}**")
                st.write(f"Model accuracy: **{accuracy:.2f}**")

            # --- LAST 5 GAMES FOR EACH TEAM ---
            st.markdown(f"### 📊 Last 5 Games for {team1}")
            last5_home = df_cleaned[((df_cleaned['HomeTeam'] == team1) | (df_cleaned['AwayTeam'] == team1))].tail(5)
            cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
            st.dataframe(last5_home[cols])

            st.markdown(f"### 📊 Last 5 Games for {team2}")
            last5_away = df_cleaned[((df_cleaned['HomeTeam'] == team2) | (df_cleaned['AwayTeam'] == team2))].tail(5)
            st.dataframe(last5_away[cols])

            # --- PERFORMANCE CHARTS ---
            st.markdown("### 📈 Team Performance Comparison")
            c1, c2 = st.columns(2)
            with c1:
                home_stats = df_cleaned[df_cleaned['HomeTeam'] == team1]['FTR'].value_counts()
                fig1, ax1 = plt.subplots()
                home_stats.plot(kind='bar', ax=ax1, color=['green', 'gray', 'orange'])
                ax1.set_title(f"{team1} - Home Stats")
                st.pyplot(fig1)
            with c2:
                away_stats = df_cleaned[df_cleaned['AwayTeam'] == team2]['FTR'].value_counts()
                fig2, ax2 = plt.subplots()
                away_stats.plot(kind='bar', ax=ax2, color=['green', 'gray', 'orange'])
                ax2.set_title(f"{team2} - Away Stats")
                st.pyplot(fig2)

            # --- GOAL DISTRIBUTION ---
            st.markdown("### ⚽ Goal Distribution")
            if 'FTHG' in df_cleaned.columns and 'FTAG' in df_cleaned.columns:
                home_goals = df_cleaned[df_cleaned['HomeTeam'] == team1]['FTHG']
                away_goals = df_cleaned[df_cleaned['AwayTeam'] == team2]['FTAG']
                fig3, ax3 = plt.subplots()
                ax3.hist([home_goals.dropna(), away_goals.dropna()],
                         bins=np.arange(0, 10) - 0.5,
                         label=[f"{team1} Goals", f"{team2} Goals"],
                         color=['green', 'orange'], alpha=0.7)
                ax3.set_title("Goal Distribution")
                ax3.set_xlabel("Goals")
                ax3.set_ylabel("Frequency")
                ax3.legend()
                st.pyplot(fig3)
            else:
                st.info("Goal columns missing in dataset.")
    else:
        st.info("Model not trained. Please load dataset first.")
