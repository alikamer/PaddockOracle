################################################
# PaddockOracle - Streamlit arayüzü
################################################

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

from src.utils import f1_data_prep

st.set_page_config(page_title="PaddockOracle", page_icon="🏁", layout="wide")

TABLO_YUKSEKLIGI = 420


st.markdown("""
<style>
    .block-container {padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1200px;}
    h1 {font-weight: 600; letter-spacing: -.02em; margin-bottom: .1rem;}
    [data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 600;}
    [data-testid="stMetricLabel"] {font-size: .72rem; text-transform: uppercase;
                                   letter-spacing: .07em; opacity: .6;}
    [data-testid="stMetricDelta"] {font-size: .78rem;}
    [data-testid="stTab"] {height: auto !important;
                           padding: .85rem 1.75rem !important;
                           border-radius: .5rem .5rem 0 0;}
    [data-testid="stTab"] p {font-size: 1.1rem !important;
                             font-weight: 600 !important;}
    [data-testid="stTab"]:hover {background: rgba(250, 250, 250, .05);}
    [data-testid="stTab"][aria-selected="true"] {background: rgba(250, 250, 250, .08);}
    [data-testid="stTabPanel"] {padding-top: 1.4rem;}
</style>
""", unsafe_allow_html=True)


################################################
# Modeli yükleme
################################################

@st.cache_resource
def load_model():
    return joblib.load("models/f1_model.pkl")


@st.cache_data
def load_predictions():
    df = pd.read_csv("data/data_processed/results_2010_2025_features.csv")
    X, _ = f1_data_prep(df)
    df["podium_prob"] = load_model().predict_proba(X)[:, 1]
    return df


@st.cache_data
def load_2026():
    new = pd.read_csv("data/data_processed/results_2026_features.csv")
    old = pd.read_csv("data/data_processed/results_2010_2025_features.csv")

    both = pd.concat([old, new], ignore_index=True)
    X, _ = f1_data_prep(both)
    X = X.reindex(columns=load_model().feature_names_in_, fill_value=0)

    new = new.copy()
    new["podium_prob"] = load_model().predict_proba(X.iloc[-len(new):])[:, 1]
    return new


@st.cache_data
def load_calendar():
    return pd.read_csv("data/data_processed/calendar_2026.csv")


def race_table(df, season, round_):
    race = df[(df["season"] == season) & (df["round"] == round_)]
    cols = ["driver_id", "team_unified", "grid", "position", "is_podium", "podium_prob"]
    return race[cols].sort_values("podium_prob", ascending=False)


def overall_hit_rate(df):
    hits, total = 0, 0
    for _, race in df.groupby(["season", "round"]):
        predicted = set(race.nlargest(3, "podium_prob")["driver_id"])
        actual = set(race.loc[race["is_podium"] == 1, "driver_id"])
        hits += len(predicted & actual)
        total += 3
    return hits, total


def predict_race(df, model, race_rows, circuit):
    rookie = df[df["career_races"] <= 20]
    derived = ["rolling_form", "season_points", "team_form", "circuit_history",
               "teammate_delta", "dnf_rate", "grid_gain"]

    rows = []
    for r in race_rows.itertuples():
        d = df[df["driver_id"] == r.driver_id].sort_values(["season", "round"])
        if len(d):
            row = d.iloc[-1].copy()
        else:
            row = df.iloc[-1].copy()
            for c in derived:
                row[c] = rookie[c].median()
            row["career_races"] = 0

        row["driver_id"] = r.driver_id
        row["team_unified"] = r.team_unified
        row["circuit_id"] = circuit
        row["grid"] = r.grid

        past = df[(df["driver_id"] == r.driver_id) & (df["circuit_id"] == circuit)]
        if len(past):
            row["circuit_history"] = past["position"].mean()

        team_rows = df[df["team_unified"] == r.team_unified].sort_values(["season", "round"])
        if len(team_rows):
            row["team_form"] = team_rows.iloc[-1]["team_form"]

        rows.append(row)

    new = pd.DataFrame(rows)

    tmp = pd.concat([df, new], ignore_index=True)
    tmp = tmp.drop(columns=["podium_prob"], errors="ignore")
    X, _ = f1_data_prep(tmp)

    X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    probs = model.predict_proba(X.iloc[-len(new):])[:, 1]

    out = new[["driver_id", "team_unified", "grid"]].copy()
    out["podium_prob"] = probs
    return out.sort_values("podium_prob", ascending=False).reset_index(drop=True)


################################################
# Sayfa
################################################

df = load_predictions()

st.title("PaddockOracle")
st.caption("Formula 1 podium prediction · Trained on 2010–2025 race results")

tab1, tab2 = st.tabs(["2010–2025 · Training data", "2026 · Unseen test"])

with tab1:

    hits, total = overall_hit_rate(df)
    races = df[["season", "round"]].drop_duplicates().shape[0]

    col1, col2, col3, col4 = st.columns([1, 1, 2.2, 1.6])
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Races", races)
    col3.metric("Model", "Voting · LR·RF·CatBoost")
    col4.metric("Top-3 accuracy", f"{100 * hits / total:.0f}%", f"{hits}/{total} correct")

    st.divider()

    with st.sidebar:
        st.header("Select race")
        season = st.selectbox("Season", sorted(df["season"].unique(), reverse=True))

        season_races = (df[df["season"] == season][["round", "race_name"]]
                        .drop_duplicates()
                        .sort_values("round"))
        names = dict(zip(season_races["round"], season_races["race_name"]))
        round_ = st.selectbox("Grand Prix", season_races["round"],
                              format_func=lambda r: names[r])

    race = race_table(df, season, round_)
    predicted = race.head(3)
    actual = race[race["is_podium"] == 1].sort_values("position")

    st.subheader(f"{names[round_]}")
    st.caption(f"{season} season · round {round_}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Predicted podium**")
        for i, row in enumerate(predicted.itertuples(), start=1):
            mark = ":green[✓]" if row.is_podium == 1 else ":red[✗]"
            st.markdown(f"{i}. **{row.driver_id}** · {row.team_unified} "
                        f"&nbsp;&nbsp;{100 * row.podium_prob:.0f}% &nbsp;{mark}")

    with right:
        st.markdown("**Actual podium**")
        for i, row in enumerate(actual.itertuples(), start=1):
            st.markdown(f"{i}. **{row.driver_id}** · {row.team_unified}")

    race_hits = len(set(predicted["driver_id"]) & set(actual["driver_id"]))
    st.metric("Correct this race", f"{race_hits}/3")

    st.divider()

    st.markdown("**All drivers** · ranked by predicted probability")
    table = race.rename(columns={"driver_id": "Driver", "team_unified": "Team",
                                 "grid": "Grid", "position": "Finish",
                                 "is_podium": "Podium", "podium_prob": "Probability"})
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Probability": st.column_config.ProgressColumn(
                     "Probability", format="%.2f", min_value=0.0, max_value=1.0)})

    st.bar_chart(race.head(10).set_index("driver_id")["podium_prob"])

################################################
# 2026 — gerçek test
################################################

with tab2:
    new26 = load_2026()
    calendar = load_calendar()

    st.markdown("### Races the model has never seen")
    st.caption("Training stopped at the end of 2025. Every 2026 race below was held out, "
               "so these numbers are a genuine out-of-sample test.")

    done_hits, done_total = overall_hit_rate(new26)
    done_races = new26["round"].nunique()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Races run", f"{done_races} of {len(calendar)}")
    m2.metric("Predictions", done_total)
    m3.metric("Top-3 accuracy", f"{100 * done_hits / done_total:.0f}%",
              f"{done_hits}/{done_total} correct")
    m4.metric("ROC-AUC", f"{roc_auc_score(new26['is_podium'], new26['podium_prob']):.3f}")

    st.divider()

    labels = {r.round: f"{'●' if r.done else '○'}  {r.race_name}"
              for r in calendar.itertuples()}
    sel = st.selectbox("Grand Prix", calendar["round"], format_func=lambda r: labels[r],
                       key="race_2026")

    row = calendar[calendar["round"] == sel].iloc[0]
    st.subheader(f"{row.race_name}")
    st.caption(f"{row['date']} · {row['circuit_id']} · round {row['round']}")

    if row["done"]:
        race26 = (new26[new26["round"] == sel]
                  .sort_values("podium_prob", ascending=False))
        pred = race26.head(3)
        real = race26[race26["is_podium"] == 1].sort_values("position")

        left26, right26 = st.columns(2)
        with left26:
            st.markdown("**Predicted podium**")
            for i, r in enumerate(pred.itertuples(), start=1):
                mark = ":green[✓]" if r.is_podium == 1 else ":red[✗]"
                st.markdown(f"{i}. **{r.driver_id}** · {r.team_unified} "
                            f"&nbsp;&nbsp;{100 * r.podium_prob:.0f}% &nbsp;{mark}")
        with right26:
            st.markdown("**Actual podium**")
            for i, r in enumerate(real.itertuples(), start=1):
                st.markdown(f"{i}. **{r.driver_id}** · {r.team_unified}")

        st.metric("Correct this race",
                  f"{len(set(pred['driver_id']) & set(real['driver_id']))}/3")

        st.markdown("**All drivers**")
        st.dataframe(
            race26[["driver_id", "team_unified", "grid", "position",
                    "is_podium", "podium_prob"]]
            .rename(columns={"driver_id": "Driver", "team_unified": "Team",
                             "grid": "Grid", "position": "Finish",
                             "is_podium": "Podium", "podium_prob": "Probability"}),
            hide_index=True, width="stretch",
            column_config={"Probability": st.column_config.ProgressColumn(
                "Probability", format="%.2f", min_value=0.0, max_value=1.0)})

    else:
        st.info("Not raced yet. The starting grid is set in qualifying — "
                "fill it in below and the model will predict the podium.")

        last_round = new26["round"].max()
        taslak = (new26[new26["round"] == last_round]
                  [["driver_id", "team_unified", "grid"]]
                  .sort_values("grid")
                  .reset_index(drop=True))
        st.caption(f"Pre-filled with the line-up from round {last_round}, the most recent race.")

        podyum_yeri = st.container()

        edit26, res26 = st.columns([1, 1])
        with edit26:
            st.markdown("**Starting grid** · editable")
            edited26 = st.data_editor(
                taslak, hide_index=True, width="stretch", num_rows="dynamic",
                height=TABLO_YUKSEKLIGI, key="editor_2026",
                column_config={
                    "driver_id": st.column_config.SelectboxColumn(
                        "Driver", options=sorted(new26["driver_id"].unique()), required=True),
                    "team_unified": st.column_config.SelectboxColumn(
                        "Team", options=sorted(new26["team_unified"].unique()), required=True),
                    "grid": st.column_config.NumberColumn(
                        "Grid", min_value=1, max_value=22, step=1),
                })

        clean26 = edited26.dropna(subset=["driver_id", "team_unified", "grid"])

        if len(clean26) < 3:
            podyum_yeri.info("At least 3 drivers are needed to predict.")
        else:
            gecmis = pd.concat([df, new26], ignore_index=True)
            sonuc = predict_race(gecmis, load_model(), clean26, row["circuit_id"])

            with podyum_yeri:
                st.markdown("**Predicted podium**")
                for i, r in enumerate(sonuc.head(3).itertuples(), start=1):
                    st.markdown(f"{i}. **{r.driver_id}** · {r.team_unified} · "
                                f"from P{int(r.grid)} &nbsp;&nbsp;{100 * r.podium_prob:.0f}%")

            with res26:
                st.markdown("**All drivers** · ranked by predicted probability")
                st.dataframe(
                    sonuc.rename(columns={"driver_id": "Driver", "team_unified": "Team",
                                          "grid": "Grid", "podium_prob": "Probability"}),
                    hide_index=True, width="stretch", height=TABLO_YUKSEKLIGI,
                    column_config={"Probability": st.column_config.ProgressColumn(
                        "Probability", format="%.2f", min_value=0.0, max_value=1.0)})
