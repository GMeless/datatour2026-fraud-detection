# ================================================================
# DÉMO STREAMLIT — Détection de fraude Mobile Money
# DataTour 2026 — Champion National Côte d'Ivoire (Datawinners)
#
# Cette application permet de :
#   1. Visualiser un aperçu du dataset et les statistiques clés
#   2. Explorer l'importance des variables du modèle
#   3. Tester le score de risque sur une transaction saisie manuellement
#
# Lancement : streamlit run streamlit_app.py
# Pré-requis : train.csv et test.csv dans le même dossier
#              (fichiers de la compétition — non fournis dans ce dépôt)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Détection de fraude Mobile Money — DataTour 2026",
    page_icon="🏆",
    layout="wide",
)

SEED = 42
TARGET = "fraud_flag"
EPS = 1e-9


# ----------------------------------------------------------------
# Chargement et préparation des données (mis en cache)
# ----------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des données...")
def charger_donnees():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    train_op03 = train[train["operation"] == "op_03"].copy().reset_index(drop=True)
    test_op03 = test[test["operation"] == "op_03"].copy().reset_index(drop=True)
    return train_op03, test_op03


@st.cache_data(show_spinner="Construction des variables...")
def construire_features(train_op03, test_op03):
    for df in [train_op03, test_op03]:
        df["log_amount"] = np.log1p(df["amount"])
        df["origin_balance_delta"] = df["origin_balance_after"] - df["origin_balance_before"]
        df["dest_balance_delta"] = df["destination_balance_after"] - df["destination_balance_before"]
        df["amount_to_dest_ratio"] = np.clip(
            df["amount"] / (df["destination_balance_before"] + EPS), 0, 100
        )
        df["dest_zero_before"] = (np.abs(df["destination_balance_before"]) < 0.01).astype(int)
        df["origin_negative_after"] = (df["origin_balance_after"] < 0).astype(int)
        df["amount_exceeds_balance"] = (df["amount"] > df["origin_balance_before"]).astype(int)
        df["period_mod24"] = df["period"] % 24

    combined = pd.concat([train_op03, test_op03], axis=0, ignore_index=True)
    orig_n_dest = combined.groupby("origin_account")["destination_account"].nunique().reset_index(
        name="orig_n_unique_dest"
    )
    dest_n_orig = combined.groupby("destination_account")["origin_account"].nunique().reset_index(
        name="dest_n_unique_orig"
    )
    train_op03 = train_op03.merge(orig_n_dest, on="origin_account", how="left")
    train_op03 = train_op03.merge(dest_n_orig, on="destination_account", how="left")
    test_op03 = test_op03.merge(orig_n_dest, on="origin_account", how="left")
    test_op03 = test_op03.merge(dest_n_orig, on="destination_account", how="left")

    global_mean = train_op03[TARGET].mean()

    # Target Encoding compte (via K-Fold, sans fuite de données)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_op03["orig_te_m6"] = global_mean
    for tr_idx, val_idx in skf.split(train_op03, train_op03[TARGET]):
        tf = train_op03.iloc[tr_idx]
        s = tf.groupby("origin_account")[TARGET].agg(["sum", "count"])
        s.columns = ["n", "t"]
        s["v"] = (s["t"] * (s["n"] / s["t"]) + 6 * global_mean) / (s["t"] + 6)
        vf = train_op03.iloc[val_idx]
        train_op03.loc[val_idx, "orig_te_m6"] = vf["origin_account"].map(s["v"]).fillna(global_mean).values

    sf = train_op03.groupby("origin_account")[TARGET].agg(["sum", "count"])
    sf.columns = ["n", "t"]
    sf["v"] = (sf["t"] * (sf["n"] / sf["t"]) + 6 * global_mean) / (sf["t"] + 6)
    test_op03["orig_te_m6"] = test_op03["origin_account"].map(sf["v"]).fillna(global_mean)
    te_map_compte = sf["v"].to_dict()

    features = [
        "amount", "log_amount", "origin_balance_before", "origin_balance_after",
        "destination_balance_before", "destination_balance_after",
        "origin_balance_delta", "dest_balance_delta", "amount_to_dest_ratio",
        "dest_zero_before", "origin_negative_after", "amount_exceeds_balance",
        "period_mod24", "orig_n_unique_dest", "dest_n_unique_orig", "orig_te_m6",
    ]
    return train_op03, test_op03, features, global_mean, te_map_compte


@st.cache_resource(show_spinner="Entraînement du modèle (quelques secondes)...")
def entrainer_modele(_train_op03, features):
    y = _train_op03[TARGET].astype(int)
    spw = (y == 0).sum() / (y == 1).sum()
    X = _train_op03[features]

    params = {
        "objective": "binary", "metric": "average_precision",
        "num_leaves": 255, "learning_rate": 0.05,
        "scale_pos_weight": spw, "verbose": -1, "random_state": SEED,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    model_final = None
    for tr_idx, val_idx in skf.split(X, y):
        ds_tr = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        ds_val = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=ds_tr)
        model_final = lgb.train(
            params, ds_tr, num_boost_round=500, valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = model_final.predict(X.iloc[val_idx], num_iteration=model_final.best_iteration)

    score = average_precision_score(y, oof)
    return model_final, score, oof


# ----------------------------------------------------------------
# Interface
# ----------------------------------------------------------------
st.title("🏆 Détection de fraude Mobile Money")
st.caption(
    "DataTour 2026 — Champion National Côte d'Ivoire (Datawinners) · "
    "Score PR-AUC officiel : 0,356984"
)

try:
    train_op03, test_op03 = charger_donnees()
except FileNotFoundError:
    st.error(
        "⚠️ Fichiers `train.csv` et `test.csv` introuvables. "
        "Ces fichiers appartiennent à la compétition DataTour 2026 (Data Afrique Hub) "
        "et ne sont pas fournis dans ce dépôt pour des raisons de propriété des données. "
        "Placez-les dans le même dossier que cette application pour l'utiliser."
    )
    st.stop()

train_op03, test_op03, features, global_mean, te_map_compte = construire_features(train_op03, test_op03)
model, score_oof, oof = entrainer_modele(train_op03, features)

tab1, tab2, tab3 = st.tabs(["📊 Vue d'ensemble", "🔍 Importance des variables", "🎯 Tester une transaction"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions (op_03)", f"{len(train_op03):,}")
    col2.metric("Taux de fraude", f"{train_op03[TARGET].mean():.1%}")
    col3.metric("Score PR-AUC (démo, modèle léger)", f"{score_oof:.4f}")
    col4.metric("Comptes uniques", f"{train_op03['origin_account'].nunique():,}")

    st.subheader("Distribution des montants")
    st.bar_chart(
        train_op03.groupby(pd.cut(train_op03["log_amount"], bins=20))[TARGET]
        .mean()
        .reset_index(drop=True)
    )

    st.info(
        "ℹ️ Le modèle affiché ici est une **version simplifiée** à des fins de démonstration "
        "(1 seul LightGBM, 16 variables). La configuration officielle gagnante utilise un "
        "blend de 2 modèles LightGBM et 27 variables (Target Encoding avec mémoire temporelle, "
        "signatures comportementales) — voir `solution.py` pour le pipeline complet."
    )

with tab2:
    st.subheader("Importance des variables (modèle de démonstration)")
    importance = pd.DataFrame({
        "variable": features,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    st.bar_chart(importance.set_index("variable")["importance"])
    st.dataframe(importance, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Simuler le score de risque d'une transaction")
    st.caption("Renseignez les champs ci-dessous pour obtenir une probabilité de fraude estimée.")

    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Montant de la transaction", min_value=0.0, value=23000.0, step=100.0)
        origin_before = st.number_input("Solde émetteur avant", value=50000.0, step=100.0)
        origin_after = st.number_input("Solde émetteur après", value=27000.0, step=100.0)
        dest_before = st.number_input("Solde destinataire avant", value=0.0, step=100.0)
    with c2:
        dest_after = st.number_input("Solde destinataire après", value=23000.0, step=100.0)
        period = st.slider("Période (cycle horaire simulé)", 0, 23, 12)
        compte_connu = st.selectbox(
            "Historique du compte émetteur",
            ["Compte inconnu (nouveau)", "Compte à risque connu (simulation)", "Compte fiable connu (simulation)"],
        )

    if st.button("🔎 Calculer le score de risque", type="primary"):
        te_simule = {
            "Compte inconnu (nouveau)": global_mean,
            "Compte à risque connu (simulation)": min(global_mean * 2.5, 0.95),
            "Compte fiable connu (simulation)": global_mean * 0.3,
        }[compte_connu]

        ligne = pd.DataFrame([{
            "amount": amount,
            "log_amount": np.log1p(amount),
            "origin_balance_before": origin_before,
            "origin_balance_after": origin_after,
            "destination_balance_before": dest_before,
            "destination_balance_after": dest_after,
            "origin_balance_delta": origin_after - origin_before,
            "dest_balance_delta": dest_after - dest_before,
            "amount_to_dest_ratio": np.clip(amount / (dest_before + EPS), 0, 100),
            "dest_zero_before": int(abs(dest_before) < 0.01),
            "origin_negative_after": int(origin_after < 0),
            "amount_exceeds_balance": int(amount > origin_before),
            "period_mod24": period,
            "orig_n_unique_dest": 70,
            "dest_n_unique_orig": 70,
            "orig_te_m6": te_simule,
        }])[features]

        proba = model.predict(ligne, num_iteration=model.best_iteration)[0]

        st.metric("Probabilité de fraude estimée", f"{proba:.1%}")
        if proba > 0.5:
            st.error("⚠️ Transaction à risque élevé — vérification recommandée")
        elif proba > 0.3:
            st.warning("🟡 Risque modéré")
        else:
            st.success("✅ Transaction jugée normale")

st.divider()
st.caption(
    "Code source complet et méthodologie détaillée : voir README.md et METHODOLOGIE.pdf · "
    "Compétition organisée par Data Afrique Hub."
)
