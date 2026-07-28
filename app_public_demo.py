# ================================================================
# APP PUBLIQUE — Détection de fraude Mobile Money (démo)
# DataTour 2026 — Champion National Côte d'Ivoire (Datawinners)
#
# Version conçue pour un déploiement public (Streamlit Community Cloud) :
# elle charge UNIQUEMENT le modèle pré-entraîné (model_demo.lgb) et ses
# métadonnées (model_meta.json) — jamais train.csv ni test.csv, qui
# restent propriété de la compétition et ne quittent jamais l'ordinateur
# où ils ont été utilisés pour l'entraînement (voir export_model.py).
#
# Lancement local : streamlit run app_public_demo.py
# ================================================================

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st

st.set_page_config(
    page_title="Détection de fraude Mobile Money — DataTour 2026",
    page_icon="🏆",
    layout="centered",
)

EPS = 1e-9


@st.cache_resource(show_spinner="Chargement du modèle...")
def charger_modele():
    model = lgb.Booster(model_file="model_demo.lgb")
    with open("model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


st.title("🏆 Détection de fraude Mobile Money")
st.caption(
    "DataTour 2026 — Champion National Côte d'Ivoire (Datawinners) · "
    "Score PR-AUC officiel : 0,356984 · Qualifié Phase Internationale"
)

st.markdown(
    """
Cette démo utilise un **modèle simplifié** entraîné sur les données de la
compétition DataTour 2026 (Data Afrique Hub) — détection de fraude sur des
transactions Mobile Money de type transfert pair-à-pair.

> ℹ️ La configuration officielle gagnante utilise un blend de 2 modèles
> LightGBM et 27 variables (mémoire temporelle du risque, signatures
> comportementales). Cette démo utilise une version allégée (16 variables,
> 1 seul modèle) à des fins d'illustration.
"""
)

try:
    model, meta = charger_modele()
except Exception:
    st.error(
        "⚠️ Fichiers `model_demo.lgb` / `model_meta.json` introuvables. "
        "Génère-les d'abord en local avec `python export_model.py` "
        "(nécessite train.csv, non fourni dans ce dépôt public)."
    )
    st.stop()

FEATURES = meta["features"]
global_mean = meta["global_mean"]

st.divider()
st.subheader("🎯 Simuler le score de risque d'une transaction")

c1, c2 = st.columns(2)
with c1:
    amount = st.number_input("Montant de la transaction", min_value=0.0, value=23000.0, step=100.0)
    origin_before = st.number_input("Solde émetteur avant", value=50000.0, step=100.0)
    origin_after = st.number_input("Solde émetteur après", value=27000.0, step=100.0)
    dest_before = st.number_input("Solde destinataire avant", value=0.0, step=100.0)
with c2:
    dest_after = st.number_input("Solde destinataire après", value=23000.0, step=100.0)
    period = st.slider("Période (cycle horaire simulé, 0-23)", 0, 23, 12)
    compte_connu = st.selectbox(
        "Historique du compte émetteur (simulation)",
        ["Compte inconnu (nouveau)", "Compte à risque connu", "Compte fiable connu"],
    )
    orig_n_dest = st.slider("Nombre de destinataires uniques du compte", 1, 150, 70)

if st.button("🔎 Calculer le score de risque", type="primary", use_container_width=True):
    te_simule = {
        "Compte inconnu (nouveau)": global_mean,
        "Compte à risque connu": min(global_mean * 2.5, 0.95),
        "Compte fiable connu": global_mean * 0.3,
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
        "orig_n_unique_dest": orig_n_dest,
        "dest_n_unique_orig": 70,
        "orig_te_m6": te_simule,
    }])[FEATURES]

    proba = model.predict(ligne, num_iteration=meta["best_iteration"])[0]

    st.metric("Probabilité de fraude estimée", f"{proba:.1%}")
    barre = st.progress(min(float(proba), 1.0))

    if proba > 0.5:
        st.error("⚠️ Transaction à risque élevé — vérification recommandée")
    elif proba > 0.3:
        st.warning("🟡 Risque modéré")
    else:
        st.success("✅ Transaction jugée normale")

st.divider()
with st.expander("📊 Importance des variables du modèle"):
    importance = pd.DataFrame({
        "variable": FEATURES,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    st.bar_chart(importance.set_index("variable")["importance"])

st.caption(
    "Code source complet, méthodologie et pipeline gagnant : "
    "voir le dépôt GitHub · Compétition organisée par Data Afrique Hub."
)
