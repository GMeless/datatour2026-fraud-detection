# ================================================================
# DATATOUR 2026 — DÉTECTION DE FRAUDE MOBILE MONEY
# Équipe Datawinners
#
# PIPELINE COMPLET REPRODUCTIBLE — Configuration gagnante confirmée
# Score public  : 0.356478 (PR-AUC / Average Precision)
# Score privé   : 0.356984 (PR-AUC) — 2ème place, classement final officiel
#
# Ce script reproduit intégralement notre meilleure soumission,
# celle retenue pour le classement privé officiel (soumise le 02/07/2026).
# Chaque étape est commentée pour expliquer le "pourquoi", pas
# seulement le "comment" — utile pour un lecteur découvrant le
# projet ou souhaitant l'auditer.
#
# ARCHITECTURE ASYMÉTRIQUE (reconstituée à partir du notebook original
# d'exécution) : les deux modèles du blend n'utilisent PAS exactement
# le même jeu de variables — une différence intentionnelle héritée de
# l'ordre réel des expérimentations :
#   - LGB2 utilise 28 variables (incluant amount_rank_in_period)
#   - LGB3 utilise 27 variables (sans amount_rank_in_period), avec
#     une régularisation renforcée
#
# Pré-requis :
#   - train.csv et test.csv dans le même dossier que ce script
#   - pip install lightgbm pandas numpy scikit-learn
#
# Graine aléatoire fixée à 42 partout : la reproductibilité exacte
# du score annoncé dépend de cette constance.
# ================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# CONSTANTES DU PROJET
# ----------------------------------------------------------------
SEED = 42  # graine aléatoire — garantit la reproductibilité exacte
TARGET = "fraud_flag"
EPS = 1e-9  # petite constante anti-division-par-zéro

# Paramètres de lissage bayésien du Target Encoding, optimisés
# empiriquement par validation croisée sur la zone temporelle
# récente du train (periods 84-105), notre meilleur proxy du test.
M_COMPTE = 6   # lissage du taux de fraude par compte émetteur
M_HEURE = 6    # lissage du taux de fraude par compte x heure du cycle
M_PAIRE = 18   # lissage du taux de fraude par paire émetteur->destinataire

# Paramètres de la mémoire temporelle du risque (section 5.3 de la
# note méthodologique) : mélange 70% récent / 30% historique complet,
# sur une fenêtre glissante de 30 périodes.
FENETRE_RECENTE = 30
POIDS_RECENT = 0.7
POIDS_GLOBAL = 0.3

# Paramètre de la pondération exponentielle du risque (section 5.3) :
# chaque transaction contribue avec un poids exp(-LAMBDA * ancienneté).
LAMBDA_EXP = 0.05


# ================================================================
# ÉTAPE 1 — CHARGEMENT ET FILTRAGE MÉTIER
# ================================================================
# Justification : l'exploration initiale a montré que 100% des
# fraudes du jeu d'entraînement sont concentrées sur l'opération
# "op_03" (transfert pair-à-pair), avec un taux de fraude de 31.2%.
# Les quatre autres types d'opération ont un taux de fraude nul sur
# tout l'historique disponible. Entraîner un modèle sur l'ensemble
# du dataset diluerait le signal utile dans du bruit sans rapport
# avec la fraude. On restreint donc toute la modélisation à op_03,
# et on affecte un score de risque fixe et minimal (0.001) aux
# autres opérations lors de la génération du fichier de soumission.
# ================================================================

print("=" * 70)
print("ÉTAPE 1 — Chargement et filtrage")
print("=" * 70)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_op03 = train[train["operation"] == "op_03"].copy().reset_index(drop=True)
test_op03 = test[test["operation"] == "op_03"].copy().reset_index(drop=True)
y = train_op03[TARGET].astype(int)

print(f"Train complet     : {len(train):,} lignes")
print(f"Train op_03       : {len(train_op03):,} lignes ({y.mean():.1%} de fraude)")
print(f"Test complet       : {len(test):,} lignes")
print(f"Test op_03         : {len(test_op03):,} lignes")


# ================================================================
# ÉTAPE 2 — VARIABLES DE BASE (19 variables)
# ================================================================
# Ces transformations directes ne nécessitent aucune validation
# croisée particulière : elles ne font qu'appliquer une fonction à
# des colonnes déjà connues dans le test (amount, soldes), sans
# jamais utiliser la variable cible fraud_flag. Aucun risque de
# fuite de données ici.
# ================================================================

print("\n" + "=" * 70)
print("ÉTAPE 2 — Construction des variables de base")
print("=" * 70)

EPS_LOCAL = EPS
for df in [train_op03, test_op03]:
    # Le montant brut a une asymétrie mesurée de 6.60 (une gaussienne
    # aurait ~0) : quelques transactions énormes déforment toute
    # statistique calculée sur la valeur brute. Le logarithme
    # compresse cette asymétrie. log1p(x)=log(1+x) évite log(0).
    df["log_amount"] = np.log1p(df["amount"])
    df["log_dest_before"] = np.log1p(np.abs(df["destination_balance_before"]))
    df["log_origin_before"] = np.log1p(np.abs(df["origin_balance_before"]))

    # Variation réelle du solde avant/après — capture si l'argent a
    # vraiment bougé, et de combien.
    df["origin_balance_delta"] = df["origin_balance_after"] - df["origin_balance_before"]
    df["dest_balance_delta"] = df["destination_balance_after"] - df["destination_balance_before"]

    # Ratio montant / solde destinataire avant transaction : un ratio
    # élevé signale un compte destinataire "vidé" par une transaction
    # disproportionnée par rapport à ce qu'il détenait.
    df["amount_to_dest_ratio"] = np.clip(
        df["amount"] / (df["destination_balance_before"] + EPS_LOCAL), 0, 100
    )

    # Indicateurs binaires d'anomalie comptable, inspirés des
    # patterns caractéristiques de comptes "mules" ou de transactions
    # fantômes (solde qui ne bouge pas malgré un débit/crédit).
    df["dest_zero_before"] = (np.abs(df["destination_balance_before"]) < 0.01).astype(np.int8)
    df["dest_frozen"] = (
        np.abs(df["destination_balance_after"] - df["destination_balance_before"]) < 0.01
    ).astype(np.int8)
    df["origin_frozen"] = (
        np.abs(df["origin_balance_after"] - df["origin_balance_before"]) < 0.01
    ).astype(np.int8)
    df["origin_negative_after"] = (df["origin_balance_after"] < 0).astype(np.int8)
    df["amount_exceeds_balance"] = (df["amount"] > df["origin_balance_before"]).astype(np.int8)

    # period_mod24 : le cycle "heure de la journée" plutôt que la
    # période brute. La période brute est hors distribution en test
    # (periods 106-143, jamais vues en train) ; son modulo 24 reste
    # une valeur connue du train, donc exploitable par le modèle.
    df["period_mod24"] = df["period"] % 24

# Degré du compte : nombre de destinataires uniques pour un compte
# émetteur, nombre d'émetteurs uniques pour un compte destinataire.
# Calculé sur train+test combinés car cette statistique n'utilise
# jamais fraud_flag : aucun risque de fuite à l'utiliser ainsi, et
# cela rend la mesure plus complète et stable.
combined_op03 = pd.concat([train_op03, test_op03], axis=0, ignore_index=True)
orig_n_dest = (
    combined_op03.groupby("origin_account")["destination_account"]
    .nunique()
    .reset_index(name="orig_n_unique_dest")
)
dest_n_orig = (
    combined_op03.groupby("destination_account")["origin_account"]
    .nunique()
    .reset_index(name="dest_n_unique_orig")
)
train_op03 = train_op03.merge(orig_n_dest, on="origin_account", how="left")
train_op03 = train_op03.merge(dest_n_orig, on="destination_account", how="left")
test_op03 = test_op03.merge(orig_n_dest, on="origin_account", how="left")
test_op03 = test_op03.merge(dest_n_orig, on="destination_account", how="left")

FEATURES_BASE = [
    "amount", "log_amount",
    "origin_balance_before", "origin_balance_after",
    "destination_balance_before", "destination_balance_after",
    "log_origin_before", "log_dest_before",
    "origin_balance_delta", "dest_balance_delta",
    "amount_to_dest_ratio",
    "dest_zero_before", "dest_frozen", "origin_frozen",
    "origin_negative_after", "amount_exceeds_balance",
    "period_mod24", "orig_n_unique_dest", "dest_n_unique_orig",
]
print(f"{len(FEATURES_BASE)} variables de base construites.")


# ----------------------------------------------------------------
# Paramètres communs de validation, réutilisés à chaque étape
# ----------------------------------------------------------------
spw = (y == 0).sum() / (y == 1).sum()  # scale_pos_weight : corrige le déséquilibre 69%/31%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
mask_recent = train_op03["period"] >= 84  # zone récente = notre proxy du futur (test)
global_mean = y.mean()  # taux de fraude global — valeur de repli par défaut


# ================================================================
# ÉTAPE 3 — TARGET ENCODING (compte / heure / paire)
# ================================================================
# PRINCIPE : encoder chaque entité (compte, créneau horaire, paire)
# par son taux de fraude historique, avec lissage bayésien pour ne
# pas sur-interpréter les entités peu observées.
#
# GARDE-FOU CRITIQUE CONTRE LA FUITE DE DONNÉES : le taux de fraude
# de chaque entité est calculé UNIQUEMENT sur les 4 autres blocs de
# la validation croisée, jamais sur le bloc auquel il est appliqué.
# Un essai initial sans cette précaution avait produit un score
# hors-échantillon de 0.96 — un résultat statistiquement
# invraisemblable confirmant la fuite avant tout usage réel.
#
# Formule de lissage bayésien :
#   taux_lissé = (n_obs * taux_brut + m * taux_global) / (n_obs + m)
# Une entité peu observée (n_obs faible) est tirée vers la moyenne
# globale ; une entité très observée (n_obs élevé) garde son taux
# propre.
# ================================================================

print("\n" + "=" * 70)
print("ÉTAPE 3 — Target Encoding (compte / heure / paire)")
print("=" * 70)

col_te = "orig_te_m6"          # taux de fraude du compte émetteur
col_h = "orig_te_hour_m6"      # taux de fraude du compte émetteur x heure du cycle
col_pair = "pair_te_m18"       # taux de fraude de la paire émetteur -> destinataire

for col in [col_te, col_h, col_pair]:
    train_op03[col] = global_mean  # valeur initiale par défaut

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_op03, y)):
    tf = train_op03.iloc[tr_idx].copy()  # 4 blocs -> sert à CALCULER le taux
    vf = train_op03.iloc[val_idx].copy()  # 1 bloc -> REÇOIT le taux, ne le calcule jamais

    tf["hk"] = tf["origin_account"] + "_" + (tf["period"] % 24).astype(str)
    vf["hk"] = vf["origin_account"] + "_" + (vf["period"] % 24).astype(str)
    tf["pk"] = tf["origin_account"] + "_" + tf["destination_account"]
    vf["pk"] = vf["origin_account"] + "_" + vf["destination_account"]

    for col_name, grp_col, m in [
        (col_te, "origin_account", M_COMPTE),
        (col_h, "hk", M_HEURE),
        (col_pair, "pk", M_PAIRE),
    ]:
        s = tf.groupby(grp_col)[TARGET].agg(["sum", "count"])
        s.columns = ["n", "t"]
        s["v"] = (s["t"] * (s["n"] / s["t"]) + m * global_mean) / (s["t"] + m)
        key = vf["origin_account"] if grp_col == "origin_account" else vf[grp_col]
        train_op03.loc[val_idx, col_name] = key.map(s["v"]).fillna(global_mean).values

    print(f"  Bloc de validation {fold + 1}/5 traité")

# Pour le TEST : le calcul utilise tout le train (aucun risque de
# fuite possible, le test ne contient jamais fraud_flag).
train_op03["hk"] = train_op03["origin_account"] + "_" + (train_op03["period"] % 24).astype(str)
train_op03["pk"] = train_op03["origin_account"] + "_" + train_op03["destination_account"]
test_op03["hk"] = test_op03["origin_account"] + "_" + (test_op03["period"] % 24).astype(str)
test_op03["pk"] = test_op03["origin_account"] + "_" + test_op03["destination_account"]

for col_name, grp_col, m in [
    (col_te, "origin_account", M_COMPTE),
    (col_h, "hk", M_HEURE),
    (col_pair, "pk", M_PAIRE),
]:
    sf = train_op03.groupby(grp_col)[TARGET].agg(["sum", "count"])
    sf.columns = ["n", "t"]
    sf["v"] = (sf["t"] * (sf["n"] / sf["t"]) + m * global_mean) / (sf["t"] + m)
    key_col = "origin_account" if grp_col == "origin_account" else grp_col
    test_op03[col_name] = test_op03[key_col].map(sf["v"]).fillna(global_mean)

FEAT_FINAL = FEATURES_BASE + [col_te, col_h, col_pair]
print(f"Corrélations : compte={abs(train_op03[col_te].corr(y)):.4f}  "
      f"heure={abs(train_op03[col_h].corr(y)):.4f}  "
      f"paire={abs(train_op03[col_pair].corr(y)):.4f}")


# ================================================================
# ÉTAPE 4 — SIGNATURES COMPORTEMENTALES (behavior features)
# ================================================================
# PRINCIPE : la découverte selon laquelle 74 à 80% des comptes
# frauduleux sont des comptes "mixtes" (légitimes puis compromis)
# suggère qu'une fraude se manifeste comme une RUPTURE de
# comportement plutôt qu'un niveau de risque global. Ces variables
# comparent chaque transaction à l'historique propre du compte.
#
# PAS DE VALIDATION CROISÉE NÉCESSAIRE ICI : ces statistiques
# utilisent uniquement la colonne "amount", jamais fraud_flag.
# Aucun risque de fuite de la variable cible.
# ================================================================

print("\n" + "=" * 70)
print("ÉTAPE 4 — Signatures comportementales")
print("=" * 70)

orig_stats = train_op03.groupby("origin_account")["amount"].agg(
    global_mean_amount="mean",
    global_std_amount="std",
    global_n_tx="count",
).reset_index()
orig_stats["global_std_amount"] = orig_stats["global_std_amount"].fillna(0)

for df in [train_op03, test_op03]:
    df_m = df.merge(orig_stats, on="origin_account", how="left")
    df["global_mean_amount"] = df_m["global_mean_amount"].fillna(df["amount"])
    df["global_std_amount"] = df_m["global_std_amount"].fillna(0)
    df["global_n_tx"] = df_m["global_n_tx"].fillna(1)

    # Ratio montant actuel / montant moyen habituel du compte.
    df["amount_vs_account_mean"] = np.clip(
        df["amount"] / (df["global_mean_amount"] + EPS_LOCAL), 0, 10
    )
    # Écart-réduit (z-score) du montant par rapport à la dispersion
    # historique du compte — mesure la rareté statistique du montant.
    df["amount_zscore_account"] = np.clip(
        (df["amount"] - df["global_mean_amount"]) / (df["global_std_amount"] + EPS_LOCAL),
        -5, 10,
    )

BEHAVIOR_FEATURES = ["amount_vs_account_mean", "amount_zscore_account", "global_n_tx"]
print(f"{len(BEHAVIOR_FEATURES)} signatures comportementales construites.")


# ================================================================
# ÉTAPE 5 — MÉMOIRE TEMPORELLE DU RISQUE
# ================================================================
# PRINCIPE : le contrôle de stabilité temporelle a montré que le
# ratio fraude/normal du Target Encoding standard s'érode entre la
# zone ancienne du train (ratio 1.06x) et la zone récente (ratio
# 1.01x) — signe que le risque d'un compte évolue dans le temps.
# Confirmé par l'organisateur : "le risque peut évoluer dans le
# temps". Deux implémentations complémentaires sont retenues.
# ================================================================

print("\n" + "=" * 70)
print("ÉTAPE 5 — Mémoire temporelle du risque")
print("=" * 70)

# ---- 5a. Fenêtre glissante 70% récent / 30% historique complet ----
col_te_recent = "orig_te_recent"
train_op03[col_te_recent] = global_mean

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_op03, y)):
    tf = train_op03.iloc[tr_idx].copy()
    vf = train_op03.iloc[val_idx].copy()

    s_g = tf.groupby("origin_account")[TARGET].agg(["sum", "count"])
    s_g.columns = ["n", "t"]
    s_g["v_g"] = (s_g["t"] * (s_g["n"] / s_g["t"]) + M_COMPTE * global_mean) / (s_g["t"] + M_COMPTE)

    period_max = tf["period"].max()
    tf_rec = tf[tf["period"] >= period_max - FENETRE_RECENTE]
    s_r = tf_rec.groupby("origin_account")[TARGET].agg(["sum", "count"])
    s_r.columns = ["n_r", "t_r"]
    s_r["v_r"] = (s_r["t_r"] * (s_r["n_r"] / s_r["t_r"]) + M_COMPTE * global_mean) / (s_r["t_r"] + M_COMPTE)

    mg = s_g.join(s_r, how="left")
    mg["v_r"] = mg["v_r"].fillna(mg["v_g"])  # pas de transaction récente -> repli sur le taux global
    mg["v"] = POIDS_RECENT * mg["v_r"] + POIDS_GLOBAL * mg["v_g"]

    train_op03.loc[val_idx, col_te_recent] = (
        vf["origin_account"].map(mg["v"]).fillna(global_mean).values
    )

period_max_train = train_op03["period"].max()
s_g_f = train_op03.groupby("origin_account")[TARGET].agg(["sum", "count"])
s_g_f.columns = ["n", "t"]
s_g_f["v_g"] = (s_g_f["t"] * (s_g_f["n"] / s_g_f["t"]) + M_COMPTE * global_mean) / (s_g_f["t"] + M_COMPTE)
s_r_f = train_op03[train_op03["period"] >= period_max_train - FENETRE_RECENTE].groupby(
    "origin_account"
)[TARGET].agg(["sum", "count"])
s_r_f.columns = ["n_r", "t_r"]
s_r_f["v_r"] = (s_r_f["t_r"] * (s_r_f["n_r"] / s_r_f["t_r"]) + M_COMPTE * global_mean) / (s_r_f["t_r"] + M_COMPTE)
mg_f = s_g_f.join(s_r_f, how="left")
mg_f["v_r"] = mg_f["v_r"].fillna(mg_f["v_g"])
mg_f["v"] = POIDS_RECENT * mg_f["v_r"] + POIDS_GLOBAL * mg_f["v_g"]
test_op03[col_te_recent] = test_op03["origin_account"].map(mg_f["v"]).fillna(global_mean)

print(f"Fenêtre glissante (orig_te_recent) : corrélation = {abs(train_op03[col_te_recent].corr(y)):.4f}")


# ---- 5b. Pondération exponentielle continue ----
def compute_te_exp(train_df, val_df, grp_col, target, global_mean, lam, m=6):
    """
    Calcule un taux de fraude pondéré exponentiellement par
    l'ancienneté de chaque transaction, pour l'appliquer aux lignes
    de val_df sans jamais utiliser leurs propres labels.

    poids(transaction) = exp(-lam * (période_max - période_transaction))
    Une transaction de la période la plus récente a un poids proche
    de 1 ; une transaction ancienne a un poids qui décroît en douceur
    (pas de coupure brutale, contrairement à la fenêtre glissante).
    """
    period_max = train_df["period"].max()
    train_df = train_df.copy()
    train_df["exp_weight"] = np.exp(-lam * (period_max - train_df["period"]))

    num = train_df.groupby(grp_col).apply(
        lambda x: (x[target] * x["exp_weight"]).sum()
    ).reset_index(name="wf")
    den = train_df.groupby(grp_col)["exp_weight"].sum().reset_index(name="sw")
    stats = num.merge(den, on=grp_col)
    stats["taux"] = stats["wf"] / stats["sw"]
    stats["te"] = (stats["sw"] * stats["taux"] + m * global_mean) / (stats["sw"] + m)

    return val_df[grp_col].map(stats.set_index(grp_col)["te"]).fillna(global_mean)


col_exp = "te_exp_l5"
train_op03[col_exp] = global_mean

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_op03, y)):
    tf = train_op03.iloc[tr_idx]
    vf = train_op03.iloc[val_idx]
    train_op03.loc[val_idx, col_exp] = compute_te_exp(
        tf, vf, "origin_account", TARGET, global_mean, lam=LAMBDA_EXP, m=M_COMPTE
    ).values

test_op03[col_exp] = compute_te_exp(
    train_op03, test_op03, "origin_account", TARGET, global_mean, lam=LAMBDA_EXP, m=M_COMPTE
)

print(f"Pondération exponentielle (te_exp_l5) : corrélation = {abs(train_op03[col_exp].corr(y)):.4f}")

# Liste de base — 27 variables, utilisées telles quelles par LGB3
FEAT_EXP = list(dict.fromkeys(
    FEAT_FINAL + BEHAVIOR_FEATURES + [col_te_recent, col_exp]
))
print(f"\nNombre de variables (base, utilisées par LGB3) : {len(FEAT_EXP)}")


# ================================================================
# ÉTAPE 5bis — RANG DU MONTANT DANS SA PÉRIODE
# ================================================================
# PRINCIPE : comparaison inter-comptes plutôt qu'intra-compte —
# pour chaque transaction, son rang percentile parmi toutes les
# transactions ayant eu lieu à la même période. Un fraudeur qui
# frappe fort à un instant T se distingue des autres transactions
# de ce même instant, indépendamment de son propre historique.
#
# Calculé sur train+test combinés : aucune utilisation de la
# variable cible, donc aucun risque de fuite de données.
#
# NOTE IMPORTANTE : cette variable est utilisée UNIQUEMENT par LGB2
# dans la configuration retenue — LGB3 utilise FEAT_EXP seul (27
# variables, sans ce rang). Cette asymétrie entre les deux modèles
# du blend est intentionnelle et fait partie de la configuration
# exacte ayant produit le score privé de 0,356984.
# ================================================================

combined_rank = pd.concat([
    train_op03[["id", "period", "amount"]],
    test_op03[["id", "period", "amount"]],
], axis=0, ignore_index=True)

combined_rank["amount_rank_in_period"] = combined_rank.groupby("period")["amount"].rank(pct=True)

rank_map = combined_rank.set_index("id")["amount_rank_in_period"]
train_op03["amount_rank_in_period"] = train_op03["id"].map(rank_map)
test_op03["amount_rank_in_period"] = test_op03["id"].map(rank_map)

print(f"Corrélation amount_rank_in_period : {abs(train_op03['amount_rank_in_period'].corr(y)):.4f}")

# Liste étendue — 28 variables, utilisées uniquement par LGB2
FEAT_LGB2 = FEAT_EXP + ["amount_rank_in_period"]
print(f"Nombre de variables (étendu, utilisées par LGB2) : {len(FEAT_LGB2)}")


# ================================================================
# ÉTAPE 6 — MODÉLISATION : BLEND DE DEUX LIGHTGBM
# ================================================================
# PRINCIPE : deux modèles de complexité différente (nombre de
# feuilles) apprennent des patterns partiellement distincts sur les
# mêmes données. Leur moyenne, pondérée par leur propre score de
# validation croisée, atténue les erreurs indépendantes de chacun.
#
# Le choix de 255 et 1024 feuilles résulte d'une exploration
# empirique : en-deçà, les modèles sous-exploitent le signal
# disponible ; au-delà de ~1500 feuilles, le score public se
# dégrade malgré un score local en hausse (signe de sur-adaptation
# à la zone de validation plutôt qu'au véritable test).
# ================================================================

print("\n" + "=" * 70)
print("ÉTAPE 6 — Entraînement du blend LGB2 + LGB3")
print("=" * 70)

params_lgb2 = {
    "objective": "binary",
    "metric": "average_precision",
    "num_leaves": 255,
    "learning_rate": 0.03,
    "scale_pos_weight": spw,
    "feature_fraction": 0.8,   # sous-échantillonnage des variables à chaque arbre (anti sur-apprentissage)
    "bagging_fraction": 0.8,   # sous-échantillonnage des lignes à chaque itération
    "bagging_freq": 1,
    "reg_alpha": 0.5,          # régularisation L1
    "reg_lambda": 2.0,         # régularisation L2
    "min_child_samples": 30,
    "verbose": -1,
    "random_state": SEED + 100,
}

# NOTE : cette configuration correspond à la variante "régularisation
# forte" testée le 02/07/2026, qui a produit un score public légèrement
# inférieur (0.356478) à notre meilleure config locale mais un score
# PRIVÉ supérieur (0.356984) — confirmant que cette régularisation plus
# forte généralise mieux sur l'ensemble complet du jeu de test.
params_lgb3 = {
    "objective": "binary",
    "metric": "average_precision",
    "num_leaves": 1024,
    "learning_rate": 0.02,
    "scale_pos_weight": spw,
    "feature_fraction": 0.7,       # régularisation renforcée (était 0.8)
    "bagging_fraction": 0.7,       # régularisation renforcée (était 0.8)
    "bagging_freq": 1,
    "reg_alpha": 2.0,              # régularisation renforcée (était 0.3)
    "reg_lambda": 5.0,             # régularisation renforcée (était 1.5)
    "min_child_samples": 50,       # régularisation renforcée (était 20)
    "verbose": -1,
    "random_state": SEED + 200,
}

X_train_lgb2 = train_op03[FEAT_LGB2]
X_test_lgb2 = test_op03[FEAT_LGB2]
X_train_lgb3 = train_op03[FEAT_EXP]
X_test_lgb3 = test_op03[FEAT_EXP]

oof_lgb2 = np.zeros(len(X_train_lgb2))
oof_lgb3 = np.zeros(len(X_train_lgb3))
pred_lgb2 = np.zeros(len(X_test_lgb2))
pred_lgb3 = np.zeros(len(X_test_lgb3))

print(f"\nModèle LGB2 (255 feuilles, {len(FEAT_LGB2)} variables — inclut amount_rank_in_period)...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_lgb2, y)):
    ds_tr = lgb.Dataset(X_train_lgb2.iloc[tr_idx], label=y.iloc[tr_idx])
    ds_val = lgb.Dataset(X_train_lgb2.iloc[val_idx], label=y.iloc[val_idx], reference=ds_tr)
    model = lgb.train(
        params_lgb2, ds_tr, num_boost_round=3000,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=-1)],
    )
    oof_lgb2[val_idx] = model.predict(X_train_lgb2.iloc[val_idx], num_iteration=model.best_iteration)
    pred_lgb2 += model.predict(X_test_lgb2, num_iteration=model.best_iteration) / 5
    fold_score = average_precision_score(y.iloc[val_idx], oof_lgb2[val_idx])
    print(f"  Bloc {fold + 1}/5 : PR-AUC = {fold_score:.4f}")

lgb2_oof = average_precision_score(y, oof_lgb2)
lgb2_rec = average_precision_score(y[mask_recent.values], oof_lgb2[mask_recent.values])
print(f"LGB2 — score global (OOF) = {lgb2_oof:.4f}  |  score zone récente = {lgb2_rec:.4f}")

print(f"\nModèle LGB3 (1024 feuilles, {len(FEAT_EXP)} variables, régularisation renforcée)...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_lgb3, y)):
    ds_tr = lgb.Dataset(X_train_lgb3.iloc[tr_idx], label=y.iloc[tr_idx])
    ds_val = lgb.Dataset(X_train_lgb3.iloc[val_idx], label=y.iloc[val_idx], reference=ds_tr)
    model = lgb.train(
        params_lgb3, ds_tr, num_boost_round=3000,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=-1)],
    )
    oof_lgb3[val_idx] = model.predict(X_train_lgb3.iloc[val_idx], num_iteration=model.best_iteration)
    pred_lgb3 += model.predict(X_test_lgb3, num_iteration=model.best_iteration) / 5
    fold_score = average_precision_score(y.iloc[val_idx], oof_lgb3[val_idx])
    print(f"  Bloc {fold + 1}/5 : PR-AUC = {fold_score:.4f}")

lgb3_oof = average_precision_score(y, oof_lgb3)
lgb3_rec = average_precision_score(y[mask_recent.values], oof_lgb3[mask_recent.values])
print(f"LGB3 — score global (OOF) = {lgb3_oof:.4f}  |  score zone récente = {lgb3_rec:.4f}")


# ================================================================
# ÉTAPE 7 — BLEND PONDÉRÉ ET GÉNÉRATION DE LA SOUMISSION
# ================================================================
# La pondération de chaque modèle dans le blend final est
# proportionnelle à son propre score de validation croisée : le
# modèle le plus performant pèse naturellement plus dans la moyenne,
# sans réglage manuel arbitraire.
# ================================================================

print("\n" + "=" * 70)
print("ÉTAPE 7 — Blend final et génération de la soumission")
print("=" * 70)

w_lgb2, w_lgb3 = lgb2_oof, lgb3_oof
total_poids = w_lgb2 + w_lgb3

oof_blend = (w_lgb2 * oof_lgb2 + w_lgb3 * oof_lgb3) / total_poids
pred_blend = (w_lgb2 * pred_lgb2 + w_lgb3 * pred_lgb3) / total_poids

blend_oof = average_precision_score(y, oof_blend)
blend_rec = average_precision_score(y[mask_recent.values], oof_blend[mask_recent.values])

print(f"Poids LGB2 dans le blend : {w_lgb2 / total_poids:.1%}")
print(f"Poids LGB3 dans le blend : {w_lgb3 / total_poids:.1%}")
print(f"\nScore global du blend (OOF)       : {blend_oof:.4f}")
print(f"Score zone récente du blend (proxy) : {blend_rec:.4f}")

# ----------------------------------------------------------------
# Génération du fichier de soumission final
# ----------------------------------------------------------------
# Les transactions hors op_03 reçoivent un score fixe de 0.001 :
# aucune fraude n'a jamais été observée sur ces opérations dans le
# train, et plusieurs tests de seuils alternatifs (0.0001 à 0.05)
# n'ont produit aucune variation de score public, confirmant
# l'absence de fraude exploitable hors op_03 également en test.
# ----------------------------------------------------------------
submission = pd.DataFrame({"id": test["id"].values, "target": 0.001})
submission.loc[submission["id"].isin(test_op03["id"].values), "target"] = pred_blend

assert len(submission) == len(test), "La soumission doit couvrir toutes les lignes du test."
assert submission["target"].between(0, 1).all(), "Les scores doivent être des probabilités valides."

submission.to_csv("submission_finale_datawinners.csv", index=False)

print(f"\nProportion de scores > 0.5 : {(submission['target'] > 0.5).mean():.1%}")
print("\nFichier généré : submission_finale_datawinners.csv")
print("Ce fichier reproduit la soumission ayant obtenu :")
print("  - Score public : 0.356478")
print("  - Score privé  : 0.356984 (2ème place, classement final officiel)")
print("=" * 70)
