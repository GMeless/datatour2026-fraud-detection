# 🏆 DataTour 2026 — Détection de fraude Mobile Money

**Champion National Côte d'Ivoire — 1ère place / 32 équipes** | Data Afrique Hub

Score PR-AUC : **0,356984** (reproductibilité validée à l'identique lors de l'audit officiel)

---

## 📋 Contexte

DataTour 2026 (phase nationale) proposait un défi de détection de fraude sur un jeu de données de transactions Mobile Money de **1,3 million de lignes**. L'objectif : prédire la probabilité de fraude sur un jeu de test strictement postérieur au jeu d'entraînement, sans aucun chevauchement temporel — un exercice de prévision, pas de simple classification.

**Métrique d'évaluation** : PR-AUC (Average Precision), choisie pour sa robustesse face au déséquilibre de classes (un modèle prédisant "jamais de fraude" afficherait déjà 69 % d'exactitude sans jamais rien détecter).

---

## 🔍 Découvertes clés

- **100 % des fraudes** concentrées sur un seul type d'opération (transfert pair-à-pair), avec un taux de 31,2 % — la modélisation a été restreinte à ce périmètre exact.
- **74 à 80 % des comptes frauduleux** avaient un historique parfaitement légitime avant l'incident — cohérent avec un scénario de piratage de carte SIM (SIM swap) plutôt que des comptes créés pour frauder.
- **Le risque évolue dans le temps** : un test de validation adversariale (AUC = 0,72) a confirmé une dérive de comportement entre le passé (train) et le futur (test).

---

## 🛠️ Architecture de la solution

```
1. Filtrage métier sur l'opération concentrant 100 % du signal de fraude
2. 19 variables de base (montants log-transformés, soldes, ratios,
   indicateurs d'incohérence comptable, degré du compte)
3. Target Encoding par validation croisée à 5 blocs :
   - taux de fraude par compte (m=6)
   - taux de fraude par compte × heure (m=6)
   - taux de fraude par paire émetteur→destinataire (m=18)
4. Signatures comportementales (écart du montant à l'historique du compte)
5. Mémoire temporelle du risque :
   - fenêtre glissante 70% récent / 30% historique
   - pondération exponentielle continue (λ=0,05)
6. Blend pondéré de 2 modèles LightGBM (255 et 1024 feuilles),
   pondération proportionnelle au score de validation croisée
```

**Stack** : Python · LightGBM · Pandas · NumPy · Scikit-learn

---

## ⚠️ Rigueur méthodologique — le vrai enjeu

Le risque le plus critique de ce projet était la **fuite de données** (data leakage) : plusieurs variables dérivent directement de la variable cible. Un essai initial sans validation croisée a produit un score halluciné de 0,96 — la preuve qu'un score local trop beau doit toujours être questionné avant d'être exploité.

**Garde-fous appliqués systématiquement :**
- Toute variable dérivée du taux de fraude est calculée en validation croisée stricte (jamais sur son propre groupe)
- Contrôle de stabilité temporelle entre zone ancienne et zone récente pour chaque variable
- Aucune conclusion tirée sans confirmation sur la plateforme réelle
- **Plus de 40 expérimentations documentées**, y compris les échecs, avec diagnostic précis de chaque piste écartée

---

## 📊 Résultats

| Étape | Score public (PR-AUC) |
|---|---|
| Référence (19 variables de base) | 0,349614 |
| + Target Encoding (compte / heure / paire) | 0,353133 |
| + Mémoire temporelle du risque | 0,355446 |
| + Blend de modèles optimisé | 0,356104 |
| **Configuration finale retenue** | **0,356984** |

**Classement final : 1ère place nationale sur 32 équipes**, qualifié pour la phase internationale.

---

## 🚀 Démo interactive (Streamlit)

Une application permet d'explorer les données, l'importance des variables, et de
tester le score de risque sur une transaction saisie manuellement.

```bash
pip install streamlit
streamlit run streamlit_app.py
```

> ⚠️ Nécessite `train.csv` et `test.csv` (fichiers de la compétition, non fournis
> dans ce dépôt) placés dans le même dossier.

---

## 📁 Contenu du dépôt

```
├── solution.py            # Pipeline complet, reproductible de bout en bout
├── streamlit_app.py        # Démo interactive (dashboard + scoring)
├── requirements.txt         # Dépendances exactes
├── METHODOLOGIE.pdf          # Note méthodologique détaillée (11 sections)
├── LICENSE                    # Licence MIT (hors données de la compétition)
├── .gitignore                  # Exclut les données brutes et fichiers sensibles
└── README.md                    # Ce fichier
```

## 🔄 Reproductibilité

```bash
pip install -r requirements.txt
python solution.py
```

Graine aléatoire fixée (`SEED=42`) sur l'ensemble des opérations stochastiques — aucune donnée externe, aucune connexion réseau requise.

---

## 🎯 Prochaine étape

Équipe qualifiée pour la **phase internationale de DataTour 2026** (Data Afrique Hub).
Le sujet du défi international n'a pas encore été communiqué à ce stade — cette
section sera mise à jour dès son annonce officielle.

---

*Compétition organisée par [Data Afrique Hub](https://dataafriquehub.org)*
