from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# 1. Initialisation de l'application Flask
# --------------------------------------------------

app = Flask(__name__)
#Historiques de 10 premières analyses
historique_analyses=[]

# Répertoire contenant app.py et les fichiers du modèle
BASE_DIR = Path(__file__).resolve().parent

# Chargement du modèle K-Means et du standardiseur
model = joblib.load(BASE_DIR / "kmeansModel.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")


# --------------------------------------------------
# 2. Définition des profils commerciaux
# --------------------------------------------------

profils = {
    0: {
        "nom": "Clients intermédiaires stables",
        "recommandation": (
            "Proposer des offres régulières et des avantages "
            "de fidélité adaptés à leurs habitudes d'achat."
        )
    },
    1: {
        "nom": "Clients premium à fort pouvoir d'achat",
        "recommandation": (
            "Proposer des produits haut de gamme, des services "
            "personnalisés et des avantages exclusifs."
        )
    },
    2: {
        "nom": "Clients à faible revenu et dépenses élevées",
        "recommandation": (
            "Proposer des promotions ciblées et des offres "
            "adaptées au budget du client."
        )
    },
    3: {
        "nom": "Clients intermédiaires jeunes",
        "recommandation": (
            "Privilégier des offres attractives, des nouveautés "
            "et des campagnes adaptées à leurs préférences."
        )
    },
    4: {
        "nom": "Clients aisés mais peu dépensiers",
        "recommandation": (
            "Mettre en avant la valeur des produits, la qualité "
            "du service et des offres personnalisées."
        )
    },
    5: {
        "nom": "Clients à faible revenu et faible engagement",
        "recommandation": (
            "Proposer des offres accessibles et des incitations "
            "progressives à la fidélisation."
        )
    }
}


def obtenir_profil(cluster_id):
    """
    Retourne le profil commercial et la recommandation
    correspondant au cluster prédit.
    """

    return profils.get(
        cluster_id,
        {
            "nom": "Profil non identifié",
            "recommandation": (
                "Analyser les caractéristiques du client "
                "avant de proposer une action commerciale."
            )
        }
    )


# --------------------------------------------------
# 3. Page d'accueil
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

# Page de consultation de l'historique
@app.route("/historique")
def afficher_historique(): 
    return render_template(
        "historique.html", 
historique=list(reversed(historique_analyses)) )

# --------------------------------------------------
# 4. Prédiction du cluster d'un client
# --------------------------------------------------

@app.route("/prediction", methods=["POST"])
def prediction():

    try:
        # Récupération et validation des données
        age = float(request.form["age"])
        revenu = float(request.form["revenu"])
        depense = float(request.form["depense"])

        # Vérification des valeurs numériques
        if not all(np.isfinite([age, revenu, depense])):
            raise ValueError(
                "Les valeurs doivent être des nombres finis."
            )

        # Vérification des plages des variables du jeu de données
        if not 18 <= age <= 100:
            raise ValueError(
                "L'âge doit être compris entre 18 et 100 ans."
            )

        if not 0 <= revenu <= 200:
            raise ValueError(
                "Le revenu annuel doit être compris entre "
                "0 et 200 milliers de dollars."
            )

        if not 1 <= depense <= 100:
            raise ValueError(
                "Le score de dépenses doit être compris "
                "entre 1 et 100."
            )

        # Construction du vecteur des caractéristiques
        X = np.array([[age, revenu, depense]])

        # Standardisation identique à celle de l'entraînement
        X_transformes = scaler.transform(X)

        # Prédiction du cluster
        cluster = int(model.predict(X_transformes)[0])

        # Récupération du profil commercial
        profil = obtenir_profil(cluster)
        # Enregistrement de l'analyse dans l'historique
        historique_analyses.append({
            "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "age": age,
            "revenu": revenu,
            "depense": depense,
            "cluster": cluster,
            "profile": profil["nom"],
            "recommandation": profil["recommandation"]
        })

        # Conserver uniquement les 10 dernières analyses
        del historique_analyses[:-10]


        # Affichage des résultats
        return render_template(
            "resultat.html",
            cluster=cluster,
            profile=profil["nom"],
            recommandation=profil["recommandation"],
            age=age,
            revenu=revenu,
            depense=depense
        )

    except (ValueError, KeyError) as erreur:
        return render_template(
            "index.html",
            erreur=str(erreur)
        ), 400


# --------------------------------------------------
# 5. Démarrage de l'application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

