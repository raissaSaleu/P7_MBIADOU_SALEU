# Projet7-(Openclassrooms/CentraleSupelec)
Parcours Data Science

Projet n°7 : "Implémentez un modèle de scoring"

## Description du projet

Une société financière "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt. Elle souhaite développer un **modèle de scoring de la probabilité de défaut de paiement du client** pour étayer la décision d'accorder ou non un prêt à un client potentiel en s’appuyant sur des sources de données variées.

Source des données : https://www.kaggle.com/c/home-credit-default-risk/data

## Mission

* Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
* Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle et d’améliorer la connaissance client des chargés de relation client.

## Compétences évaluées

* Utiliser un logiciel de version de code pour assurer l’intégration du modèle
* Déployer un modèle via une API dans le Web
* Réaliser un dashboard pour présenter son travail de modélisation
* Rédiger une note méthodologique afin de communiquer sa démarche de modélisation
* Présenter son travail de modélisation à l'oral

## Livrables

* Le [dashboard interactif](https://p7-dashboard-mbiadou.herokuapp.com/) répondant aux spécifications ci-dessus et l’API de prédiction du score, déployées chacunes sur le cloud.
* Un dossier sur un outil de versioning de code contenant :
    - Le [code de la modélisation](https://github.com/raissaSaleu/P7_MBIADOU_SALEU/blob/main/P7_02_dossier/Notebook.ipynb) (du prétraitement à la prédiction)
    - Le [code générant le dashboard](https://github.com/raissaSaleu/P7_MBIADOU_SALEU/blob/main/P7_02_dossier/frontend/dashboard.py)
    - Le [code permettant de déployer le modèle sous forme d'API](https://github.com/raissaSaleu/P7_MBIADOU_SALEU/blob/main/P7_02_dossier/backend/app.py)
* Une [note méthodologique](https://github.com/raissaSaleu/P7_MBIADOU_SALEU/blob/main/P7_03_Note_M%C3%A9thodologique.pdf) décrivant :
    - La méthodologie d'entraînement du modèle (2 pages maximum)
    - La fonction coût métier, l'algorithme d'optimisation et la métrique d'évaluation (1 page maximum)
    - L’interprétabilité globale et locale du modèle (1 page maximum)
    - Les limites et les améliorations possibles (1 page maximum)
* Un [support de présentation](https://github.com/raissaSaleu/P7_MBIADOU_SALEU/blob/main/P7_04_Presentation.pdf) pour la soutenance, détaillant le travail réalisé.
