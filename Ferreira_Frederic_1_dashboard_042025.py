import os
import requests
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer
from dash import Dash, html, dcc, callback, Output, Input, State
import base64  # pour encoder les images en base64 pour Dash
import dash_bootstrap_components as dbc

import matplotlib
matplotlib.use('Agg')  # Change le backend de Matplotlib

import os
import dash_auth
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# Liste des utilisateurs
VALID_USERNAME_PASSWORD_PAIRS = {
    'admin': os.environ.get('ADMIN_PASSWORD'),
    'user': os.environ.get('USER_PASSWORD')
}


# Chargement des données
df = pd.read_csv('clients_test_new.csv')

# définir les dataframes pour les graphiques analyse bi-variée
df_no_id = df.drop(columns=['SK_ID_CURR'])
x_features = [
        'CODE_GENDER_M',
        'NAME_INCOME_TYPE_Businessman', 'NAME_INCOME_TYPE_Commercial_associate',
        'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State_servant',
        'NAME_INCOME_TYPE_Student', 'NAME_INCOME_TYPE_Unemployed',
        'NAME_INCOME_TYPE_Working', 'NAME_EDUCATION_TYPE_Academic_degree',
        'NAME_EDUCATION_TYPE_Higher_education',
        'NAME_EDUCATION_TYPE_Incomplete_higher',
        'NAME_EDUCATION_TYPE_Lower_secondary',
        'LOAN_TYPE_Cash_0_or_Revolving_1', 
        'REG_REGION_NOT_WORK_REGION']


y_features = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CREDIT_INCOME_PERCENT',
        'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'AMT_CREDIT', 'AMT_ANNUITY',
        'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED_PERCENT',
        'CNT_CHILDREN', 'OWN_CAR_AGE'
        ]

scatter_plot_vars = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CREDIT_INCOME_PERCENT',
        'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'AMT_CREDIT', 'AMT_ANNUITY',
        'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED_PERCENT',
        'CNT_CHILDREN', 'OWN_CAR_AGE']

# Initialisation de l'application Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])  # Utilise le thème Bootstrap

# Définition de la clé secrète Flask
app.server.secret_key = os.environ.get('SECRET_KEY')  # Remplace par une clé forte et unique

# Ajout de l'authentification au Dash
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

# Clé API et URL de l'API Django
API_URL = "https://projet7-production.up.railway.app/predict/"
API_KEY = "hkih8+0qk7hk2dl*!gg*rrl89@14hh4!*z"

# Charger le modèle sauvegardé 
MODEL_PATH = "./mlartifacts/406813215569809067/f805d569705e46e4984d8f5a44d80118/artifacts/mlflow_model/model.pkl"

try:
    model = joblib.load(MODEL_PATH)  # Charger le modèle entraîné
except FileNotFoundError:
    raise ValueError(f"Fichier du modèle introuvable au chemin : {MODEL_PATH}")
except Exception as e:
    raise ValueError(f"Erreur lors du chargement du modèle : {str(e)}")



# Chemin vers le Lime Explainer sauvegardé
LIME_EXPLAINER_PATH = "./mlartifacts/406813215569809067/f805d569705e46e4984d8f5a44d80118/artifacts/explainers/lime_explainer_params.joblib"

# Charger le Lime Explainer
try:
    params = joblib.load(LIME_EXPLAINER_PATH)
    lime_explainer = LimeTabularExplainer(
        training_data=params['training_data'],
        feature_names=params['feature_names'],
        mode=params['mode']
    )
except FileNotFoundError:
    raise ValueError(f"Fichier introuvable au chemin : {LIME_EXPLAINER_PATH}")
except Exception as e:
    raise ValueError(f"Erreur lors du chargement du Lime Explainer : {str(e)}")

app.title = "Dashboard - Prêt à dépenser"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div(
                html.H1("Dashboard Interactif - Prêt à dépenser"),
                **{"aria-label": "Titre principal du tableau de bord"}
            ), 
            width=12, 
            style={'textAlign': 'center'}
        )
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col(
            # Slider pour la taille du texte
            html.Div([
                html.Label("Modifier la taille du texte :", 
                        **{"aria-label": "Slider pour changer la taille du texte du dashboard"}),
                dcc.Slider(
                    id='text-size-slider',
                    min=10,
                    max=30,
                    step=2,
                    value=16,
                    marks={10: '10px', 20: '20px', 30: '30px'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'width': '100%', 'marginBottom': '20px', 'padding': '10px'}), 
            width=12, style={'textAlign': 'center'}
        )
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Div(
                html.H3("Informations du client"),
                **{"aria-label": "Informations client"}
            ),
            html.Div(
                html.P("Sélectionnez un ID client"),
                **{"aria-label": "Instruction pour sélectionner un ID client"}
            ),
            html.Div(
                dcc.Dropdown(
                    id='client-id-dropdown',
                    options=[{'label': client_id, 'value': client_id} for client_id in df['SK_ID_CURR']],
                    value=df['SK_ID_CURR'][0]
                ),
                **{"aria-label": "Menu déroulant pour sélectionner l'ID d'un client"}
            ),
            html.Br(),
            html.Div(id='client-info', **{"aria-label": "Informations détaillées du client sélectionné"})
        ], width=3, xs=12, sm=12, md=12, lg=3, 
           className="dash-col", 
           style={'margin': 'auto', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'stretch'}
        ),
        
        # new col
        dbc.Col([
            html.Div(
                html.H3("Probabilité de défaut et statut du prêt"),
                **{"aria-label": "Section des prédictions renvoyées par l'API"}
            ),
            html.Div(
                html.Button(
                    'Prédire', 
                    id='predict-button', 
                    className='btn btn-outline-dark w-100'
                ),
                **{"aria-label": "Bouton pour obtenir les prédictions via l'API"}
            ),
            html.Div(id='api-output', className='mt-4', 
                **{"aria-label": "Résultats des prédictions de l'API : ; probabilité de défaut du client et statut du prêt (Accepté ou refusé"}),
            
        ], width=3, xs=12, sm=12, md=12, lg=3, 
           className="dash-col", 
           style={'margin-top': 0, 'margin-left': 'auto', 'margin-right': 'auto', 'margin-bottom': 'auto',
                'display': 'flex', 'flexDirection': 'column', 'alignItems': 'stretch'}
        ),
        dbc.Col([
            html.Div(
                html.H3("Explication locale (LIME)"),
                **{"aria-label": "Graphique d'explication locale basé sur LIME (feature importance locale)"}
            ),
            html.Div(
                dcc.Loading(
                    id="loading-lime",
                    children=[
                        html.Div([
                            html.Img(
                                id='lime-image', 
                                style={'maxWidth': '100%', 'height': 'auto'}
                            )
                        ], style={'textAlign': 'center'})
                    ],
                    type="circle"
                ),
                **{"aria-label": "Chargement en cours pour le graphique LIME (feature importance locale)"}
            )
        ], width=6, xs=12, sm=12, md=12, lg=6, className="dash-col", 
                style={'margin-top': 0, 'margin-left': 'auto', 'margin-right': 'auto', 
                    'margin-bottom': 'auto', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'stretch'}),
    ], justify="around", align="start"),
    html.Hr(),
    # new row
    dbc.Row([
        dbc.Col([
            html.Div(
                html.H3("Modifier certaines variables pour rafraîchir les prédictions"),
                **{"aria-label": "Section pour modifier les variables d'un dossier client existant"}
            ),
            html.Div([
                html.Div(
                    html.Label("Âge (en jours) :"),
                    **{"aria-label": "Champ pour modifier l'âge du client"}
                ),
                html.Div(
                    dcc.Input(id='modified-age', type='number', 
                    placeholder="Entrez l'âge en jours", style={'width': '100%'}),
                    **{"aria-label": "Entrée pour l'âge modifié"}
                ),
                html.Div(
                    html.Label("Nombre de jours en emploi :"),
                    **{"aria-label": "Champ pour modifier les jours en emploi du client"}
                ),
                html.Div(
                    dcc.Input(id='modified-days-employed', type='number', 
                    placeholder="Nombre de jours travaillés", style={'width': '100%'}),
                    **{"aria-label": "Entrée pour les jours en emploi modifiés"}
                ),
                html.Div(
                    html.Label("Montant du prêt (€) :"),
                    **{"aria-label": "Champ pour modifier le montant du prêt"}
                ),
                html.Div(
                    dcc.Input(id='modified-loan-amount', type='number', 
                    placeholder="Montant du prêt en €", style={'width': '100%'}),
                    **{"aria-label": "Entrée pour le montant du prêt modifié"}
                ),
                html.Div(
                    html.Label("Revenus annuels (€) :"),
                    **{"aria-label": "Champ pour modifier les revenus annuels du client"}
                ),
                html.Div(
                    dcc.Input(id='modified-income', type='number', 
                    placeholder="Revenus annuels en €", style={'width': '100%'}),
                    **{"aria-label": "Entrée pour les revenus modifiés"}
                ),
                html.Div(
                    html.Label("Nombre d'enfants :"),
                    **{"aria-label": "Champ pour modifier le nombre d'enfants"}
                ),
                html.Div(
                    dcc.Input(id='modified-children', type='number', 
                    placeholder="Nombre d'enfants", style={'width': '100%'}),
                    **{"aria-label": "Entrée pour le nombre d'enfants modifié"}
                ),
            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}),
            html.Br(),
            html.Button('Rafraîchir la prédiction', id='refresh-prediction-button', 
                        className='btn btn-outline-dark w-100', n_clicks=0),
        ], width=3, xs=12, sm=12, md=12, lg=3, className="dash-col"),
        # new col
        dbc.Col([
            html.Div(
                html.H3("Résultats après modification"),
                **{"aria-label": "Résultats mis à jour après modification des variables"}
            ),
            html.Div(id='modified-api-output', **{"aria-label": "Résultats de prédiction mis à jour"})
        ], width=3, xs=12, sm=12, md=12, lg=3, className="dash-col"),
        # new col
        dbc.Col([
            html.Div(
                html.H3("Explication locale (LIME) après modifications"),
                **{"aria-label": "Graphique de la feature importance locale mis à jour après modification des variables"}
            ),
            html.Div(id='modified-lime-image', style={'textAlign': 'center'})
        ], width=6, xs=12, sm=12, md=12, lg=6, className="dash-col")
    ]),
    html.Hr(),
    # new row
    dbc.Row([
        dbc.Col([
            html.Div(
                html.H3("Critères de similarité"),
                **{"aria-label": "Section des critères de similarité"}
            ),
            html.Div([
                html.Div(
                    html.Label("Écart d'âge (tolérance en jours) :"),
                    **{"aria-label": "Label pour tolérance d'âge"}
                ),
                html.Div(
                    dcc.Input(
                        id='age-tolerance',
                        type='number',
                        value=3650,
                        step=50,
                        style={'width': '100%'}
                    ),
                    **{"aria-label": "Entrée pour définir la tolérance d'âge en jours"}
                ),
                html.Div(
                    html.Label("Écart de revenus (tolérance en €) :"),
                    **{"aria-label": "Label pour tolérance de revenus"}
                ),
                html.Div(
                    dcc.Input(
                        id='income-tolerance',
                        type='number',
                        value=20000,
                        step=1000,
                        style={'width': '100%'}
                    ),
                    **{"aria-label": "Entrée pour définir la tolérance de revenus en euros"}
                ),
                html.Div(
                    html.Label("Sexe :"),
                    **{"aria-label": "Label pour le critère de genre"}
                ),
                html.Div(
                    dcc.Dropdown(
                        id='gender-filter',
                        options=[
                            {'label': 'Même sexe', 'value': 'same'},
                            {'label': 'Les deux sexes', 'value': 'any'}
                        ],
                        value='same',
                        style={'width': '100%'}
                    ),
                    **{"aria-label": "Menu déroulant pour sélectionner le filtre de genre, choix entre même sexe et les deux sexes"}
                )
            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'})
        ], width=3, xs=12, sm=12, md=12, lg=3, className="dash-col"),
        dbc.Col([
            html.Div(
                html.H3("Comparaison avec d'autres clients"),
                **{"aria-label": "Graphique de comparaison"}
            ),
            html.Div(
                html.H6(
            "Cette section permet de comparer les caractéristiques du client sélectionné avec celles de tous les clients ou un groupe similaire."
                ),
                **{"aria-label": "Description de la section de comparaison des clients"}
            ),
            html.Div(
                html.P("Sélectionnez une variable"),
                **{"aria-label": "Instruction pour sélectionner une variable"}
            ),
            html.Div(
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in df_no_id.columns if df_no_id[col].dtype in ['float64', 'int64', 'bool']],
                    value=df.columns[1]
                ),
                **{"aria-label": "Menu déroulant pour sélectionner la variable à comparer"}
            ),
            html.Br(),
            html.Div(
                html.P("Sélectionnez une option de comparaison"),
                **{"aria-label": "Instruction pour sélectionner une option de comparaison, choix entre tous les clients et clients similaires"}
            ),
            html.Div(
                dcc.Dropdown(
                    id='group-filter',
                    options=[
                        {'label': 'Tous les clients', 'value': 'all'},
                        {'label': 'Clients similaires', 'value': 'similar'}
                    ],
                    value='all'
                ),
                **{"aria-label": "Menu déroulant pour sélectionner le groupe de comparaison"}
            ),
            html.Br(),
            html.Div(
                dcc.Graph(
                    id='comparison-graph',
                    config={
                        'responsive': True,
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['lasso2d', 'zoomIn2d', 'zoomOut2d'],
                        'scrollZoom': False
                    }
                ),
                **{"aria-label": "Graphique de comparaison des caractéristiques du client"}
            )
        ], width=9, xs=12, sm=12, md=12, lg=9, className="dash-col"),
    ], justify="around", align="start"),
    html.Hr(),
    # new row
    dbc.Row([
        dbc.Col([
            # Graphique d’analyse bi-variée nuage de points
            html.Div([
                html.H3("Analyse bi-variée - Nuage de points"),
                html.Div([
                    html.Label("Sélectionner la variable quantitative en abscisse :", 
                    **{"aria-label": "Dropdown pour sélectionner la variable quantitative sur l'axe des abscisses du nuage de points"}),
                    dcc.Dropdown(
                        id="scatterplot-feature-1",
                        options=[{"label": col, "value": col} for col in df_no_id[scatter_plot_vars].columns],
                        value=df_no_id.columns[1]
                    )
                ]),
                html.Div([
                    html.Label("Sélectionner la variable quantitative en ordonnée :", 
                    **{"aria-label": "Dropdown pour sélectionner la seconde variable quantitative sur l'axe des ordonnées du nuage de points"}),
                    dcc.Dropdown(
                        id="scatterplot-feature-2",
                        options=[{"label": col, "value": col} for col in df_no_id[scatter_plot_vars].columns],
                        value=df_no_id.columns[2]
                    )
                ]),
                html.Div(
                    dcc.Graph(
                        id="scatterplot-graph",
                        config={
                            'responsive': True,
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'zoomIn2d', 'zoomOut2d'],
                            'scrollZoom': False
                        }
                    ),
                    **{"aria-label": "Graphique d'analyse bi-variée en nuage de points comparant deux variables"}
                )
            ]),
        ], width=6, xs=12, sm=12, md=12, lg=6, className="dash-col"),  # end of col
        
        dbc.Col([
            # Graphique d’analyse bi-variée, boxplot
            html.Div([
                html.H3("Analyse bi-variée - Boxplot"),
                html.Div([
                    html.Label("Sélectionner la variable en abscisse:", 
                    **{"aria-label": "Dropdown pour sélectionner la variable catégorielle sur l'axe des abscisses du boxplot"}),
                    dcc.Dropdown(
                        id="boxplot-feature-1",
                        options=[{"label": col, "value": col} for col in df[x_features].columns],
                        value=df_no_id.columns[1]
                    )
                ]),
                html.Div([
                    html.Label("Sélectionner la deuxième variable :", 
                    **{"aria-label": "Dropdown pour sélectionner la variable numérique sur l'axe des ordonnées du boxplot"}),
                    dcc.Dropdown(
                        id="boxplot-feature-2",
                        options=[{"label": col, "value": col} for col in df[y_features].columns],
                        value=df_no_id.columns[2]
                    )
                ]),
                html.Div(
                    dcc.Graph(
                        id="boxplot-graph",
                        config={
                            'responsive': True,
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['lasso2d', 'zoomIn2d', 'zoomOut2d'],
                            'scrollZoom': False
                        }
                    ),
                    **{"aria-label": "Graphique d'analyse bi-variée sous forme de boxplot comparant deux variables"}
                )
            ]),  # end of parent div
        ], width=6, xs=12, sm=12, md=12, lg=6, className="dash-col")  # end of col
    ], justify="around", align="start"),  # end of row

], id="main-content") # end of layout



# Fonction pour appeler l'API
def call_api(client_data):
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": API_KEY
    }
    response = requests.post(API_URL, json=client_data, headers=headers)
    if response.status_code == 200:
        return response.json()  # Renvoie les résultats sous forme de dictionnaire
    else:
        return {"error": f"Erreur API : {response.status_code}"}

# Callback pour afficher les informations descriptives du client
@callback(
    Output('client-info', 'children'),
    Input('client-id-dropdown', 'value')
)
def display_client_info(client_id):
    client_data = df[df['SK_ID_CURR'] == client_id].iloc[0]
    
    # Extraire les variables principales
    age_in_days = client_data['DAYS_BIRTH']  # Conversion des jours en années
    gender = "Homme" if client_data['CODE_GENDER_M'] else "Femme"
    days_employed = client_data['DAYS_EMPLOYED']
    credit_term = client_data['CREDIT_TERM']
    amt_credit = client_data['AMT_CREDIT']
    amt_annuity = client_data['AMT_ANNUITY']
    income_total = client_data['AMT_INCOME_TOTAL']
    cnt_children = client_data['CNT_CHILDREN']
    
    # Retourner les informations formatées avec des classes CSS
    return html.Div([
        html.P([
            html.Span("Âge du client en jours au moment du prêt : ", className="text-blue"),
            html.Span(f"{age_in_days}", className="text-black")
        ]),
        html.P([
            html.Span("Sexe : ", className="text-blue"),
            html.Span(f"{gender}", className="text-black")
        ]),
        html.P([
            html.Span("Nombre de jours en emploi : ", className="text-blue"),
            html.Span(f"{days_employed}", className="text-black")
        ]),
        html.P([
            html.Span("Durée du prêt : ", className="text-blue"),
            html.Span(f"{credit_term}", className="text-black")
        ]),
        html.P([
            html.Span("Montant du prêt en € : ", className="text-blue"),
            html.Span(f"{amt_credit}", className="text-black")
        ]),
        html.P([
            html.Span("Montant d'une mensualité en €: ", className="text-blue"),
            html.Span(f"{amt_annuity}", className="text-black")
        ]),
        html.P([
            html.Span("Revenus annuels en € : ", className="text-blue"),
            html.Span(f"{income_total}", className="text-black")
        ]),
        html.P([
            html.Span("Nombre d'enfants : ", className="text-blue"),
            html.Span(f"{cnt_children}", className="text-black")
        ])
])

# Callback pour obtenir les prédictions ET générer le graphique LIME
@callback(
    [Output('api-output', 'children'),  # Résultats des prédictions
     Output('lime-image', 'src')],     # Graphique LIME
    Input('predict-button', 'n_clicks'),  # Clic sur le bouton
    State('client-id-dropdown', 'value')  # État du client sélectionné
)
def get_predictions_and_lime_graph(n_clicks, client_id):
    if n_clicks:
        # Récupérer les données du client
        client_data = df[df['SK_ID_CURR'] == client_id].iloc[0].to_dict()
        
        # Appeler l'API pour obtenir les prédictions
        prediction = call_api(client_data)
        if "error" in prediction:
            return html.Div(f"Erreur API : {prediction['error']}"), None
        
        # Extraire les champs de la réponse
        probability = prediction.get("probability")
        status = prediction.get("status")
        threshold = prediction.get("threshold")
        
        # Déterminer la couleur du texte selon le statut
        color = "green" if status == "Accepté" else "red"
        
        # Résultat des prédictions
        api_output = html.Div([
            html.P(f"Probabilité : {probability}"),
            html.P(f"Statut : {status}", style={'color': color}),
            html.P(f"Seuil : {threshold}")
        ])
        
        # Générer une explication LIME
        client_data_df = df[df['SK_ID_CURR'] == client_id]  # Données du client sous forme de DataFrame
        X_features_array = client_data_df.drop(columns=['SK_ID_CURR'], errors='ignore').to_numpy()
        X_features_array_df = pd.DataFrame(X_features_array, columns=df.drop(columns=['SK_ID_CURR']).columns)
        lime_exp = lime_explainer.explain_instance(
            X_features_array_df.iloc[0],  # Données du client sous forme de DataFrame
            model.predict_proba,  # Méthode de prédiction du modèle
            num_features=15
        )

        # Sauvegarder le graphique LIME
        lime_graph_path = f'lime_graph_{client_id}.png'
        lime_exp.as_pyplot_figure()
        plt.title(f"Variables les plus importantes pour le client {client_id}")
        plt.savefig(lime_graph_path, bbox_inches='tight')
        plt.close()

        # Encoder l'image en base64 pour Dash
        with open(lime_graph_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        lime_image_src = f"data:image/png;base64,{encoded_image}"

        return api_output, lime_image_src
    
    # Si le bouton n'a pas encore été cliqué, ne rien afficher
    return "", None


@callback(
    Output('comparison-graph', 'figure'),
    [Input('feature-dropdown', 'value'),
     Input('group-filter', 'value'),
     Input('client-id-dropdown', 'value'),
     Input('age-tolerance', 'value'),
     Input('income-tolerance', 'value'),
     Input('gender-filter', 'value')]
)

def update_comparison_graph(selected_feature, group_filter, client_id, age_tolerance, income_tolerance, gender_filter):
    client_data = df[df['SK_ID_CURR'] == client_id].iloc[0]
    
    if group_filter == 'similar':
        similar_clients = df[
            (abs(df['DAYS_BIRTH'] - client_data['DAYS_BIRTH']) <= age_tolerance) &
            (abs(df['AMT_INCOME_TOTAL'] - client_data['AMT_INCOME_TOTAL']) <= income_tolerance) &
            (((gender_filter == 'same') & (df['CODE_GENDER_M'] == client_data['CODE_GENDER_M'])) | (gender_filter == 'any'))
        ]
    else:
        similar_clients = df

    client_value = df[df['SK_ID_CURR'] == client_id][selected_feature].values[0]

    # Conversion pour les binaires
    if df[selected_feature].dtype == 'bool':
        similar_clients[selected_feature] = similar_clients[selected_feature].map({True: 1, False: 0})
        client_value = 1 if client_value else 0

    fig = px.histogram(
        similar_clients,
        x=selected_feature,
        title=f"Comparaison de la caractéristique '{selected_feature}'",
        labels={'x': selected_feature},
        color_discrete_sequence=['blue']
    )

    # Options spécifiques pour les binaires
    if df[selected_feature].dtype == 'bool':
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = [0, 1],
                ticktext = ['False', 'True']
            ),
            autosize=True,
            width=None,
            height=300
        )
    else :
        # Mise en page globale
        fig.update_layout(
            autosize=True,
            width=None,
            height=300,
            margin={'l': 20, 'r': 20, 't': 30, 'b': 30}
        )
    
     # Ajouter la ligne verticale pour représenter la valeur du client
    fig.add_vline(
        x=client_value,
        line_dash="dot",
        line_color="red",
        annotation_text="Client sélectionné"
    )
    return fig


@callback(
    [Output('modified-api-output', 'children'),  # Résultats des prédictions mis à jour
     Output('modified-lime-image', 'children')],  # Graphique LIME mis à jour
    [Input('refresh-prediction-button', 'n_clicks')],  # Bouton pour rafraîchir
    [State('client-id-dropdown', 'value'),          # ID du client sélectionné
     State('modified-age', 'value'),               # Âge modifié
     State('modified-days-employed', 'value'),     # Jours en emploi modifiés
     State('modified-loan-amount', 'value'),       # Montant du prêt modifié
     State('modified-income', 'value'),            # Revenus modifiés
     State('modified-children', 'value')]          # Nombre d'enfants modifié
)

def update_client_data(n_clicks, client_id, age, days_employed, loan_amount, income, children):
    if n_clicks > 0:  # Exécuter uniquement après un clic sur le bouton
        # Récupérer les données du client sélectionné
        client_data = df[df['SK_ID_CURR'] == client_id].iloc[0].to_dict()
        
        # Appliquer les modifications (si une valeur est saisie, elle remplace l'existante)
        if age is not None:
            client_data['DAYS_BIRTH'] = age
        if days_employed is not None:
            client_data['DAYS_EMPLOYED'] = days_employed
        if loan_amount is not None:
            client_data['AMT_CREDIT'] = loan_amount
        if income is not None:
            client_data['AMT_INCOME_TOTAL'] = income
        if children is not None:
            client_data['CNT_CHILDREN'] = children

        # Appeler l'API avec les données modifiées
        prediction = call_api(client_data)
        if "error" in prediction:
            return html.Div(f"Erreur API : {prediction['error']}"), None
        
        # Extraire les champs de la réponse
        probability = prediction.get("probability", "N/A")
        status = prediction.get("status", "N/A")
        threshold = prediction.get("threshold", "N/A")
        color = "green" if status == "Accepté" else "red"
        
        # Résultats des prédictions
        api_output = html.Div([
            html.P(f"Probabilité : {probability}"),
            html.P(f"Statut : {status}", style={'color': color}),
            html.P(f"Seuil : {threshold}")
        ])
        
        # Générer une explication LIME
        client_data_df = pd.DataFrame([client_data])  # Données du client sous forme de DataFrame
        X_features_array = client_data_df.drop(columns=['SK_ID_CURR'], errors='ignore').to_numpy()
        X_features_array_df = pd.DataFrame(X_features_array, columns=df.drop(columns=['SK_ID_CURR']).columns)
        lime_exp = lime_explainer.explain_instance(
            X_features_array_df.iloc[0],  # Données du client sous forme de DataFrame
            model.predict_proba,  # Méthode de prédiction du modèle
            num_features=15
        )
    
        # Sauvegarder le graphique LIME
        lime_graph_path = f'lime_graph_{client_id}.png'
        lime_exp.as_pyplot_figure()
        plt.title(f"Variables importantes pour le client {client_id}")
        plt.savefig(lime_graph_path, bbox_inches='tight')
        plt.close()

        # Encoder l'image en Base64
        try:
            with open(lime_graph_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            lime_image_src = f"data:image/png;base64,{encoded_image}"
        except FileNotFoundError:
            return html.Div("Erreur : l'image LIME n'a pas pu être trouvée."), None

        # Retourner le contenu
        lime_image = html.Img(src=lime_image_src, style={'maxWidth': '100%', 'height': 'auto'})
        
        return api_output, lime_image
    
    return "", None


@callback(
    Output("scatterplot-graph", "figure"),
    [Input("scatterplot-feature-1", "value"),
     Input("scatterplot-feature-2", "value")]
)
def update_scatterplot_graph(feature1, feature2):
    # Récupérer uniquement les colonnes numériques
    data = df_no_id[scatter_plot_vars]
    fig = px.scatter(
        data, 
        x=feature1, 
        y=feature2, 
        title=f"Analyse bi-variée : {feature1} vs {feature2}",
        opacity=0.7
    )
    fig.update_traces(
        marker=dict(color='#1900ff')  # Changer la couleur des points
    )
    fig.update_layout(template="plotly_white")
    return fig



@callback(
    Output("boxplot-graph", "figure"),
    [Input("boxplot-feature-1", "value"),
     Input("boxplot-feature-2", "value")]
)
def update_boxplot_graph(feature1, feature2):
    # Récupérer uniquement les colonnes numériques
    fig = px.box(
        df_no_id, 
        x=feature1, 
        y=feature2, 
        color=feature1,
        title=f"Analyse bi-variée : {feature1} vs {feature2}",
        
    )
    fig.update_layout(template="plotly_white")
    return fig



@callback(
    Output('main-content', 'style'),  # Applique le style à tout le conteneur
    Input('text-size-slider', 'value')  # Taille choisie par l'utilisateur
)
def update_text_size(font_size):
    return {
        'font-size': f'{font_size}px',  # Ajuste la taille du texte
        'line-height': '1.5',           # Espacement entre les lignes
        'padding': '10px'               # Espacement général
    }


# Lancement de l'application
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
    


# # Lancement de l'application
# if __name__ == '__main__':
#     app.run(debug=True)
