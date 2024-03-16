import dash
from dash import html, dcc, Input, Output
import pickle
import numpy as np

with open('C:\\Users\\m.naja\\creditmodel.sav', 'rb') as f:
    pickeled_model = pickle.load(f)

    
app = dash.Dash(__name__)

input_style = {'width': '500px'}  
# Définir la mise en page de l'application
app.layout = html.Div([
    html.H1("Prédiction Crédit"),
    # Ajouter des entrées pour les caractéristiques du modèle
    dcc.Input(id='EXT_SOURCE_2', type='number', placeholder='Normalized score from external data source',style=input_style),
    dcc.Input(id='EXT_SOURCE_3', type='number', placeholder='Normalized score from external data source',style=input_style),
    dcc.Input(id='CNT_FAM_MEMBERS', type='number', placeholder='How many family members does client have',style=input_style),
    dcc.Input(id='DAYS_REGISTRATION', type='number', placeholder='How many days before the application did client change his registration',style=input_style),
    dcc.Input(id='AMT_REQ_CREDIT_BUREAU_HOUR', type='number', placeholder='Number of enquiries to Credit Bureau about the client one hour before application',style=input_style),

    # Vous pouvez ajouter plus d'entrées selon les caractéristiques de votre modèle
    html.Button('Prédire', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output')
])

# Définir la callback pour la prédiction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('EXT_SOURCE_2', 'value'), Input('EXT_SOURCE_3', 'value'), Input('CNT_FAM_MEMBERS', 'value'), Input('DAYS_REGISTRATION', 'value'),
    Input('AMT_REQ_CREDIT_BUREAU_HOUR', 'value')]  # Ajouter plus d'inputs si nécessaire
)
def update_output(n_clicks, EXT_SOURCE_2, EXT_SOURCE_3, CNT_FAM_MEMBERS, DAYS_REGISTRATION, AMT_REQ_CREDIT_BUREAU_HOUR):
    if n_clicks > 0:
        # Préparer les données pour la prédiction
        features = np.array([[EXT_SOURCE_2, EXT_SOURCE_3, CNT_FAM_MEMBERS, DAYS_REGISTRATION, AMT_REQ_CREDIT_BUREAU_HOUR]])  # Assurez-vous que ceci correspond à la forme attendue par votre modèle
        prediction = pickeled_model.predict(features)
        return f'Prédiction: {prediction[0]}'
    return ""

# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)