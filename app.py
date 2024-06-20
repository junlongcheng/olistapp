import dash
from dash import dcc, html, Input, Output, exceptions
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import dash_bootstrap_components as dbc

# 1. Data Preparation (Assuming you've read your data into a DataFrame named 'df')
df = pd.read_csv('permanently_cleaned_geolocation_data.csv')  

df = df.rename(columns = {'id_left': 'id', 
                          'geolocation_state': 'state', 
                          'geolocation_lat': 'latitude', 
                          'geolocation_lng': 'longitude'})

df['object'] = df['object'].astype(str).str.capitalize()

# 2. Dash App Setup
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# State Filter Options with "Select All"
state_options = [{'label': 'Select All', 'value': 'all'}] + [
    {'label': i, 'value': i} for i in df['state'].unique()
]

app.layout = html.Div([
    html.H1("Olist Warehouse Location Optimization Tool", style={'textAlign': 'center'}),

    # Outer container for slicers
    html.Div(id='outer_div', children=[
        html.Div([  # Slicer 1 (Object)
            html.Label("Select Object:"),
            dcc.Dropdown(id='object-filter', options=[{'label': i, 'value': i} for i in df['object'].unique()], multi=True)
        ], style={'width': '250px', 'margin': '10px'}),  # Adjust width as needed

        html.Div([  # Slicer 2 (State)
            html.Label("Select State:"),
            dcc.Dropdown(id='state-filter', options=state_options, value='all', multi=True)
        ], style={'width': '250px', 'margin': '10px'}),

        html.Div([  # Slicer 3 (Number of Clusters)
            html.Label("Number of Warehouses:"),
            dcc.Input(id='cluster-input', type='number', min=1, value=3)
        ], style={'width': '250px', 'margin': '10px'}),
    ], style={
        'display': 'flex',       
        'flex-direction': 'row',  
        'justify-content': 'center',
        'align-items': 'center',
        'flex-wrap': 'wrap'
    }),

    html.Div(  # Map container
        dcc.Graph(id='cluster-plot', style={'height': '600px', 'width': '800px'}), 
        style={
            'display': 'flex',
            'justify-content': 'center', 
            'align-items': 'center'  
        }
    )
])

# 5. Callback for Interactivity
@app.callback(
    Output('cluster-plot', 'figure'),
    [Input('object-filter', 'value'),
     Input('state-filter', 'value'),
     Input('cluster-input', 'value')]
)
def update_plot(selected_objects, selected_states, num_clusters):
    # Handle "Select All" for States
    if 'all' in selected_states:
        selected_states = df['state'].unique()

    if selected_objects is None or num_clusters is None:
        return px.scatter()  # Empty plot if filters are not selected

    filtered_df = df[df['object'].isin(selected_objects) & df['state'].isin(selected_states)]

    if filtered_df.empty:
        return px.scatter()

    # Clustering
    kmeans = KMeans(n_clusters=num_clusters).fit(filtered_df[['longitude', 'latitude']])
    filtered_df['cluster'] = kmeans.labels_

    # Calculate Cluster Centers
    cluster_centers = filtered_df.groupby('cluster')[['longitude', 'latitude']].mean().reset_index()

        # Create Plot (NO hovertemplate here)
    fig = px.scatter_mapbox(filtered_df, lat="latitude", lon="longitude", color="cluster",
                            hover_name=None,
                            custom_data=['cluster'],
                            mapbox_style="carto-positron")

    # Add Cluster Centers to Plot (WITH hover data)
    fig.add_scattermapbox(
        lat=cluster_centers["latitude"],
        lon=cluster_centers["longitude"],
        mode="markers",
        marker=dict(size=20, color="red"),
        name="Warehouses",
        hoverinfo="text",  # Enable hover for cluster centers
        text=[
            f"Warehouse {i + 1}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}"
            for i, lat, lon in zip(
                cluster_centers.index, cluster_centers["latitude"], cluster_centers["longitude"]
            )
        ],
        showlegend=False,
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port = 8061)