import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Define the question labels for better readability
QUESTION_LABELS = {
    'Q1': 'Intellectual Impairment',
    'Q2': 'Mood',
    'Q3': 'Rest Tremor',
    'Q4': 'Finger Taps',
    'Q5': 'Gait/Walking',
    'Q6': 'Off Time',
    'Q7': 'Dyskinesia Duration',
    'Q8': 'Dyskinesia Disability'
}

# Define colors for better visualization
COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1']
SEVERITY_COLORS = ['#4caf50', '#8bc34a', '#ffeb3b', '#ff9800', '#f44336']

def extract_score(response):
    """Extract numerical score from text response"""
    if pd.isna(response):
        return None
    match = re.match(r'^(\d+):', str(response))
    return int(match.group(1)) if match else None

def load_and_process_data(file_path):
    """Load and process the UPDRS 8Q Excel file"""
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Check if the required columns exist
    required_columns = ['Timestamp']
    for i in range(1, 9):
        # Try different possible column naming patterns
        patterns = [
            f"{i}. ",  # e.g., "1. INTELLECTUAL IMPAIRMENT"
            f"INTELLECTUAL" if i == 1 else "",
            f"MOOD" if i == 2 else "",
            f"REST TREMOR" if i == 3 else "",
            f"FINGER TAPS" if i == 4 else "",
            f"GAIT" if i == 5 else "",
            f"OFF TIME" if i == 6 else "",
            f"DYSKINESIA DURATION" if i == 7 else "",
            f"DYSKINESIA DISABLITIY" if i == 8 else ""
        ]
        
        # Find the correct column for each question
        for pattern in patterns:
            if pattern:
                matching_cols = [col for col in df.columns if pattern in col]
                if matching_cols:
                    required_columns.append(matching_cols[0])
                    break
    
    # Extract scores from text responses
    processed_df = pd.DataFrame()
    processed_df['timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Extract numeric scores for each question
    for i, col in enumerate(required_columns[1:], 1):
        processed_df[f'Q{i}'] = df[col].apply(extract_score)
    
    # Add month-year column for time series analysis
    processed_df['month_year'] = processed_df['timestamp'].dt.strftime('%Y-%m')
    
    return processed_df

def calculate_average_scores(df):
    """Calculate average scores for each UPDRS question"""
    avg_scores = []
    
    for i in range(1, 9):
        col = f'Q{i}'
        avg = df[col].mean()
        avg_scores.append({
            'question': col,
            'label': QUESTION_LABELS[col],
            'averageScore': avg,
            'fullMark': 4
        })
    
    return pd.DataFrame(avg_scores)

def prepare_time_series_data(df):
    """Prepare time series data for trend visualization"""
    time_series = df.groupby('month_year').apply(
        lambda x: pd.Series({
            'count': len(x),
            **{f'Q{i}': x[f'Q{i}'].mean() for i in range(1, 9)}
        })
    ).reset_index()
    
    # Sort by date
    time_series['date'] = pd.to_datetime(time_series['month_year'] + '-01')
    time_series = time_series.sort_values('date')
    
    return time_series

def prepare_distribution_data(df):
    """Prepare distribution data for score frequency visualization"""
    distribution_data = []
    
    for q in range(1, 9):
        col = f'Q{q}'
        for score in range(5):  # 0-4 scores
            count = (df[col] == score).sum()
            distribution_data.append({
                'question': col,
                'label': QUESTION_LABELS[col],
                'score': score,
                'count': count
            })
    
    return pd.DataFrame(distribution_data)

def create_radar_chart(avg_scores_df):
    """Create a radar chart for UPDRS scores"""
    fig = go.Figure()
    
    # Add radar chart
    fig.add_trace(go.Scatterpolar(
        r=avg_scores_df['averageScore'],
        theta=avg_scores_df['question'],
        fill='toself',
        name='Average Score',
        line_color='#8884d8',
        fillcolor='rgba(136, 132, 216, 0.6)'
    ))
    
    # Configure layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 4]
            )
        ),
        title='UPDRS Symptom Radar Chart',
        showlegend=True
    )
    
    return fig

def create_bar_chart(avg_scores_df):
    """Create a bar chart for average UPDRS scores"""
    fig = px.bar(
        avg_scores_df,
        x='question',
        y='averageScore',
        color='question',
        labels={'averageScore': 'Average Score (0-4)', 'question': 'UPDRS Question'},
        title='Average UPDRS Scores',
        color_discrete_sequence=COLORS
    )
    
    # Add hover text with full labels
    fig.update_traces(
        hovertemplate='<b>%{x}</b>: %{y:.2f}/4<br>%{customdata}',
        customdata=avg_scores_df['label']
    )
    
    # Configure layout
    fig.update_layout(
        xaxis_title='Question',
        yaxis_title='Average Score',
        yaxis=dict(range=[0, 4]),
        xaxis={'categoryorder': 'array', 'categoryarray': [f'Q{i}' for i in range(1, 9)]}
    )
    
    return fig

def create_time_series_chart(time_series_df):
    """Create a line chart for UPDRS score trends over time"""
    fig = go.Figure()
    
    # Add a line for each question
    for i, q in enumerate(range(1, 9)):
        col = f'Q{q}'
        fig.add_trace(go.Scatter(
            x=time_series_df['month_year'],
            y=time_series_df[col],
            mode='lines+markers',
            name=f'{col}: {QUESTION_LABELS[col]}',
            line=dict(color=COLORS[i % len(COLORS)]),
            connectgaps=True
        ))
    
    # Configure layout
    fig.update_layout(
        title='UPDRS Score Trends Over Time',
        xaxis_title='Month',
        yaxis_title='Average Score',
        yaxis=dict(range=[0, 4]),
        legend=dict(orientation='h', yanchor='bottom', y=-0.5, xanchor='center', x=0.5)
    )
    
    return fig

def create_response_count_chart(time_series_df):
    """Create a bar chart for response counts by month"""
    fig = px.bar(
        time_series_df,
        x='month_year',
        y='count',
        labels={'count': 'Number of Responses', 'month_year': 'Month'},
        title='Response Count by Month',
        color_discrete_sequence=['#82ca9d']
    )
    
    # Configure layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Number of Responses'
    )
    
    return fig

def create_distribution_charts(distribution_df):
    """Create distribution charts for each UPDRS question"""
    figs = []
    
    for q in range(1, 9):
        q_data = distribution_df[distribution_df['question'] == f'Q{q}']
        
        fig = px.bar(
            q_data,
            x='score',
            y='count',
            title=f'Q{q}: {QUESTION_LABELS[f"Q{q}"]} - Score Distribution',
            labels={'count': 'Count', 'score': 'Score (0-4)'},
            color='score',
            color_discrete_sequence=[SEVERITY_COLORS[i] for i in range(5)]
        )
        
        # Configure layout
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1, 2, 3, 4],
                ticktext=['0: None', '1: Slight', '2: Mild', '3: Moderate', '4: Severe']
            ),
            showlegend=False
        )
        
        figs.append(fig)
    
    return figs

def init_dashboard(df, avg_scores_df, time_series_df, distribution_df):
    """Initialize the Dash app for the UPDRS dashboard"""
    app = dash.Dash(__name__, 
                    external_stylesheets=[
                        'https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap',
                        'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
                    ])
    
    # Create initial charts
    radar_chart = create_radar_chart(avg_scores_df)
    bar_chart = create_bar_chart(avg_scores_df)
    time_series_chart = create_time_series_chart(time_series_df)
    response_count_chart = create_response_count_chart(time_series_df)
    distribution_charts = create_distribution_charts(distribution_df)
    
    # App layout
    app.layout = html.Div([
        html.Div([
            html.H1("UPDRS 8Q Assessment Dashboard", className="mt-4 mb-4"),
            html.P(f"Data spans from {df['timestamp'].min().strftime('%B %Y')} to {df['timestamp'].max().strftime('%B %Y')} with {len(df)} total responses", className="text-muted"),
        ], className="container"),
        
        html.Div([
            # Tabs
            dcc.Tabs(id='tabs', value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Time Trends', value='timetrends'),
                dcc.Tab(label='Score Distribution', value='distribution'),
            ], className="mb-4"),
            
            # Tab content
            html.Div(id='tab-content')
        ], className="container"),
    ], className="bg-light min-vh-100 py-3")
    
    # Callback to update tab content
    @app.callback(
        Output('tab-content', 'children'),
        Input('tabs', 'value')
    )
    def render_tab_content(tab):
        if tab == 'overview':
            return html.Div([
                html.Div([
                    html.Div([
                        html.H4("Average UPDRS Scores", className="card-title"),
                        dcc.Graph(figure=bar_chart)
                    ], className="card p-3 mb-4"),
                    
                    html.Div([
                        html.H4("Symptom Radar Chart", className="card-title"),
                        dcc.Graph(figure=radar_chart)
                    ], className="card p-3 mb-4"),
                    
                    html.Div([
                        html.H4("UPDRS Question Descriptions", className="card-title"),
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.Span("", className="color-dot", 
                                             style={"backgroundColor": COLORS[i % len(COLORS)]}),
                                    html.Span(f"Q{i+1}: {QUESTION_LABELS[f'Q{i+1}']}")
                                ], className="col-md-6 mb-2 d-flex align-items-center")
                            for i in range(8)
                            ], className="row")
                        ], className="card-body")
                    ], className="card p-3"),
                    
                    html.Div([
                        html.H4("About UPDRS", className="card-title"),
                        html.P([
                            "The Unified Parkinson's Disease Rating Scale (UPDRS) is a comprehensive assessment used to measure the severity and progression of Parkinson's disease.",
                            html.Br(),
                            "Each question is scored from 0 (normal) to 4 (severe symptoms)."
                        ]),
                    ], className="card p-3 mt-4"),
                ])
            ])
        
        elif tab == 'timetrends':
            return html.Div([
                html.Div([
                    html.H4("Score Trends Over Time", className="card-title"),
                    dcc.Graph(figure=time_series_chart)
                ], className="card p-3 mb-4"),
                
                html.Div([
                    html.H4("Response Count by Month", className="card-title"),
                    dcc.Graph(figure=response_count_chart)
                ], className="card p-3")
            ])
        
        elif tab == 'distribution':
            return html.Div([
                html.Div([
                    html.H4("Score Distributions by Question", className="card-title"),
                    html.P("Distribution of scores (0-4) for each UPDRS question"),
                    
                    html.Div([
                        html.Div([
                            dcc.Graph(figure=distribution_charts[i])
                        ], className="col-md-6 mb-4")
                    for i in range(8)
                    ], className="row")
                ], className="card p-3")
            ])
        
        return html.Div("No content selected")
    
    return app

def main():
    # Set file path - update this to your local file path
    file_path = r'E:\U the mind company\UPDRS (8-Q) (Responses).xlsx'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        print("Please place the UPDRS 8Q Responses.xlsx file in the same directory as this script,")
        print("or update the file_path variable with the correct path.")
        return
    
    try:
        # Process data
        print("Loading and processing data...")
        df = load_and_process_data(file_path)
        
        # Calculate metrics
        avg_scores_df = calculate_average_scores(df)
        time_series_df = prepare_time_series_data(df)
        distribution_df = prepare_distribution_data(df)
        
        # Initialize and run dashboard
        print("Initializing dashboard...")
        app = init_dashboard(df, avg_scores_df, time_series_df, distribution_df)
        
        print("\nDashboard initialized successfully!")
        print("Opening dashboard at http://127.0.0.1:8050/")
        print("Press Ctrl+C to quit.")
        
        # Run the app
        app.run_server(debug=False)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()