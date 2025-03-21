import pandas as pd
from ortools.linear_solver import pywraplp
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2

# Load data
locations = pd.read_csv('locations.csv')
demand_df = pd.read_csv('customer_demand.csv')
transit_df = pd.read_csv('transit_matrix.csv')
loc_constraints = pd.read_csv('location_constraints.csv')
prod_constraints = pd.read_csv('production_constraints.csv')
lane_constraints = pd.read_csv('lane_constraints.csv')

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def run_optimization(rate_increase, min_fulfillment):
    location_types = dict(zip(locations['name'], locations['type']))
    dc_capacity = {row['location_name']: {'max_capacity_units': row['max_capacity_units'],
                                        'throughput_limit_units_per_day': row['throughput_limit_units_per_day']}
                  for _, row in loc_constraints.iterrows() if row['location_type'] == 'DC'}
    production_capacity = dict(zip(prod_constraints['plant_name'], prod_constraints['max_production_per_day']))
    customer_demand = dict(zip(demand_df['customer_id'], demand_df['demand']))
    lane_cost = {(row['origin_id'], row['dest_id']): row['lane_cost_usd'] for _, row in transit_df.iterrows()}
    lane_distance = {(row['origin_id'], row['dest_id']): row['distance_miles'] for _, row in transit_df.iterrows()}
    lane_cap = {(row['origin_id'], row['dest_id']): row['max_shipments_per_day'] for _, row in lane_constraints.iterrows()}
    
    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        return None
    
    flow_vars = {(o, d): solver.NumVar(0, lane_cap.get((o, d), 1e9), f'flow_{o}_to_{d}')
                for (o, d) in lane_cost.keys()}
    shortfall_vars = {cust: solver.NumVar(0, d, f'shortfall_{cust}')
                     for cust, d in customer_demand.items()}
    
    for plant, capacity in production_capacity.items():
        outflow = [flow_vars[(o, d)] for (o, d) in flow_vars if o == plant]
        solver.Add(solver.Sum(outflow) <= capacity)
    
    for dc, cap_info in dc_capacity.items():
        inbound = [flow_vars[(o, d)] for (o, d) in flow_vars if d == dc]
        outbound = [flow_vars[(o, d)] for (o, d) in flow_vars if o == dc]
        solver.Add(solver.Sum(inbound) + solver.Sum(outbound) <= cap_info['throughput_limit_units_per_day'])
    
    for cust, demand in customer_demand.items():
        inbound = [flow_vars[(o, d)] for (o, d) in flow_vars if d == cust]
        solver.Add(solver.Sum(inbound) + shortfall_vars[cust] == demand)
    
    for (o, d), max_ship in lane_cap.items():
        solver.Add(flow_vars[(o, d)] <= max_ship)
    
    # Minimum demand fulfillment constraint
    total_demand = sum(customer_demand.values())
    total_shortfall = solver.Sum(shortfall_vars.values())
    solver.Add(total_shortfall <= (1 - min_fulfillment) * total_demand)
    
    # Objective with rate increase (as a percentage)
    BIG_PENALTY = 10000
    cost_multiplier = 1 + (rate_increase / 100.0)  # Convert % increase to multiplier (e.g., 5% -> 1.05)
    objective = solver.Sum(flow_vars[(o, d)] * lane_cost[(o, d)] * cost_multiplier for (o, d) in flow_vars) + \
                solver.Sum(sf * BIG_PENALTY for sf in shortfall_vars.values())
    solver.Minimize(objective)
    
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        return flow_vars, shortfall_vars, location_types, lane_distance, lane_cost, solver
    return None

# Dash app
app = dash.Dash(__name__)
server = app.server  # For Render

app.layout = html.Div([
    html.H1("Cost Impact Dashboard", style={'textAlign': 'center', 'marginBottom': '10px'}),
    html.Div([
        html.Label("Transportation Rate Increase (%):", style={'marginRight': '10px'}),
        dcc.Input(
            id='rate-increase-input',
            type='number',
            value=5.0,  # Default to 5%
            min=0.0,
            max=10.0,
            step=0.1,
            style={'width': '200px', 'marginRight': '20px'}
        ),
        html.Label("Minimum Demand Fulfillment (%):", style={'marginRight': '10px'}),
        dcc.Slider(
            id='min-fulfillment-slider',
            min=80,
            max=100,
            step=1,
            value=90,
            marks={i: f'{i}%' for i in range(80, 101, 5)},
            tooltip={'placement': 'bottom', 'always_visible': True}
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            html.Div([
                html.Div([
                    dash_table.DataTable(
                        id='metrics-table',
                        style_table={'width': '100%'}
                    ),
                    dcc.Graph(
                        id='cost-chart',
                        style={'height': '350px', 'marginTop': '10px'}
                    )
                ], style={
                    'width': '40%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '10px'
                }),
                dcc.Graph(
                    id='network-map',
                    style={
                        'width': '60%',
                        'display': 'inline-block',
                        'height': '550px'
                    }
                )
            ], style={'display': 'flex', 'flexDirection': 'row', 'height': 'calc(100vh - 150px)'})
        ]
    )
])

@app.callback(
    [Output('metrics-table', 'data'),
     Output('metrics-table', 'columns'),
     Output('network-map', 'figure'),
     Output('cost-chart', 'figure')],
    [Input('rate-increase-input', 'value'),
     Input('min-fulfillment-slider', 'value')]
)
def update_dashboard(rate_increase, min_fulfillment):
    if rate_increase is None or rate_increase < 0.0 or rate_increase > 10.0:
        empty_fig = go.Figure()
        return ([], [], empty_fig, empty_fig)
    if min_fulfillment is None or min_fulfillment < 80 or min_fulfillment > 100:
        empty_fig = go.Figure()
        return ([], [], empty_fig, empty_fig)
    
    # Convert min_fulfillment to fraction
    min_fulfillment = min_fulfillment / 100.0
    
    # Run three optimizations: 0%, user input, and user input + 2%
    rate_increases = [0.0, rate_increase, rate_increase + 2.0]
    results = [run_optimization(r, min_fulfillment) for r in rate_increases]
    
    if any(r is None for r in results):
        empty_fig = go.Figure()
        return ([], [], empty_fig, empty_fig)
    
    # Calculate metrics for all scenarios
    total_demand = sum(demand_df['demand'])
    metrics_data = []
    total_costs = []
    
    for i, (r, result) in enumerate(zip(rate_increases, results)):
        flow_vars, shortfall_vars, location_types, lane_distance, lane_cost, _ = result
        total_shortfall = sum(sf.solution_value() for sf in shortfall_vars.values())
        fulfilled_demand = total_demand - total_shortfall
        
        cost_multiplier = 1 + (r / 100.0)  # Convert % increase to multiplier
        transport_cost = sum(flow_vars[(o, d)].solution_value() * lane_cost[(o, d)] * cost_multiplier
                            for (o, d) in flow_vars)
        shortfall_cost = sum(sf.solution_value() * 10000 for sf in shortfall_vars.values())
        total_cost = transport_cost + shortfall_cost
        
        total_costs.append(total_cost)
        
        scenario = f'{rate_increases[i]:.1f}%'
        metrics_data.append({
            'Scenario': scenario,
            'Transport Cost': f'${transport_cost:.0f}',
            'Shortfall Cost': f'${shortfall_cost:.0f}',
            'Total Cost': f'${total_cost:.0f}',
            '% Fulfilled': f'{(fulfilled_demand/total_demand*100):.1f}%'
        })
    
    # Network Map (using user input scenario, i.e., middle scenario)
    flow_vars, _, location_types, lane_distance, _, _ = results[1]
    map_fig = go.Figure()
    map_fig.add_trace(go.Scattergeo(
        lon=locations['longitude'],
        lat=locations['latitude'],
        text=locations['name'],
        mode='markers',
        marker=dict(size=12, color=['red' if t == 'DC' else 'blue' if t == 'Plant' else 'green' for t in locations['type']])
    ))
    
    for (o, d), var in flow_vars.items():
        flow_value = var.solution_value()
        if flow_value > 0:
            o_loc = locations[locations['name'] == o].iloc[0]
            d_loc = locations[locations['name'] == d].iloc[0]
            color = 'purple' if d.startswith('CUST_') else 'gray'
            width = max(1, flow_value / 2)
            map_fig.add_trace(go.Scattergeo(
                lon=[o_loc['longitude'], d_loc['longitude']],
                lat=[o_loc['latitude'], d_loc['latitude']],
                mode='lines',
                line=dict(width=width, color=color),
                opacity=0.7,
                hoverinfo='text',
                text=f'{o} to {d}: {flow_value:.1f} units'
            ))
    
    map_fig.update_layout(
        geo=dict(scope='usa', projection_type='albers usa'),
        showlegend=False,
        height=550,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    # Cost Chart
    cost_fig = go.Figure()
    cost_fig.add_trace(go.Scatter(
        x=rate_increases,
        y=total_costs,
        mode='lines+markers',
        line=dict(color='blue'),
        marker=dict(size=10),
        hovertemplate='Rate Increase: %{x:.1f}%<br>Total Cost: $%{y:.0f}'
    ))
    cost_fig.update_layout(
        title='Total Cost vs Transportation Rate Increase',
        xaxis_title='Rate Increase (%)',
        yaxis_title='Total Cost ($)',
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Table columns
    columns = [
        {'name': 'Scenario', 'id': 'Scenario'},
        {'name': 'Transport Cost', 'id': 'Transport Cost'},
        {'name': 'Shortfall Cost', 'id': 'Shortfall Cost'},
        {'name': 'Total Cost', 'id': 'Total Cost'},
        {'name': '% Fulfilled', 'id': '% Fulfilled'}
    ]
    
    return (metrics_data, columns, map_fig, cost_fig)

if __name__ == '__main__':
    app.run_server(debug=True)