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

UNITS_PER_TRUCK = 100

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def run_optimization(rate_increase, dc_throughput_cost, min_fulfillment):
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
    
    total_demand = sum(customer_demand.values())
    total_shortfall = solver.Sum(shortfall_vars.values())
    solver.Add(total_shortfall <= (1 - min_fulfillment) * total_demand)
    
    cost_multiplier = 1 + (rate_increase / 100.0)
    transport_cost = solver.Sum((flow_vars[(o, d)] / UNITS_PER_TRUCK) * lane_cost[(o, d)] * cost_multiplier
                               for (o, d) in flow_vars)
    
    dc_throughput = {}
    for dc in dc_capacity.keys():
        inbound = [flow_vars[(o, d)] for (o, d) in flow_vars if d == dc]
        outbound = [flow_vars[(o, d)] for (o, d) in flow_vars if o == dc]
        dc_throughput[dc] = solver.Sum(inbound) + solver.Sum(outbound)
    dc_cost = solver.Sum(dc_throughput[dc] * dc_throughput_cost for dc in dc_capacity.keys())
    
    BIG_PENALTY = 10000
    objective = transport_cost + dc_cost + solver.Sum(sf * BIG_PENALTY for sf in shortfall_vars.values())
    solver.Minimize(objective)
    
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        return flow_vars, shortfall_vars, location_types, lane_distance, lane_cost, dc_throughput, solver
    return None

# Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Cost Impact Dashboard", style={'textAlign': 'center', 'marginBottom': '10px'}),
    html.Div([
        html.Label("Transportation Rate Increase (%):", style={'marginRight': '10px'}),
        dcc.Input(
            id='rate-increase-input',
            type='number',
            value=1.0,
            min=0.0,
            max=20.0,
            step=0.1,
            style={'width': '200px', 'marginRight': '20px'}
        ),
        html.Label("DC Throughput Cost ($/unit):", style={'marginRight': '10px'}),
        dcc.Input(
            id='dc-throughput-cost-input',
            type='number',
            value=2.0,
            min=0.0,
            max=100.0,
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
                    dash_table.DataTable(id='metrics-table', style_table={'width': '100%'}),
                    dcc.Graph(id='cost-chart', style={'height': '350px', 'marginTop': '10px'}),
                    dcc.Graph(id='cost-distribution-chart', style={'height': '350px', 'marginTop': '10px'})
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                dcc.Graph(id='network-map', style={'width': '60%', 'display': 'inline-block', 'height': '750px'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'height': 'calc(100vh - 150px)'})
        ]
    )
])

@app.callback(
    [Output('metrics-table', 'data'),
     Output('metrics-table', 'columns'),
     Output('network-map', 'figure'),
     Output('cost-chart', 'figure'),
     Output('cost-distribution-chart', 'figure')],
    [Input('rate-increase-input', 'value'),
     Input('dc-throughput-cost-input', 'value'),
     Input('min-fulfillment-slider', 'value')]
)
def update_dashboard(rate_increase, dc_throughput_cost, min_fulfillment):
    if rate_increase is None or rate_increase < 0.0 or rate_increase > 20.0 or \
       dc_throughput_cost is None or dc_throughput_cost < 0.0 or dc_throughput_cost > 50.0:
        empty_fig = go.Figure()
        return ([], [], empty_fig, empty_fig, empty_fig)
    min_fulfillment = min_fulfillment / 100.0
    
    result = run_optimization(rate_increase, dc_throughput_cost, min_fulfillment)
    if result is None:
        empty_fig = go.Figure()
        return ([], [], empty_fig, empty_fig, empty_fig)
    
    flow_vars, shortfall_vars, location_types, lane_distance, lane_cost, dc_throughput, _ = result
    
    total_demand = sum(demand_df['demand'])
    total_shortfall = sum(sf.solution_value() for sf in shortfall_vars.values())
    fulfilled_demand = total_demand - total_shortfall
    
    cost_multiplier = 1 + (rate_increase / 100.0)
    transport_cost = sum((flow_vars[(o, d)].solution_value() / UNITS_PER_TRUCK) * lane_cost[(o, d)] * cost_multiplier
                        for (o, d) in flow_vars)
    dc_cost = sum(dc_throughput[dc].solution_value() * dc_throughput_cost for dc in dc_throughput.keys())
    shortfall_cost = sum(sf.solution_value() * 10000 for sf in shortfall_vars.values())
    total_cost = transport_cost + dc_cost + shortfall_cost
    
    transport_cost_per_unit = transport_cost / fulfilled_demand if fulfilled_demand > 0 else 0
    dc_cost_per_unit = dc_cost / fulfilled_demand if fulfilled_demand > 0 else 0
    total_cost_per_unit = (transport_cost + dc_cost) / fulfilled_demand if fulfilled_demand > 0 else 0
    
    print(f"Total Demand: {total_demand}, Fulfilled: {fulfilled_demand}")
    print(f"Transport Cost: ${transport_cost:.2f}, DC Cost: ${dc_cost:.2f}, Ratio (Trans/DC): {transport_cost/dc_cost if dc_cost > 0 else 'inf'}")
    print(f"Per Unit - Transport: ${transport_cost_per_unit:.2f}, DC: ${dc_cost_per_unit:.2f}, Total: ${total_cost_per_unit:.2f}")
    
    metrics_data = [{
        'Scenario': f'{rate_increase}% Rate, ${dc_throughput_cost}/unit DC',
        'Transport Cost': f'${transport_cost:.0f}',
        'DC Cost': f'${dc_cost:.0f}',
        'Shortfall Cost': f'${shortfall_cost:.0f}',
        'Total Cost': f'${total_cost:.0f}',
        '% Fulfilled': f'{(fulfilled_demand/total_demand*100):.1f}%'
    }]
    columns = [{'name': k, 'id': k} for k in metrics_data[0].keys()]
    
    # Customizable map colors
    MARKER_COLORS = {
        'DC': '#FFA500',      # Orange for DCs
        'Plant': '#00CED1',   # Turquoise for Plants
        'Customer': '#32CD32' # Lime green for Customers
    }
    LINE_COLORS = {
        'to_customer': '#FF4500',  # Orange-red for flows to customers
        'other': '#4682B4'         # Steel blue for other flows
    }
    
    map_fig = go.Figure()
    map_fig.add_trace(go.Scattergeo(
        lon=locations['longitude'],
        lat=locations['latitude'],
        text=locations['name'],
        mode='markers',
        marker=dict(
            size=12,
            color=[MARKER_COLORS['DC'] if t == 'DC' else MARKER_COLORS['Plant'] if t == 'Plant' else MARKER_COLORS['Customer']
                   for t in locations['type']]
        )
    ))
    for (o, d), var in flow_vars.items():
        flow_value = var.solution_value()
        if flow_value > 0:
            o_loc = locations[locations['name'] == o].iloc[0]
            d_loc = locations[locations['name'] == d].iloc[0]
            color = LINE_COLORS['to_customer'] if d.startswith('CUST_') else LINE_COLORS['other']
            width = max(1, flow_value / 2)
            map_fig.add_trace(go.Scattergeo(
                lon=[o_loc['longitude'], d_loc['longitude']],
                lat=[o_loc['latitude'], d_loc['latitude']],
                mode='lines',
                line=dict(width=1, color=color),
                opacity=0.7,
                hoverinfo='text',
                text=f'{o} to {d}: {flow_value:.1f} units'
            ))
    map_fig.update_layout(
        geo=dict(scope='usa', projection_type='albers usa'),
        showlegend=False,
        height=750,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    cost_fig = go.Figure()
    cost_fig.add_trace(go.Bar(
        x=['Transport', 'DC', 'Shortfall'],
        y=[transport_cost, dc_cost, shortfall_cost],
        marker_color=['#1f77b4', '#ff7f0e', '#d62728'],
        hovertemplate='%{x}: $%{y:.0f}'
    ))
    cost_fig.update_layout(
        title='Total Cost Breakdown',
        yaxis_title='Cost ($)',
        height=350
    )
    
    dist_fig = go.Figure()
    dist_fig.add_trace(go.Bar(
        x=['Transportation', 'DC Throughput', 'Total (Trans + DC)'],
        y=[transport_cost_per_unit, dc_cost_per_unit, total_cost_per_unit],
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        hovertemplate='%{x}: $%{y:.2f}/unit<br>Total: $%{customdata:.0f}',
        customdata=[transport_cost, dc_cost, transport_cost + dc_cost]
    ))
    dist_fig.update_layout(
        title='Cost Distribution per Unit Fulfilled',
        yaxis_title='Cost per Unit ($)',
        height=350
    )
    
    return (metrics_data, columns, map_fig, cost_fig, dist_fig)

if __name__ == '__main__':
    app.run_server(debug=True)