import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
from dataclasses import dataclass
from time import time
from itertools import product
# Importing custom modules
import beamforming.beamforming_module_para as bm
import beamforming.read_sims_module as rm
import beamforming.signal_module as sm
import beamforming.sampling_module_para as sp
import beamforming.parameter_reconstruction as rec

from beamforming.adf_weights import compute_adf_weights

R2D = 180. / np.pi  # Conversion factor from radians to degrees

@dataclass
class Config:
    """
    Configuration class to hold all parameters for the reconstruction process.
    """
    path_to_library: str = "/volatile/home/af274537/Documents/WorkingDir/HERON/SphericalPhasing_2/data/TauLibrary_972Events_Eshower_2e7-1e9GeV_XYZCoordinates_CorrXmax.npz"
    save_path: str = './results_adf'
    sampling: str = 'random'
    n_walkers: int = 200
    n_steps: int = 400
    step_size: float = 1000.
    f_min: float = 0
    f_max: float = 5000
    noise_std: float = 0
    jitter_std: float = 0
    bounds: str = 'flat_sphere'  # Options: 'cubic', 'sphere', 'flat_sphere'
    r: float = 10.e3  # Radius for spherical bounds
    thickness: float = 4e3  # Thickness for flat spherical bounds
    intens_method: str = 'amplitude'  # Intensity calculation method
    temp: float = 5000.  # Temperature for sampling
    x_max_method: str = 'max'  # Method for finding maximum
    n_best_walkers: int = 0  # Number of best walkers to consider
    sep_walkers: bool = False  # Separate walkers flag
    burn_in: int = 0  # Number of burn-in steps to discard

def main(config: Config):
    name_prefix = "LEevents_"
    tau_events, antennas, efields, sttimes = rm.read_library(config.path_to_library)
    Eventlist = list(range(0, 250))  # List of events to process

    # Create interactive Plotly figure with event selector
    create_interactive_figure(
        tau_events, antennas, efields, sttimes, 
        Eventlist, config
    )


def compute_event_data(eventi, tau_events, antennas, efields, sttimes, config):
    """Compute all data needed for a single event."""
    np.random.seed(eventi)
    
    # Read event data
    (azim, zen, en_nu, en_tau, tau_pos, xmax_pos, event_efield, antenna_pos, 
     xant, yant, zant, signals, phi, theta, shower_dir) = rm.read_trace(
            eventi, tau_events, antennas, efields, sttimes, 
            config.f_min, config.f_max, config.noise_std, config.jitter_std
        )

    Xsource = xmax_pos
    shifted_times = bm.spherical_phasing(Xsource[0], Xsource[1], Xsource[2], xant, yant, zant, signals[0, :, 0])
    range_theta = np.linspace(-4, 4, 100) * np.pi / 180 + theta
    range_phi = np.linspace(-4, 4, 100) * np.pi / 180 + phi
    All_intensity = np.zeros((len(range_theta), len(range_phi)))
    
    for i, theta_i in enumerate(range_theta):
        for j, phi_j in enumerate(range_phi):
            weights = compute_adf_weights(Xsource[0], Xsource[1], Xsource[2], theta_i, phi_j, xant, yant, zant)
            output = bm.signals_summing_ref_adf(signals, shifted_times, weights[None, :])
            norm_2 = output[0, 1, :]**2 + output[0, 2, :]**2 + output[0, 3, :]**2
            compute_intensity = sp.compute_amp if config.intens_method == 'amplitude' else sp.compute_power
            intensity = compute_intensity(norm_2)
            All_intensity[i, j] = intensity
    
    # Compute antenna amplitudes for reference
    amps = np.max(np.linalg.norm(signals[1:], axis=0), axis=-1)
    
    # Compute layout direction
    k_layout = np.average(antenna_pos, axis=0, weights=amps) - xmax_pos
    k_layout /= np.linalg.norm(k_layout)
    theta_layout = np.arccos(-k_layout[2])
    phi_layout = np.arctan2(-k_layout[1], -k_layout[0])
    
    # Create grid for ADF heatmap
    x_margin = (xant.max() - xant.min()) * 0.1
    y_margin = (yant.max() - yant.min()) * 0.1
    grid_x = np.linspace(xant.min() - x_margin, xant.max() + x_margin, 100)
    grid_y = np.linspace(yant.min() - y_margin, yant.max() + y_margin, 100)
    
    return {
        'range_theta': range_theta.tolist(),
        'range_phi': range_phi.tolist(),
        'All_intensity': All_intensity.tolist(),
        'theta_true': float(theta),
        'phi_true': float(phi),
        'theta_layout': float(theta_layout),
        'phi_layout': float(phi_layout),
        'xant': xant.tolist(),
        'yant': yant.tolist(),
        'zant': zant.tolist(),
        'amps': amps.tolist(),
        'signals': signals.tolist(),
        'shifted_times': shifted_times.tolist(),
        'Xsource': Xsource.tolist(),
        'grid_x': grid_x.tolist(),
        'grid_y': grid_y.tolist(),
        'selected_theta': float(theta),
        'selected_phi': float(phi),
        'eventi': eventi
    }


def create_interactive_figure(tau_events, antennas, efields, sttimes, Eventlist, config):
    """
    Create an interactive Plotly figure with three subplots:
    1. Intensity map (clickable)
    2. Summed signal traces
    3. Antenna layout with ADF weights heatmap
    Plus an event selector dropdown.
    """
    import dash
    from dash import dcc, html, ctx
    from dash.dependencies import Input, Output, State
    
    # Create the Dash app
    app = dash.Dash(__name__)
    
    # Compute initial event data
    initial_event = Eventlist[0]
    print(f"Computing initial data for event {initial_event}...")
    initial_data = compute_event_data(initial_event, tau_events, antennas, efields, sttimes, config)
    
    app.layout = html.Div([
        html.H1(id='title', children=f"Interactive ADF Beamforming - Event {initial_event}"),
        
        # Event selector
        html.Div([
            html.Label("Select Event: ", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='event-selector',
                options=[{'label': f'Event {e}', 'value': e} for e in Eventlist],
                value=initial_event,
                style={'width': '200px', 'display': 'inline-block'}
            ),
            html.Span(id='loading-indicator', children="", style={'marginLeft': '20px', 'color': 'orange'})
        ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
        
        html.Div([
            html.Div([
                html.H3("Intensity Map (Click to select direction)"),
                dcc.Graph(id='intensity-map', style={'height': '500px'})
            ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.H3("Summed Signal"),
                dcc.Graph(id='signal-plot', style={'height': '500px'})
            ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.H3("Antenna Layout with ADF Weights"),
                dcc.Graph(id='antenna-layout', style={'height': '500px'})
            ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
        ]),
        html.Div([
            html.P(id='click-info', children=f"Selected: θ={np.degrees(initial_data['theta_true']):.2f}°, φ={np.degrees(initial_data['phi_true']):.2f}°")
        ]),
        # Store data for callbacks
        dcc.Store(id='stored-data', data=initial_data),
        # Store library reference info (we can't store the actual library, so we recompute)
        dcc.Store(id='config-store', data={
            'f_min': config.f_min,
            'f_max': config.f_max,
            'noise_std': config.noise_std,
            'jitter_std': config.jitter_std,
            'intens_method': config.intens_method
        })
    ])
    
    # Store references for the callback (using closure)
    library_data = {
        'tau_events': tau_events,
        'antennas': antennas,
        'efields': efields,
        'sttimes': sttimes,
        'config': config
    }
    
    @app.callback(
        [Output('intensity-map', 'figure'),
         Output('signal-plot', 'figure'),
         Output('antenna-layout', 'figure'),
         Output('click-info', 'children'),
         Output('stored-data', 'data'),
         Output('title', 'children')],
        [Input('intensity-map', 'clickData'),
         Input('event-selector', 'value')],
        [State('stored-data', 'data')]
    )
    def update_plots(clickData, selected_event, stored_data):
        # Check which input triggered the callback
        triggered_id = ctx.triggered_id if ctx.triggered_id else 'event-selector'
        
        # If event changed, recompute everything
        if triggered_id == 'event-selector' or stored_data.get('eventi') != selected_event:
            print(f"Loading event {selected_event}...")
            stored_data = compute_event_data(
                selected_event, 
                library_data['tau_events'],
                library_data['antennas'],
                library_data['efields'],
                library_data['sttimes'],
                library_data['config']
            )
            # Reset click data when changing events
            clickData = None
        
        # Retrieve stored data
        range_theta_arr = np.array(stored_data['range_theta'])
        range_phi_arr = np.array(stored_data['range_phi'])
        All_intensity_arr = np.array(stored_data['All_intensity'])
        theta_true_val = stored_data['theta_true']
        phi_true_val = stored_data['phi_true']
        theta_layout_val = stored_data['theta_layout']
        phi_layout_val = stored_data['phi_layout']
        xant_arr = np.array(stored_data['xant'])
        yant_arr = np.array(stored_data['yant'])
        zant_arr = np.array(stored_data['zant'])
        amps_arr = np.array(stored_data['amps'])
        signals_arr = np.array(stored_data['signals'])
        shifted_times_arr = np.array(stored_data['shifted_times'])
        Xsource_arr = np.array(stored_data['Xsource'])
        grid_x_arr = np.array(stored_data['grid_x'])
        grid_y_arr = np.array(stored_data['grid_y'])
        
        # Determine selected theta/phi
        if clickData is not None and triggered_id == 'intensity-map':
            clicked_phi = clickData['points'][0]['x']
            clicked_theta = clickData['points'][0]['y']
            selected_theta = np.radians(clicked_theta)
            selected_phi = np.radians(clicked_phi)
        else:
            selected_theta = stored_data['selected_theta']
            selected_phi = stored_data['selected_phi']
        
        # Update stored data
        stored_data['selected_theta'] = float(selected_theta)
        stored_data['selected_phi'] = float(selected_phi)
        
        # Create intensity map figure
        fig_intensity = go.Figure()
        fig_intensity.add_trace(go.Heatmap(
            x=np.degrees(range_phi_arr),
            y=np.degrees(range_theta_arr),
            z=All_intensity_arr,
            colorscale='Viridis',
            colorbar=dict(title='Intensity')
        ))
        fig_intensity.add_trace(go.Scatter(
            x=[np.degrees(phi_true_val)],
            y=[np.degrees(theta_true_val)],
            mode='markers',
            marker=dict(color='red', size=12, symbol='x'),
            name='True Direction'
        ))
        fig_intensity.add_trace(go.Scatter(
            x=[np.degrees(phi_layout_val)],
            y=[np.degrees(theta_layout_val)],
            mode='markers',
            marker=dict(color='blue', size=12, symbol='circle'),
            name='Layout Direction'
        ))
        fig_intensity.add_trace(go.Scatter(
            x=[np.degrees(selected_phi)],
            y=[np.degrees(selected_theta)],
            mode='markers',
            marker=dict(color='white', size=10, symbol='cross'),
            name='Selected'
        ))
        fig_intensity.update_layout(
            xaxis_title='Phi (degrees)',
            yaxis_title='Theta (degrees)',
            title='Beamformed Intensity Map',
            showlegend=True
        )
        
        # Compute weighted signals for selected direction
        weights = compute_adf_weights(Xsource_arr[0], Xsource_arr[1], Xsource_arr[2],
                                       selected_theta, selected_phi, xant_arr, yant_arr, zant_arr)
        output = bm.signals_summing_ref_adf(signals_arr, shifted_times_arr, weights[None, :])
        
        # Create signal plot
        time_axis = output[0, 0, :]
        fig_signal = go.Figure()
        fig_signal.add_trace(go.Scatter(
            x=time_axis,
            y=output[0, 1, :],
            mode='lines',
            name='Ex',
            line=dict(color='red')
        ))
        fig_signal.add_trace(go.Scatter(
            x=time_axis,
            y=output[0, 2, :],
            mode='lines',
            name='Ey',
            line=dict(color='green')
        ))
        fig_signal.add_trace(go.Scatter(
            x=time_axis,
            y=output[0, 3, :],
            mode='lines',
            name='Ez',
            line=dict(color='blue')
        ))
        # Add norm
        norm_signal = np.sqrt(output[0, 1, :]**2 + output[0, 2, :]**2 + output[0, 3, :]**2)
        fig_signal.add_trace(go.Scatter(
            x=time_axis,
            y=norm_signal,
            mode='lines',
            name='|E|',
            line=dict(color='black', dash='dash')
        ))
        fig_signal.update_layout(
            xaxis_title='Time (ns)',
            yaxis_title='Electric Field',
            title=f'Weighted Sum (θ={np.degrees(selected_theta):.2f}°, φ={np.degrees(selected_phi):.2f}°)',
            showlegend=True
        )
        
        # Create antenna layout with ADF heatmap
        grid_xx, grid_yy = np.meshgrid(grid_x_arr, grid_y_arr)
        grid_zant = zant_arr.mean() * np.ones_like(grid_xx.flatten())
        weights_grid = compute_adf_weights(Xsource_arr[0], Xsource_arr[1], Xsource_arr[2],
                                            selected_theta, selected_phi,
                                            grid_xx.flatten(), grid_yy.flatten(), grid_zant)
        grid_z = weights_grid.reshape(grid_xx.shape)
        
        fig_antenna = go.Figure()
        # Add ADF heatmap
        fig_antenna.add_trace(go.Heatmap(
            x=grid_x_arr / 1e3,
            y=grid_y_arr / 1e3,
            z=grid_z,
            colorscale='Viridis',
            colorbar=dict(title='ADF Weight'),
            opacity=0.7
        ))
        # Add antenna scatter
        fig_antenna.add_trace(go.Scatter(
            x=xant_arr / 1e3,
            y=yant_arr / 1e3,
            mode='markers',
            marker=dict(
                size=8,
                # color=10 * np.log10(amps_arr),
                color=amps_arr,
                colorscale='Hot',
                # colorbar=dict(title='Amplitude (dB)', x=1.15),
                colorbar=dict(title='Amplitude (µV/m)', x=1.15),
                line=dict(color='white', width=1)
            ),
            name='Antennas'
        ))
        fig_antenna.update_layout(
            xaxis_title='X Position (km)',
            yaxis_title='Y Position (km)',
            title='Antenna Layout with ADF Weights',
            yaxis=dict(scaleanchor='x', scaleratio=1)
        )
        
        click_info = f"Selected: θ={np.degrees(selected_theta):.2f}°, φ={np.degrees(selected_phi):.2f}°"
        title = f"Interactive ADF Beamforming - Event {selected_event}"
        
        return fig_intensity, fig_signal, fig_antenna, click_info, stored_data, title
    
    print("Starting interactive Dash server...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    config = Config()
    main(config)