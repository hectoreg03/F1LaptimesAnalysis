import matplotlib.pyplot as plt
import seaborn as sns
import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.collections import LineCollection
from matplotlib import colormaps

# --- CONFIGURACIÓN DE ESTILO ---
F1_RED = '#e10600'
F1_BG = '#15151e' 
STANDARD_FIGSIZE = (13, 7)

def apply_f1_style(ax, title):
    """Aplica estilos consistentes a los ejes de Matplotlib."""
    ax.set_title(title, color='white', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(ax.get_xlabel(), color='white', fontsize=12)
    ax.set_ylabel(ax.get_ylabel(), color='white', fontsize=12)
    ax.tick_params(colors='white', which='both', labelsize=10)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.4)
    ax.set_facecolor(F1_BG)

# --- HELPER: ROTACIÓN ---
def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

def get_track_map_with_corners(session):
    """Genera el mapa del circuito con las curvas numeradas y rotado."""
    try:
        circuit_info = session.get_circuit_info()
        lap = session.laps.pick_fastest()
        pos = lap.get_pos_data()

        # 1. Obtener coordenadas y rotar
        track = pos.loc[:, ('X', 'Y')].to_numpy()
        track_angle = circuit_info.rotation / 180 * np.pi
        rotated_track = rotate(track, angle=track_angle)

        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
        fig.patch.set_facecolor(F1_BG)
        ax.set_facecolor(F1_BG)

        # 2. Dibujar Pista (Línea blanca simple)
        ax.plot(rotated_track[:, 0], rotated_track[:, 1], color='white', linewidth=4)

        # 3. Dibujar Curvas
        offset_vector = [500, 0]
        for _, corner in circuit_info.corners.iterrows():
            txt = f"{corner['Number']}{corner['Letter']}"
            offset_angle = corner['Angle'] / 180 * np.pi
            offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
            
            text_x = corner['X'] + offset_x
            text_y = corner['Y'] + offset_y
            
            text_x, text_y = rotate([text_x, text_y], angle=track_angle)
            track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

            # Línea conectora
            ax.plot([track_x, text_x], [track_y, text_y], color='#888', linestyle='--')
            
            # Círculo marcador
            ax.scatter(text_x, text_y, color='#222', edgecolor='white', s=300, zorder=5)
            
            # Número
            ax.text(text_x, text_y, txt, va='center', ha='center', 
                    size='small', color='white', fontweight='bold', zorder=6)

        ax.set_title(f"Circuit Layout: {session.event['Location']}", color='white', fontsize=18, fontweight='bold', pad=20)
        ax.axis('equal')
        ax.axis('off')
        
        return fig
    except Exception as e:
        print(f"Error generando Track Map: {e}")
        return None

def get_speed_map(session, driver):
    """Mapa de velocidad rotado."""
    try:
        lap = session.laps.pick_drivers(driver).pick_fastest()
        tel = lap.get_telemetry()
        
        # Obtener rotación
        try:
            circuit_info = session.get_circuit_info()
            track_angle = circuit_info.rotation / 180 * np.pi
        except:
            track_angle = 0 

        # Preparar coordenadas rotadas
        x = np.array(tel['X'].values)
        y = np.array(tel['Y'].values)
        xy_points = np.array([x, y]).T
        rotated_points = rotate(xy_points, angle=track_angle)
        
        x_rot = rotated_points[:, 0]
        y_rot = rotated_points[:, 1]
        
        points = np.array([x_rot, y_rot]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        speed = tel['Speed']

        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
        fig.patch.set_facecolor(F1_BG)
        ax.set_facecolor(F1_BG)
        
        cmap = plt.get_cmap('plasma')
        norm = plt.Normalize(speed.min(), speed.max())
        
        lc = LineCollection(segments, cmap=cmap, norm=norm, linestyle='-', linewidth=5)
        lc.set_array(speed)
        line = ax.add_collection(lc)
        
        cbar = plt.colorbar(line, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Velocidad (km/h)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # --- FIX: FORZAR LÍMITES DE EJES ---
        # Al usar add_collection sin plot(), matplotlib no auto-escala. Lo hacemos manual:
        margin = 100
        ax.set_xlim(x_rot.min() - margin, x_rot.max() + margin)
        ax.set_ylim(y_rot.min() - margin, y_rot.max() + margin)

        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title(f"Speed Map - {driver}", color='white', fontweight='bold', fontsize=18, pad=20)
        return fig
    except Exception as e:
        print(f"Error en Speed Map: {e}")
        return None

def get_gear_map(session, driver):
    """Mapa de marchas rotado."""
    try:
        lap = session.laps.pick_drivers(driver).pick_fastest()
        tel = lap.get_telemetry()
        
        try:
            circuit_info = session.get_circuit_info()
            track_angle = circuit_info.rotation / 180 * np.pi
        except:
            track_angle = 0

        x = np.array(tel['X'].values)
        y = np.array(tel['Y'].values)
        xy_points = np.array([x, y]).T
        rotated_points = rotate(xy_points, angle=track_angle)
        
        x_rot = rotated_points[:, 0]
        y_rot = rotated_points[:, 1]

        points = np.array([x_rot, y_rot]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        gear = tel['nGear'].to_numpy().astype(float)

        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
        fig.patch.set_facecolor(F1_BG)
        ax.set_facecolor(F1_BG)

        cmap = colormaps['Paired']
        lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
        lc_comp.set_array(gear)
        lc_comp.set_linewidth(5)
        ax.add_collection(lc_comp)
        
        # --- FIX: FORZAR LÍMITES DE EJES TAMBIÉN AQUÍ ---
        margin = 100
        ax.set_xlim(x_rot.min() - margin, x_rot.max() + margin)
        ax.set_ylim(y_rot.min() - margin, y_rot.max() + margin)

        ax.axis('equal')
        ax.axis('off')

        cbar = plt.colorbar(mappable=lc_comp, ax=ax, boundaries=np.arange(1, 10), pad=0.02)
        cbar.set_ticks(np.arange(1.5, 9.5))
        cbar.set_ticklabels(np.arange(1, 9))
        cbar.set_label('Marcha', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        ax.set_title(f"Gear Shift Map - {driver}", color='white', fontweight='bold', fontsize=18, pad=20)
        return fig
    except Exception as e:
        return None

def get_driver_laptimes(session, driver):
    try:
        driver_laps = session.laps.pick_drivers(driver).reset_index()
        if driver_laps.empty: return None

        driver_laps['LapStatus'] = 'Ritmo'
        if 'Rainfall' in driver_laps.columns:
            driver_laps.loc[driver_laps['Rainfall'] == True, 'LapStatus'] = 'Lluvia'
        driver_laps.loc[~pd.isnull(driver_laps['PitInTime']), 'LapStatus'] = 'Pitstop'
        driver_laps.loc[~pd.isnull(driver_laps['PitOutTime']), 'LapStatus'] = 'Pitstop'

        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
        fig.patch.set_facecolor(F1_BG)
        
        sns.scatterplot(data=driver_laps, x="LapNumber", y="LapTime", ax=ax,
                        hue="Compound", palette=fastf1.plotting.get_compound_mapping(session=session),
                        style="LapStatus", markers={'Ritmo': 'o', 'Pitstop': 'X', 'Lluvia': 'D'},
                        s=120, linewidth=0, legend='auto')

        ax.invert_yaxis()
        apply_f1_style(ax, f"Lap Time Evolution - {driver}")
        
        ax.legend(frameon=True, facecolor=F1_BG, edgecolor='white', labelcolor='white', loc='upper right')
        return fig
    except Exception as e:
        return None

def get_lap_distribution(session):
    try:
        point_finishers = session.drivers[:10]
        driver_laps = session.laps.pick_drivers(point_finishers).pick_quicklaps().reset_index()
        if driver_laps.empty: return None

        finishing_order = [session.get_driver(i)["Abbreviation"] for i in point_finishers]
        driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
        fig.patch.set_facecolor(F1_BG)

        sns.violinplot(data=driver_laps, x="Driver", y="LapTime(s)", hue="Driver",
                       inner=None, density_norm="area", order=finishing_order,
                       palette=fastf1.plotting.get_driver_color_mapping(session=session),
                       ax=ax, linewidth=0, alpha=0.4)
        
        sns.swarmplot(data=driver_laps, x="Driver", y="LapTime(s)", order=finishing_order,
                      hue="Compound", palette=fastf1.plotting.get_compound_mapping(session=session),
                      hue_order=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
                      linewidth=0, size=4, ax=ax)
        
        apply_f1_style(ax, "Top 10 Pace Distribution")
        ax.set_xlabel("")
        ax.get_legend().remove()
        return fig
    except Exception as e:
        return None

def get_strategy_chart(session):
    try:
        laps = session.laps
        stints = laps[["Driver", "Stint", "Compound", "LapNumber"]].groupby(["Driver", "Stint", "Compound"]).count().reset_index()
        stints = stints.rename(columns={"LapNumber": "StintLength"})
        drivers_ordered = [session.get_driver(d)["Abbreviation"] for d in session.drivers]

        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
        fig.patch.set_facecolor(F1_BG)
        ax.set_facecolor(F1_BG)
        
        for driver in drivers_ordered:
            driver_stints = stints.loc[stints["Driver"] == driver]
            previous_stint_end = 0
            for idx, row in driver_stints.iterrows():
                compound_color = fastf1.plotting.get_compound_color(row["Compound"], session=session)
                ax.barh(y=driver, width=row["StintLength"], left=previous_stint_end,
                           color=compound_color, edgecolor="black", fill=True, height=0.6)
                previous_stint_end += row["StintLength"]

        apply_f1_style(ax, "Tyre Strategy")
        ax.invert_yaxis()
        ax.grid(axis='x', color='gray', linestyle=':', alpha=0.3)
        return fig
    except:
        return None

def get_position_changes(session, highlight_driver):
    try:
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
        fig.patch.set_facecolor(F1_BG)
        ax.set_facecolor(F1_BG)

        for drv in session.drivers:
            drv_laps = session.laps.pick_drivers(drv)
            if drv_laps.empty: continue
            abb = drv_laps['Driver'].iloc[0]

            if abb == highlight_driver:
                style = fastf1.plotting.get_driver_style(identifier=abb, style=['color', 'linestyle'], session=session)
                alpha, lw, zorder = 1.0, 4, 10
                ax.plot(drv_laps['LapNumber'], drv_laps['Position'], color='white', linewidth=6, zorder=9, alpha=0.5)
            else:
                style = fastf1.plotting.get_driver_style(identifier=abb, style=['color', 'linestyle'], session=session)
                alpha, lw, zorder = 0.3, 1.5, 1
            
            ax.plot(drv_laps['LapNumber'], drv_laps['Position'], alpha=alpha, linewidth=lw, zorder=zorder, **style)

        ax.set_ylim([20.5, 0.5])
        ax.set_yticks([1, 5, 10, 15, 20])
        apply_f1_style(ax, "Position Changes")
        return fig
    except:
        return None

def get_team_pace(session):
    try:
        laps = session.laps.pick_quicklaps()
        transformed_laps = laps.copy()
        transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()
        team_order = (transformed_laps[["Team", "LapTime (s)"]].groupby("Team").median()["LapTime (s)"].sort_values().index)
        team_palette = {team: fastf1.plotting.get_team_color(team, session=session) for team in team_order}

        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
        fig.patch.set_facecolor(F1_BG)
        
        sns.boxplot(data=transformed_laps, x="Team", y="LapTime (s)",
                    hue="Team", order=team_order, palette=team_palette,
                    whiskerprops=dict(color="white"), boxprops=dict(edgecolor="white"),
                    medianprops=dict(color="grey"), capprops=dict(color="white"), ax=ax)
        
        apply_f1_style(ax, "Team Race Pace")
        ax.set_xlabel("")
        return fig
    except:
        return None

def get_driver_weekend_laptimes(current_session, driver):
    try:
        year = current_session.event.year
        event_name = current_session.event['EventName']
        session_identifiers = ['FP1', 'FP2', 'FP3', 'Q', 'R']
        all_laps_list = []

        for ident in session_identifiers:
            try:
                sess = fastf1.get_session(year, event_name, ident)
                sess.load(laps=True, telemetry=False, weather=False, messages=False)
                drv_laps = sess.laps.pick_drivers(driver).reset_index()
                if not drv_laps.empty:
                    drv_laps['Session'] = ident
                    drv_laps['LapTimeSeconds'] = drv_laps['LapTime'].dt.total_seconds()
                    drv_laps = drv_laps.dropna(subset=['LapTimeSeconds'])
                    all_laps_list.append(drv_laps[['Session', 'LapTimeSeconds', 'Compound']])
            except: continue
        
        if not all_laps_list: return None
        weekend_df = pd.concat(all_laps_list)

        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
        fig.patch.set_facecolor(F1_BG)

        sns.violinplot(data=weekend_df, x='Session', y='LapTimeSeconds',
                       inner=None, color='#1f77b4', linewidth=0, ax=ax, alpha=0.3, cut=0)
        sns.swarmplot(data=weekend_df, x='Session', y='LapTimeSeconds',
                      hue='Compound', palette=fastf1.plotting.get_compound_mapping(session=current_session),
                      size=3, ax=ax, alpha=0.9)
        
        apply_f1_style(ax, f"Weekend Overview - {driver}")
        try:
            ax.legend(title="Compound", loc='upper right', frameon=False, labelcolor='white')
        except: pass
        return fig
    except Exception as e:
        return None

def get_race_replay(session):
    try:
        laps = session.laps.pick_drivers(session.drivers).reset_index()
        laps = laps.dropna(subset=['Time', 'LapNumber', 'LapTime'])
        laps['EndTime'] = laps['Time'].dt.total_seconds()
        laps['Duration'] = laps['LapTime'].dt.total_seconds()
        laps['StartTime'] = laps['EndTime'] - laps['Duration']
        laps = laps[laps['Duration'] > 0]
        
        start_t = laps['StartTime'].min()
        end_t = laps['EndTime'].max()
        n_frames = 60 
        time_steps = np.linspace(start_t, end_t, n_frames)
        team_map = laps.set_index('Driver')['Team'].to_dict()
        
        animation_rows = []
        for t in time_steps:
            drivers_in_frame = []
            for driver in session.drivers:
                d_laps = laps[laps['Driver'] == driver]
                if d_laps.empty: continue
                
                if t >= d_laps['EndTime'].max():
                    prog = float(d_laps['LapNumber'].max())
                elif t <= d_laps['StartTime'].min():
                    prog = 0.0
                else:
                    curr = d_laps[(d_laps['StartTime'] <= t) & (d_laps['EndTime'] >= t)]
                    if not curr.empty:
                        r = curr.iloc[0]
                        ratio = (t - r['StartTime']) / r['Duration']
                        prog = (r['LapNumber'] - 1) + ratio
                    else:
                        past = d_laps[d_laps['EndTime'] < t]
                        prog = float(past['LapNumber'].max()) if not past.empty else 0.0
                
                drivers_in_frame.append({'TimeIndex': t, 'Driver': driver, 'Progress': prog, 'Team': team_map.get(driver, 'Unknown')})
            
            if drivers_in_frame:
                drivers_in_frame.sort(key=lambda x: x['Progress'], reverse=True)
                for rank, row in enumerate(drivers_in_frame, 1):
                    row['Rank'] = rank
                animation_rows.extend(drivers_in_frame)
        
        if not animation_rows: return None
        df_anim = pd.DataFrame(animation_rows)
        df_anim = df_anim[df_anim['Rank'] <= 20]
        color_map = {t: fastf1.plotting.get_team_color(t, session=session) for t in df_anim['Team'].unique()}
        
        fig = px.bar(
            df_anim, x="Progress", y="Rank", animation_frame="TimeIndex", animation_group="Driver",
            orientation='h', text="Driver", color="Team", color_discrete_map=color_map,
            range_x=[0, laps['LapNumber'].max() + 0.5], range_y=[20.5, 0.5], height=700
        )
        
        fig.update_layout(
            title="🏁 Race Replay",
            plot_bgcolor=F1_BG,
            paper_bgcolor=F1_BG,
            font=dict(color='white', family="Arial"),
            xaxis=dict(title="Laps", gridcolor='#333', showgrid=True),
            yaxis=dict(title="Position", autorange="reversed", showgrid=False),
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50
        fig.update_layout(sliders=[dict(visible=False)])
        return fig
    except Exception as e:
        return None
