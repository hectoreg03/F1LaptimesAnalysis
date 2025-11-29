import matplotlib.pyplot as plt
import seaborn as sns
import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.collections import LineCollection
from matplotlib import colormaps

# No configuramos el setup_mpl aquí, lo haremos en app.py para evitar conflictos

def get_speed_map(session, driver):
    """Genera el mapa de velocidad para un conductor específico."""
    try:
        lap = session.laps.pick_drivers(driver).pick_fastest()
        tel = lap.get_telemetry()

        x = np.array(tel['X'].values)
        y = np.array(tel['Y'].values)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        speed = tel['Speed']

        fig, ax = plt.subplots(figsize=(10, 6))
        # Estilos oscuros forzados por si fastf1 no los carga
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        cmap = plt.get_cmap('plasma')
        norm = plt.Normalize(speed.min(), speed.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm, linestyle='-', linewidth=5)
        lc.set_array(speed)
        line = ax.add_collection(lc)

        ax.plot(x, y, color='gray', linestyle='-', linewidth=12, zorder=0, alpha=0.3)

        cbar = plt.colorbar(line, ax=ax, orientation='vertical')
        cbar.set_label('Velocidad (km/h)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        ax.axis('off')
        ax.set_aspect('equal')
        return fig
    except Exception as e:
        print(f"Error en Speed Map: {e}")
        return None

def get_gear_map(session, driver):
    """Genera un mapa del circuito coloreado según la marcha (gear)."""
    try:
        lap = session.laps.pick_drivers(driver).pick_fastest()
        tel = lap.get_telemetry()

        x = np.array(tel['X'].values)
        y = np.array(tel['Y'].values)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        gear = tel['nGear'].to_numpy().astype(float)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        cmap = colormaps['Paired']
        lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
        lc_comp.set_array(gear)
        lc_comp.set_linewidth(4)
        ax.add_collection(lc_comp)
        
        ax.axis('equal')
        ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

        cbar = plt.colorbar(mappable=lc_comp, ax=ax, boundaries=np.arange(1, 10))
        cbar.set_ticks(np.arange(1.5, 9.5))
        cbar.set_ticklabels(np.arange(1, 9))
        cbar.set_label('Marcha (Gear)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        return fig
    except Exception as e:
        return None

def get_driver_laptimes(session, driver):
    """Scatterplot de tiempos de vuelta."""
    try:
        driver_laps = session.laps.pick_drivers(driver).reset_index()
        if driver_laps.empty: return None

        driver_laps['LapStatus'] = 'Ritmo Normal'
        if 'Rainfall' in driver_laps.columns:
            driver_laps.loc[driver_laps['Rainfall'] == True, 'LapStatus'] = 'Lluvia'
        driver_laps.loc[~pd.isnull(driver_laps['PitInTime']), 'LapStatus'] = 'Pitstop'
        driver_laps.loc[~pd.isnull(driver_laps['PitOutTime']), 'LapStatus'] = 'Pitstop'

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        sns.scatterplot(data=driver_laps, x="LapNumber", y="LapTime", ax=ax,
                        hue="Compound", palette=fastf1.plotting.get_compound_mapping(session=session),
                        style="LapStatus", markers={'Ritmo Normal': 'o', 'Pitstop': 'X', 'Lluvia': 'D'},
                        s=100, linewidth=0, legend='auto')

        ax.set_xlabel("Número de Vuelta")
        ax.set_ylabel("Tiempo de Vuelta")
        ax.invert_yaxis()
        ax.grid(color='gray', linestyle='--', linewidth=0.3, alpha=0.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, labelcolor='white')
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def get_lap_distribution(session):
    """Violin plot de distribución de tiempos (Top 10)."""
    try:
        point_finishers = session.drivers[:10]
        driver_laps = session.laps.pick_drivers(point_finishers).pick_quicklaps().reset_index()
        if driver_laps.empty: return None

        finishing_order = [session.get_driver(i)["Abbreviation"] for i in point_finishers]
        driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

        sns.violinplot(data=driver_laps, x="Driver", y="LapTime(s)", hue="Driver",
                       inner=None, density_norm="area", order=finishing_order,
                       palette=fastf1.plotting.get_driver_color_mapping(session=session),
                       ax=ax, linewidth=0.8)
        
        sns.swarmplot(data=driver_laps, x="Driver", y="LapTime(s)", order=finishing_order,
                      hue="Compound", palette=fastf1.plotting.get_compound_mapping(session=session),
                      hue_order=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
                      linewidth=0, size=4, ax=ax)
        
        ax.set_xlabel("Piloto")
        ax.set_ylabel("Tiempo (s)")
        sns.despine(left=True, bottom=True)
        try:
            ax.get_legend().remove()
        except:
            pass
        return fig
    except Exception as e:
        return None

def get_strategy_chart(session):
    """Gráfica de barras horizontales de estrategias."""
    try:
        laps = session.laps
        stints = laps[["Driver", "Stint", "Compound", "LapNumber"]].groupby(["Driver", "Stint", "Compound"]).count().reset_index()
        stints = stints.rename(columns={"LapNumber": "StintLength"})
        drivers_ordered = [session.get_driver(d)["Abbreviation"] for d in session.drivers]

        fig, ax = plt.subplots(figsize=(6, 10))
        # Ajuste para fondo transparente/negro si se desea, aunque esta funcion suele verse mejor simple
        
        for driver in drivers_ordered:
            driver_stints = stints.loc[stints["Driver"] == driver]
            previous_stint_end = 0
            for idx, row in driver_stints.iterrows():
                compound_color = fastf1.plotting.get_compound_color(row["Compound"], session=session)
                ax.barh(y=driver, width=row["StintLength"], left=previous_stint_end,
                           color=compound_color, edgecolor="black", fill=True)
                previous_stint_end += row["StintLength"]

        ax.set_xlabel("Vueltas")
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        return fig
    except:
        return None

def get_position_changes(session, highlight_driver):
    """Gráfica de líneas de cambios de posición."""
    try:
        fig, ax = plt.subplots(figsize=(6, 10))
        for drv in session.drivers:
            drv_laps = session.laps.pick_drivers(drv)
            if drv_laps.empty: continue
            abb = drv_laps['Driver'].iloc[0]

            if abb == highlight_driver:
                style = fastf1.plotting.get_driver_style(identifier=abb, style=['color', 'linestyle'], session=session)
                alpha, lw, zorder = 1.0, 3, 10
            else:
                style = fastf1.plotting.get_driver_style(identifier=abb, style=['color', 'linestyle'], session=session)
                alpha, lw, zorder = 0.3, 1, 1
            
            ax.plot(drv_laps['LapNumber'], drv_laps['Position'], alpha=alpha, linewidth=lw, zorder=zorder, **style)

        ax.set_ylim([20.5, 0.5])
        ax.set_yticks([1, 5, 10, 15, 20])
        ax.set_xlabel('Lap')
        ax.set_ylabel('Position')
        return fig
    except:
        return None

def get_team_pace(session):
    """Boxplot de ritmo por equipos."""
    try:
        laps = session.laps.pick_quicklaps()
        transformed_laps = laps.copy()
        transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()
        team_order = (transformed_laps[["Team", "LapTime (s)"]].groupby("Team").median()["LapTime (s)"].sort_values().index)
        team_palette = {team: fastf1.plotting.get_team_color(team, session=session) for team in team_order}

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=transformed_laps, x="Team", y="LapTime (s)",
                    hue="Team", order=team_order, palette=team_palette,
                    whiskerprops=dict(color="white"), boxprops=dict(edgecolor="white"),
                    medianprops=dict(color="grey"), capprops=dict(color="white"), ax=ax)
        
        ax.set_title("Ritmo de Carrera")
        ax.grid(visible=False)
        ax.set(xlabel=None)
        return fig
    except:
        return None

def get_driver_weekend_laptimes(current_session, driver):
    """Gráfica compleja que requiere cargar múltiples sesiones. CUIDADO con el rendimiento."""
    try:
        year = current_session.event.year
        event_name = current_session.event['EventName']
        session_identifiers = ['FP1', 'FP2', 'FP3', 'Q', 'R']
        all_laps_list = []

        # Esta función asume que se llamará desde un contexto donde el usuario sabe que tardará
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

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        sns.violinplot(data=weekend_df, x='Session', y='LapTimeSeconds',
                       inner=None, color='#1f77b4', linewidth=0, ax=ax, alpha=0.3, cut=0)
        sns.swarmplot(data=weekend_df, x='Session', y='LapTimeSeconds',
                      hue='Compound', palette=fastf1.plotting.get_compound_mapping(session=current_session),
                      size=3, ax=ax, alpha=0.9)
        
        ax.set_title(f"Resumen Fin de Semana: {driver}", color='white')
        try:
            ax.legend(title="Neumático", loc='upper right', frameon=False, labelcolor='white')
        except: pass
        return fig
    except Exception as e:
        return None

def get_race_replay(session):
    """Genera animación Plotly de Race Bars."""
    try:
        laps = session.laps.pick_drivers(session.drivers).reset_index()
        laps = laps.dropna(subset=['Time', 'LapNumber', 'LapTime'])
        laps['EndTime'] = laps['Time'].dt.total_seconds()
        laps['Duration'] = laps['LapTime'].dt.total_seconds()
        laps['StartTime'] = laps['EndTime'] - laps['Duration']
        laps = laps[laps['Duration'] > 0]
        
        start_t = laps['StartTime'].min()
        end_t = laps['EndTime'].max()
        n_frames = 60 # Reducido para mejor performance en web
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
            range_x=[0, laps['LapNumber'].max() + 0.5], range_y=[20.5, 0.5], height=600
        )
        fig.update_layout(
            title="🏁 Replay de Carrera", xaxis_title="Vueltas", yaxis_title="Posición",
            plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'),
            yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(gridcolor='#333'), showlegend=False
        )
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50
        fig.update_layout(sliders=[dict(visible=False)])
        return fig
    except Exception as e:
        return None
