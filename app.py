import streamlit as st
import fastf1
import fastf1.plotting
import os
import f1_analysis as f1  # <--- IMPORTAMOS TU ARCHIVO DE LÓGICA AQUÍ

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="F1 Telemetry Pro", page_icon="🏎️", layout="wide")

# Configuración de FastF1 y Matplotlib
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache') 
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')

# Título
st.title("🏎️ F1 Telemetry Pro Dashboard")
st.markdown("Análisis avanzado de telemetría y estrategias de Fórmula 1.")

# --- BARRA LATERAL INTELIGENTE ---
st.sidebar.header("1. Configuración de Carrera")

# A. Selección de Año
year = st.sidebar.selectbox("Año", [2024, 2023, 2022, 2021, 2020], index=1)

# B. Carga dinámica del Calendario (Cacheada)
@st.cache_data
def get_schedule(y):
    return fastf1.get_event_schedule(y, include_testing=False)

try:
    schedule = get_schedule(year)
    # Crear lista de eventos: "Round 1: Bahrain Grand Prix"
    event_options = schedule['EventName'].tolist()
    selected_event_name = st.sidebar.selectbox("Gran Premio", event_options)
    
    # C. Tipo de Sesión
    session_map = {'R': 'Carrera', 'Q': 'Clasificación', 'S': 'Sprint', 'FP1': 'Práctica 1', 'FP2': 'Práctica 2'}
    session_key = st.sidebar.selectbox("Sesión", list(session_map.keys()), format_func=lambda x: session_map[x])

    # Botón de cargar sesión
    load_btn = st.sidebar.button("⬇️ Cargar Datos de Sesión", type="primary")

except Exception as e:
    st.error("Error cargando el calendario. Revisa tu conexión.")
    st.stop()


# --- LÓGICA DE CARGA DE DATOS ---
if 'session' not in st.session_state:
    st.session_state['session'] = None

if load_btn:
    with st.spinner(f"Descargando datos de {selected_event_name} {year}..."):
        try:
            session = fastf1.get_session(year, selected_event_name, session_key)
            session.load()
            st.session_state['session'] = session
            st.success(f"✅ Datos cargados: {session.event['EventName']}")
        except Exception as e:
            st.error(f"Error cargando sesión: {e}")

# --- CONTENIDO PRINCIPAL (SOLO SI HAY SESIÓN CARGADA) ---
if st.session_state['session'] is not None:
    session = st.session_state['session']
    
    # Selector de Piloto (Dinámico según la sesión cargada)
    drivers = session.drivers
    driver_list = [session.get_driver(d)['Abbreviation'] for d in drivers]
    
    st.sidebar.header("2. Configuración de Piloto")
    selected_driver = st.sidebar.selectbox("Piloto a Analizar", driver_list, index=0)

    # --- PESTAÑAS DE ANÁLISIS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Telemetría y Pista", 
        "⏱️ Tiempos", 
        "🧠 Estrategia",
        "📅 Fin de Semana",
        "🎬 Replay"
    ])

    # TAB 1: MAPAS
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Mapa de Velocidad - {selected_driver}")
            fig = f1.get_speed_map(session, selected_driver)
            if fig: st.pyplot(fig)
        
        with col2:
            st.subheader(f"Mapa de Marchas - {selected_driver}")
            fig = f1.get_gear_map(session, selected_driver)
            if fig: st.pyplot(fig)

    # TAB 2: TIEMPOS
    with tab2:
        st.subheader("Distribución de Tiempos (Top 10)")
        fig = f1.get_lap_distribution(session)
        if fig: st.pyplot(fig)
        
        st.divider()
        st.subheader(f"Ritmo de Vuelta: {selected_driver}")
        fig = f1.get_driver_laptimes(session, selected_driver)
        if fig: st.pyplot(fig)

        if session_key == 'R':
            st.divider()
            st.subheader("Ritmo de Equipos (Race Pace)")
            fig = f1.get_team_pace(session)
            if fig: st.pyplot(fig)

    # TAB 3: ESTRATEGIA
    with tab3:
        colA, colB = st.columns([1, 1])
        with colA:
            st.subheader("Uso de Neumáticos")
            fig = f1.get_strategy_chart(session)
            if fig: st.pyplot(fig)
        
        with colB:
            st.subheader(f"Posiciones en Carrera ({selected_driver})")
            fig = f1.get_position_changes(session, selected_driver)
            if fig: st.pyplot(fig)

    # TAB 4: FIN DE SEMANA
    with tab4:
        st.info("Este gráfico carga datos de FP1, FP2, FP3, Q y R. Puede tardar unos segundos.")
        if st.button("Analizar Fin de Semana Completo"):
            with st.spinner("Procesando todas las sesiones..."):
                fig = f1.get_driver_weekend_laptimes(session, selected_driver)
                if fig: st.pyplot(fig)
                else: st.warning("No se encontraron datos completos del fin de semana.")

    # TAB 5: REPLAY
    with tab5:
        if session_key == 'R':
            st.subheader("Replay de Carrera")
            if st.button("Generar Animación"):
                with st.spinner("Renderizando animación..."):
                    fig = f1.get_race_replay(session)
                    if fig: st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("El Replay solo está disponible para carreras.")

else:
    # Mensaje de bienvenida cuando no hay datos
    st.info("👈 Usa el menú lateral para seleccionar una carrera y cargar los datos.")
