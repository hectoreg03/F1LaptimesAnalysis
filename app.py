import streamlit as st
import fastf1
import fastf1.plotting
import os
import f1_analysis as f1

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="F1 Data Hub", 
    page_icon="🏁", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS F1 THEMATIC ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700&display=swap');

    .stApp {
        background-color: #15151e;
        color: white;
        font-family: 'Titillium Web', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #1b1b26;
        border-right: 2px solid #e10600;
    }
    h1 {
        font-family: 'Titillium Web', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        color: white;
        border-bottom: 4px solid #e10600;
        padding-bottom: 10px;
        letter-spacing: 1px;
    }
    h2, h3, h4 {
        font-family: 'Titillium Web', sans-serif;
        font-weight: 600;
        color: #f0f0f0;
        text-transform: uppercase;
    }
    div.stButton > button {
        background-color: #e10600;
        color: white;
        border: none;
        border-radius: 5px 15px 5px 15px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #ff1801;
        border: 1px solid white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2b2b3b;
        border-radius: 5px 5px 0 0;
        color: white;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e10600 !important;
        color: white !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Titillium Web', sans-serif;
        color: #e10600;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuración de FastF1
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache') 
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')

# --- HEADER PRINCIPAL ---
col_logo, col_title = st.columns([1, 6])
with col_title:
    st.title("F1 Telemetry Hub")
    st.caption("OFFICIAL DATA ANALYTICS DASHBOARD")

# --- BARRA LATERAL ---
st.sidebar.markdown("## ⚙️ RACE CONTROL")

year = st.sidebar.selectbox("Season", [2024, 2023, 2022, 2021], index=0)

@st.cache_data
def get_schedule(y):
    return fastf1.get_event_schedule(y, include_testing=False)

try:
    schedule = get_schedule(year)
    event_options = schedule['EventName'].tolist()
    default_ix = len(event_options) - 1 if len(event_options) > 0 else 0
    selected_event_name = st.sidebar.selectbox("Grand Prix", event_options, index=default_ix)
    
    session_map = {'R': 'Race', 'Q': 'Qualifying', 'S': 'Sprint', 'FP1': 'Practice 1', 'FP2': 'Practice 2'}
    session_key = st.sidebar.selectbox("Session", list(session_map.keys()), format_func=lambda x: session_map[x])

    load_btn = st.sidebar.button("INITIALIZE SESSION DATA", type="primary")

except Exception as e:
    st.error("Connection Error: Could not fetch season schedule.")
    st.stop()

# --- LÓGICA DE CARGA ---
if 'session' not in st.session_state:
    st.session_state['session'] = None

if load_btn:
    with st.spinner(f"📥 FETCHING TELEMETRY: {selected_event_name} {year}..."):
        try:
            session = fastf1.get_session(year, selected_event_name, session_key)
            session.load()
            st.session_state['session'] = session
            st.success("SYSTEM READY")
        except Exception as e:
            st.error(f"DATA LOAD ERROR: {e}")

# --- DASHBOARD ---
if st.session_state['session'] is not None:
    session = st.session_state['session']
    
    st.markdown(f"### 🏁 {session.event['EventName'].upper()} - {year}")
    
    drivers = session.drivers
    driver_list = [session.get_driver(d)['Abbreviation'] for d in drivers]
    
    st.sidebar.markdown("## 🏎️ DRIVER SELECT")
    selected_driver = st.sidebar.selectbox("Select Driver", driver_list, index=0)
    
    d_info = session.get_driver(selected_driver)
    st.sidebar.markdown(f"""
    <div style='background-color: #2b2b3b; padding: 10px; border-radius: 5px; border-left: 4px solid #{d_info.TeamColor if d_info.TeamColor else 'fff'};'>
        <h3 style='margin:0; color:white;'>{d_info.BroadcastName}</h3>
        <p style='margin:0; color:#aaa;'>#{d_info.TeamName}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- TABS (LAYOUT HORIZONTAL ÚNICO) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "TRACK DATA", 
        "LAP ANALYSIS", 
        "STRATEGY",
        "WEEKEND",
        "REPLAY"
    ])

    # TAB 1: MAPAS
    with tab1:
        st.markdown("#### SPEED TRAP VISUALIZATION")
        fig = f1.get_speed_map(session, selected_driver)
        if fig: st.pyplot(fig, use_container_width=True)
        
        st.divider() # Línea de separación limpia
        
        st.markdown("#### GEAR SHIFTS ANALYSIS")
        fig = f1.get_gear_map(session, selected_driver)
        if fig: st.pyplot(fig, use_container_width=True)

    # TAB 2: TIEMPOS
    with tab2:
        st.markdown("#### PACE DISTRIBUTION (TOP 10)")
        fig = f1.get_lap_distribution(session)
        if fig: st.pyplot(fig, use_container_width=True)
        
        st.divider()
        
        st.markdown(f"#### LAP EVOLUTION: {selected_driver}")
        fig = f1.get_driver_laptimes(session, selected_driver)
        if fig: st.pyplot(fig, use_container_width=True)

        if session_key == 'R':
            st.divider()
            st.markdown("#### TEAM RACE PACE")
            fig = f1.get_team_pace(session)
            if fig: st.pyplot(fig, use_container_width=True)

    # TAB 3: ESTRATEGIA
    with tab3:
        st.markdown("#### TYRE STRATEGY OVERVIEW")
        fig = f1.get_strategy_chart(session)
        if fig: st.pyplot(fig, use_container_width=True)
        
        st.divider()
        
        st.markdown(f"#### POSITION TRACKER: {selected_driver}")
        fig = f1.get_position_changes(session, selected_driver)
        if fig: st.pyplot(fig, use_container_width=True)

    # TAB 4: FIN DE SEMANA
    with tab4:
        st.info("Analyzing full weekend data (FP1-Race). This may take moment.")
        if st.button("LOAD WEEKEND OVERVIEW"):
            with st.spinner("Processing multi-session telemetry..."):
                fig = f1.get_driver_weekend_laptimes(session, selected_driver)
                if fig: st.pyplot(fig, use_container_width=True)

    # TAB 5: REPLAY
    with tab5:
        if session_key == 'R':
            st.markdown("#### RACE REPLAY")
            if st.button("LAUNCH REPLAY"):
                with st.spinner("Rendering animation..."):
                    fig = f1.get_race_replay(session)
                    if fig: st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Replay available for Race sessions only.")

else:
    st.markdown("""
    <div style='text-align: center; padding: 50px; opacity: 0.5;'>
        <h1>NO DATA LOADED</h1>
        <p>PLEASE SELECT A GRAND PRIX FROM RACE CONTROL</p>
    </div>
    """, unsafe_allow_html=True)
