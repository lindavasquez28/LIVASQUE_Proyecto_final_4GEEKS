import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Arreglos visuales ya que la pagina original se veia muy simple
st.markdown("""
    <style>
    /* Color fondo */
    .stApp {
        background-color: #f5f3ef !important;
    }
    
    /* Color sidebar */
    [data-testid="stSidebar"] {
        background-color: #eae6df !important;
    }
    
    /* Tamaño de la fuente */
    html, body, p, div, span, label, input, select {
        font-size: 22px !important;
    }
    
    /* Ajustar tamaños de títulos proporcionales y grandes */
    h1 { font-size: 46px !important; font-weight: bold !important; }
    h2 { font-size: 36px !important; font-weight: bold !important; }
    h3 { font-size: 30px !important; font-weight: bold !important; }
    
    /* Para que las etiquetas de los campos se vean grandes */
    [data-testid="stWidgetLabel"] p {
        font-size: 22px !important;
        font-weight: bold !important;
    }

    /* Boton de credito rojo */
    .stButton>button {
        background-color: #ff4b4b !important;
        color: white !important;
        font-size: 24px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 28px !important;
        width: 100% !important;
        transition: 0.3s;
    }
    /* Manita al pasar el mouse por encima del botón */
    .stButton>button:hover {
        background-color: #e03e3e !important;
        color: white !important;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Your Risk Engine - Riesgo Crediticio", page_icon="🏦")
st.title("YOUR RISK ENGINE")
st.write("¡Bienvenidx! Evalua el riesgo crediticio de tu cliente.")
st.write("Ingresa los datos de tu cliente en el menú de la izquierda.")

# Almacenamiento en caché y datos base
@st.cache_resource 
def cargar_recursos():
    clasificador = joblib.load('models/mejor_modelo_clasificador.pkl')
    regresor = joblib.load('models/mejor_modelo_regresor.pkl')
    datos_prueba = pd.read_csv('data/processed/X_prueba.csv')
    cliente_promedio = datos_prueba.median().to_frame().T
    return clasificador, regresor, cliente_promedio

try:
    modelo_clase, modelo_reg, cliente_base = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar los recursos: {e}")
    st.stop()

# Sidebards
st.sidebar.header("Perfil del cliente ↓")

# Ingresar el numero en vez de slider
monto_solicitado = st.sidebar.number_input("Monto Solicitado", min_value=5000.0, max_value=3000000.0, value=150000.0, step=5000.0)
ingresos = st.sidebar.number_input("Ingreso Total Anual", min_value=10000.0, max_value=500000.0, value=50000.0, step=1000.0)

# Regla del negocio: calculo una cuota anual realista (por ej: 8% del monto solicitado a varios años)
cuota_anual_estimada = monto_solicitado * 0.08 
# Regla del negocio: Capacidad de pago máxima (no se permite endeudar más del 30% del salario)
limite_capacidad_pago = ingresos * 0.30 

edad_anos = st.sidebar.number_input("Edad (Años)", min_value=18, max_value=100, value=35, step=1)
antiguedad_anos = st.sidebar.number_input("Antigüedad Laboral (Años)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
score_buro = st.sidebar.number_input("Score en Central de Riesgo (0=Pésimo, 1=Excelente)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

st.sidebar.markdown("---")
genero = st.sidebar.selectbox("Género", ["Femenino", "Masculino", "Otro"])
tiene_carro = st.sidebar.selectbox("¿Tiene vehículo propio?", ["No", "Sí"])
tiene_casa = st.sidebar.selectbox("¿Tiene propiedades/bienes raíces?", ["No", "Sí"])

# Perfil matemático
datos_usuario = cliente_base.copy()

if 'AMT_CREDIT' in datos_usuario.columns: datos_usuario['AMT_CREDIT'] = monto_solicitado
if 'AMT_GOODS_PRICE' in datos_usuario.columns: datos_usuario['AMT_GOODS_PRICE'] = monto_solicitado
if 'AMT_ANNUITY' in datos_usuario.columns: datos_usuario['AMT_ANNUITY'] = cuota_anual_estimada
if 'AMT_INCOME_TOTAL' in datos_usuario.columns: datos_usuario['AMT_INCOME_TOTAL'] = ingresos
if 'EDAD_ANOS' in datos_usuario.columns: datos_usuario['EDAD_ANOS'] = edad_anos
if 'ANTIGUEDAD_ANOS' in datos_usuario.columns: datos_usuario['ANTIGUEDAD_ANOS'] = antiguedad_anos

if 'CODE_GENDER_M' in datos_usuario.columns: datos_usuario['CODE_GENDER_M'] = 1 if genero == "Masculino" else 0
if 'CODE_GENDER_XNA' in datos_usuario.columns: datos_usuario['CODE_GENDER_XNA'] = 1 if genero == "Otro" else 0
if 'FLAG_OWN_CAR_Y' in datos_usuario.columns: datos_usuario['FLAG_OWN_CAR_Y'] = 1 if tiene_carro == "Sí" else 0
if 'FLAG_OWN_REALTY_Y' in datos_usuario.columns: datos_usuario['FLAG_OWN_REALTY_Y'] = 1 if tiene_casa == "Sí" else 0

for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
    if col in datos_usuario.columns: datos_usuario[col] = score_buro

# Predicción
if st.button("Evaluar Crédito"):
    st.subheader("Decisión del Sistema:")
    
    # Regla del negocio logicas que no se deben ignorar (no dar credito si tiene estas redflags - con opcion de flexiblizacion)
    es_muy_viejo = edad_anos >= 75 
    mal_score = score_buro <= 0.35  
    poca_antiguedad = antiguedad_anos < 1.0
    sin_respaldo = (tiene_carro == "No" and tiene_casa == "No")
    tiene_respaldo = (tiene_carro == "Sí" or tiene_casa == "Sí")
    
    # Se es mas flexible si el cliente tiene activos 
    porcentaje_endeudamiento = 0.45 if tiene_respaldo else 0.30
    limite_capacidad_pago = ingresos * porcentaje_endeudamiento
    
    # Redflags demográficas
    tiene_bandera_roja = (es_muy_viejo or mal_score or poca_antiguedad)
    
    # FILTRO 1: Capacidad de Pago Dinámica
    if cuota_anual_estimada > limite_capacidad_pago:
        st.error("RECHAZO AUTOMÁTICO POR CAPACIDAD DE PAGO.")
        st.write(f"**Motivo:** La cuota anual (\${cuota_anual_estimada:,.2f}) supera el límite permitido para su perfil (\${limite_capacidad_pago:,.2f}).")
        
    # FILTRO 2: Descarte (si tiene banderas de riesgo y no tiene bienes)
    elif tiene_bandera_roja and sin_respaldo:
        st.error("RECHAZO AUTOMÁTICO POR POLÍTICA DE RIESGO.")
        st.write("**Motivo:** El cliente presenta alertas de riesgo (edad, score o estabilidad) y no posee activos de respaldo (carro o casa).")

    # Si el cliente pasa las politicas, el modelo pasa a revisar:
    else:
        st.info("El cliente cumple con las políticas básicas de activos y capacidad de pago.")
        
        # Fase 1: Clasificación de Riesgo
        probabilidad_mora = modelo_clase.predict_proba(datos_usuario)[0][1]
        
        if probabilidad_mora > 0.5: 
            st.warning(f"**CRÉDITO DENEGADO**. El modelo de clasificación detectó un riesgo de impago del {probabilidad_mora*100:.1f}%.")
        else:
            st.success(f"**CRÉDITO APROBADO**. El modelo de clasificación detectó que el cliente es estadísticamente confiable (Riesgo: {probabilidad_mora*100:.1f}%).")
            
            # Fase 2: que el regresor sugiera el monto segun el perfil
            datos_para_regresor = datos_usuario.drop(columns=['AMT_CREDIT'])
            monto_sugerido = modelo_reg.predict(datos_para_regresor)[0]
            
            st.write(f"### **Monto Recomendado por el modelo de regresión:**")
            st.write(f"## $ {monto_sugerido:,.2f}")
            
            # Nota analítica para el usuario
            if monto_sugerido < monto_solicitado:
                st.caption(f"*Nota del sistema: Se recomienda un ajuste a la baja de $ {monto_solicitado - monto_sugerido:,.2f} sobre lo solicitado debido a sus condiciones de ingreso/edad.*")
            else:
                st.caption(f"*Nota del sistema: El cliente cuenta con un perfil sólido que soporta el monto solicitado o incluso una línea de crédito mayor.*")