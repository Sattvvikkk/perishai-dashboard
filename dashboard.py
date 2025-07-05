import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup
import io
import base64
import pydeck as pdk

st.set_page_config(page_title="PerishAI: Smart Routing Dashboard", layout="wide")

# --- Branding & Header ---
st.markdown("""
    <style>
        .main {background-color: #f7fafd;}
        .block-container {padding-top: 1rem;}
        .css-1d391kg {background-color: #f7fafd;}
    </style>
""", unsafe_allow_html=True)

st.title("PerishAI: Smart Shelf-Life Aware Routing Dashboard")
st.markdown("<h4 style='color:#2a6f97;'>Industry-Grade Perishable Goods Route Optimization</h4>", unsafe_allow_html=True)

# --- Standard shelf-life lookup table (static, not per user input) ---
STANDARD_SHELF_LIFE = {
    'apple': {'cold': 90, 'ambient': 30, 'frozen': 365},
    'banana': {'cold': 14, 'ambient': 7, 'frozen': 180},
    'milk': {'cold': 10, 'ambient': 1, 'frozen': 90},
    'cheese': {'cold': 60, 'ambient': 14, 'frozen': 365},
    'lettuce': {'cold': 10, 'ambient': 3, 'frozen': 0},
    'chicken': {'cold': 6, 'ambient': 1, 'frozen': 365},
    'yogurt': {'cold': 14, 'ambient': 2, 'frozen': 90}
}
STANDARD_RISK_FACTORS = {
    'apple': 'Ethylene exposure, bruising, high humidity',
    'banana': 'Chilling injury, ethylene, rapid ripening',
    'milk': 'Bacterial growth, temperature abuse',
    'cheese': 'Mold, temperature, humidity',
    'lettuce': 'Wilting, dehydration, ethylene',
    'chicken': 'Bacterial spoilage, temperature',
    'yogurt': 'Yeast/mold, temperature'
}

def get_standard_shelf_life(product, storage):
    return STANDARD_SHELF_LIFE.get(product, {}).get(storage, None)

def get_standard_risk_factors(product):
    return STANDARD_RISK_FACTORS.get(product, '')

# --- User Input for Prediction & Analytics ---
st.sidebar.header('Add Deliveries for Prediction & Analysis')
product_types = ['apple', 'banana', 'milk', 'cheese', 'lettuce', 'chicken', 'yogurt']
storage_conditions = ['cold', 'ambient', 'frozen', 'vacuum', 'controlled atmosphere', 'modified atmosphere', 'dry', 'humidified']

input_product = st.sidebar.selectbox('Product Type', product_types, help="Select the type of perishable product.")
input_storage = st.sidebar.selectbox('Storage Condition', storage_conditions, help="Select the storage condition for the product.")
input_time = st.sidebar.number_input('Time in Transit (hours)', min_value=1.0, max_value=48.0, value=1.0, step=0.5, help="Total hours in transit.")
input_temp = st.sidebar.number_input('Temperature Exposure (°C)', min_value=0.0, max_value=40.0, value=0.0, step=0.5, help="Average temperature during transit.")
input_humidity = st.sidebar.number_input('Humidity (%)', min_value=40.0, max_value=90.0, value=40.0, step=1.0, help="Average humidity during transit.")

# --- Session State for Multi-Item Input ---
if 'custom_items' not in st.session_state:
    st.session_state['custom_items'] = []
if 'predicted_items' not in st.session_state:
    st.session_state['predicted_items'] = None

# Add item to list
def add_item():
    std_shelf = get_standard_shelf_life(input_product, input_storage)
    std_risk = get_standard_risk_factors(input_product)
    st.session_state['custom_items'].append({
        'product_type': input_product,
        'storage_conditions': input_storage,
        'time_in_transit (hours)': input_time,
        'temperature_exposure (°C)': input_temp,
        'humidity (%)': input_humidity,
        'standard_shelf_life (days)': std_shelf,
        'standard_risk_factors': std_risk
    })

st.sidebar.button('Add Item', on_click=add_item)

# Show current items and allow removal
if st.session_state['custom_items']:
    st.sidebar.markdown('**Items to Predict:**')
    for idx, item in enumerate(st.session_state['custom_items']):
        st.sidebar.write(f"{idx+1}. {item['product_type']} | {item['storage_conditions']} | {item['time_in_transit (hours)']}h | {item['temperature_exposure (°C)']}°C | {item['humidity (%)']}%")
        if st.sidebar.button(f"Remove Item {idx+1}", key=f"remove_{idx}"):
            st.session_state['custom_items'].pop(idx)
            st.experimental_rerun()
else:
    st.sidebar.info('Add one or more items above.')

# Predict for all items in the list
if st.sidebar.button('Predict & Show Analytics') and st.session_state['custom_items']:
    encoder = joblib.load('encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('shelf_life_model.pkl')
    input_df = pd.DataFrame(st.session_state['custom_items'])
    # Compare user input time_in_transit to standard shelf life
    input_df['at_risk_std'] = input_df.apply(lambda row: row['time_in_transit (hours)']/24 > (row['standard_shelf_life (days)'] or 0), axis=1)
    # ML prediction fallback
    X_cat = input_df[['product_type', 'storage_conditions']]
    X_cont = input_df[['time_in_transit (hours)', 'temperature_exposure (°C)', 'humidity (%)']]
    X_cat_encoded = encoder.transform(X_cat)
    X_cont_scaled = scaler.transform(X_cont)
    X_prepared = np.hstack([X_cat_encoded, X_cont_scaled])
    ml_pred = model.predict(X_prepared)
    # Remove web_pred logic, use only ML prediction
    input_df['predicted_shelf_life (days)'] = ml_pred
    input_df['prediction_source'] = ['model'] * len(input_df)
    input_df['at_risk'] = input_df['predicted_shelf_life (days)'] < 2
    input_df['cold_chain_compliance (%)'] = [100 if (row['storage_conditions'] == 'cold' and row['predicted_shelf_life (days)'] >= 2) else 0 for _, row in input_df.iterrows()]
    # Example sustainability/business impact calculations
    input_df['waste_avoided'] = np.maximum(0, np.round((input_df['predicted_shelf_life (days)']/14)*10, 2))
    input_df['co2_saved'] = np.maximum(0, np.round((input_df['predicted_shelf_life (days)']/14)*6, 2))
    input_df['cost_savings'] = np.maximum(0, np.round((input_df['predicted_shelf_life (days)']/14)*1200, 2))
    input_df['customer_satisfaction'] = np.minimum(5.0, np.round(3.5 + (input_df['predicted_shelf_life (days)']/14)*1.5, 2))
    st.session_state['predicted_items'] = input_df

# --- Enhanced Custom CSS for Modern UI ---
st.markdown("""
    <style>
        body, .main, .block-container {
            background: linear-gradient(135deg, #f7fafd 0%, #e3f0ff 100%) !important;
        }
        .stButton>button {
            background: linear-gradient(90deg, #38b000 0%, #0081a7 100%) !important;
            color: #fff !important;
            border-radius: 10px !important;
            font-weight: bold !important;
            font-size: 1.1rem !important;
            box-shadow: 0 2px 12px rgba(56,176,0,0.18);
            transition: background 0.2s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #0081a7 0%, #38b000 100%) !important;
        }
        .stDownloadButton>button {
            background: linear-gradient(90deg, #ffb703 0%, #fb8500 100%) !important;
            color: #fff !important;
            border-radius: 10px !important;
            font-weight: bold !important;
            font-size: 1.1rem !important;
            box-shadow: 0 2px 12px rgba(251,133,0,0.18);
        }
        .stDownloadButton>button:hover {
            background: linear-gradient(90deg, #fb8500 0%, #ffb703 100%) !important;
        }
        .stDataFrame, .stTable {
            background: #fff !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 16px rgba(0,129,167,0.10);
        }
        .stMetric {
            background: linear-gradient(90deg, #e3f0ff 0%, #bde0fe 100%) !important;
            border-radius: 12px !important;
            padding: 0.5rem 1rem !important;
            font-size: 1.1rem !important;
            color: #023e8a !important;
        }
        .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
            border-radius: 10px !important;
            font-size: 1.08rem !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #023e8a !important;
            font-family: 'Segoe UI', 'Arial', sans-serif !important;
        }
        .sidebar .sidebar-content, .stSidebar {
            background: linear-gradient(135deg, #bde0fe 0%, #e3f0ff 100%) !important;
        }
        .stRadio > div {
            background: #fff3e0 !important;
            border-radius: 8px !important;
            padding: 0.5rem 0.5rem !important;
        }
        .stSlider > div {
            background: #e0f7fa !important;
            border-radius: 8px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Theme Toggle ---
theme = st.sidebar.radio('Theme', options=['Light', 'Dark'], index=0)
if theme == 'Dark':
    st.markdown("""
        <style>
        body, .main, .block-container {
            background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
            color: #f7fafd !important;
        }
        .stButton>button, .stDownloadButton>button {
            background: linear-gradient(90deg, #720026 0%, #ce4257 100%) !important;
            color: #fff !important;
        }
        .stDataFrame, .stTable {
            background: #232526 !important;
            color: #f7fafd !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffb703 !important;
        }
        .stMetric {
            background: linear-gradient(90deg, #232526 0%, #414345 100%) !important;
            color: #ffb703 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Dashboard Display ---
st.markdown('---')
st.markdown('<h2 style="color:#014f86;font-family:Segoe UI,Arial,sans-serif;">Analytics for Your Selected Items</h2>', unsafe_allow_html=True)
if st.session_state['predicted_items'] is not None:
    input_df = st.session_state['predicted_items']
    # Comparison Table: User Input vs Web Optimal Conditions
    comp_data = []
    for idx, row in input_df.iterrows():
        comp_data.append({
            'Product': row['product_type'],
            'Storage': row['storage_conditions'],
            'Your Transit (days)': round(row['time_in_transit (hours)']/24,2),
            'Your Temp (°C)': row['temperature_exposure (°C)'],
            'Your Humidity (%)': row['humidity (%)'],
            'Standard Shelf Life (days)': row['standard_shelf_life (days)'] if row['standard_shelf_life (days)'] else 'N/A',
            'Predicted Shelf Life (days)': round(row['predicted_shelf_life (days)'],2),
            'Risk Factors': row['standard_risk_factors'],
            'Alert': 'At Risk' if row['at_risk_std'] else ''
        })
    comp_df = pd.DataFrame(comp_data)
    st.subheader('Comparison: Your Inputs vs Standard Shelf Life')
    st.dataframe(comp_df, use_container_width=True)
    # Show risk factors and alerts
    for idx, row in comp_df.iterrows():
        if row['Alert']:
            st.error(f"{row['Product']} ({row['Storage']}): ALERT! Transit time exceeds standard shelf life. {row['Risk Factors'] if row['Risk Factors'] else ''}")
        elif row['Risk Factors']:
            st.warning(f"{row['Product']} ({row['Storage']}): {row['Risk Factors']}")
    # KPIs and Graphs based on predicted data
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg. Predicted Shelf Life (days)", f"{input_df['predicted_shelf_life (days)'].mean():.2f}")
    k2.metric("At-Risk Deliveries", int(input_df['at_risk_std'].sum()))
    k3.metric("Avg. Standard Shelf Life (days)", f"{input_df['standard_shelf_life (days)'].mean() if input_df['standard_shelf_life (days)'].notnull().any() else 0:.2f}")
    k4.metric("Total Deliveries", len(input_df))

    # --- Enhanced Charts ---
    st.markdown('---')
    st.subheader('Charts & Visualizations')
    chart1, chart2 = st.columns(2)
    # Bar chart: Predicted shelf life by product
    with chart1:
        fig1 = px.bar(input_df, x='product_type', y='predicted_shelf_life (days)', color='storage_conditions',
                     title='Predicted Shelf Life by Product',
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig1, use_container_width=True)
    # Pie chart: Distribution of storage conditions
    with chart2:
        fig2 = px.pie(input_df, names='storage_conditions', title='Storage Condition Distribution',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)
    # Line chart: Shelf life vs. time in transit
    st.markdown('---')
    st.subheader('Shelf Life vs. Time in Transit')
    fig4 = px.line(input_df, x='time_in_transit (hours)', y='predicted_shelf_life (days)', color='product_type',
                  markers=True, title='Shelf Life vs. Time in Transit',
                  color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig4, use_container_width=True)
    # Heatmap: Product vs. Storage Condition (mean shelf life)
    st.markdown('---')
    st.subheader('Heatmap: Product & Storage vs. Shelf Life')
    heatmap_df = input_df.groupby(['product_type', 'storage_conditions'])['predicted_shelf_life (days)'].mean().reset_index()
    heatmap_pivot = heatmap_df.pivot(index='product_type', columns='storage_conditions', values='predicted_shelf_life (days)')
    fig5 = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Blues',
        colorbar=dict(title='Shelf Life (days)')
    ))
    fig5.update_layout(title='Mean Predicted Shelf Life by Product & Storage', xaxis_title='Storage Condition', yaxis_title='Product')
    st.plotly_chart(fig5, use_container_width=True)
    # Histogram: Freshness Distribution
    st.markdown('---')
    st.subheader("Freshness Distribution")
    fig3 = px.histogram(
        input_df,
        x='predicted_shelf_life (days)',
        nbins=20,
        color='product_type',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title='Distribution of Predicted Shelf Life'
    )
    st.plotly_chart(fig3, use_container_width=True)
    # --- At-Risk Deliveries Table ---
    st.subheader("At-Risk Deliveries (Shelf Life < 2 days)")
    at_risk = input_df[input_df['at_risk']]
    if not at_risk.empty:
        st.dataframe(at_risk)
    else:
        st.success("No at-risk deliveries!")
    # --- Sustainability & Business Impact ---
    st.subheader("Sustainability & Business Impact")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Waste Avoided", f"{input_df['waste_avoided'].sum()} kg")
    s2.metric("Total CO₂ Saved", f"{input_df['co2_saved'].sum()} kg CO₂")
    s3.metric("Total Cost Savings", f"${input_df['cost_savings'].sum()}")
    s4.metric("Avg. Customer Satisfaction", f"{input_df['customer_satisfaction'].mean():.2f}/5")
    # --- Export Data ---
    st.download_button(
        label="Export Predicted Data as CSV",
        data=input_df.to_csv(index=False),
        file_name="predicted_delivery_plan.csv",
        mime="text/csv"
    )

    # --- Animated Chart: Shelf Life Decay Over Time ---
    st.markdown('---')
    st.markdown('<h2 style="color:#1e90ff;font-family:Segoe UI,Arial,sans-serif;background:linear-gradient(90deg,#f7971e,#ffd200,#21d4fd,#b721ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Animated Shelf Life Decay by Product</h2>', unsafe_allow_html=True)
    st.markdown('<div style="background:linear-gradient(135deg,#f7971e 0%,#ffd200 40%,#21d4fd 80%,#b721ff 100%);border-radius:18px;padding:1.5rem 1rem 1rem 1rem;margin-bottom:2rem;box-shadow:0 4px 24px rgba(33,212,253,0.10);">', unsafe_allow_html=True)
    # Simulate shelf life decay for animation
    anim_rows = []
    for idx, row in input_df.iterrows():
        max_days = int(np.ceil(row['predicted_shelf_life (days)']))
        for t in range(max_days+1):
            anim_rows.append({
                'Product': row['product_type'],
                'Storage': row['storage_conditions'],
                'Day': t,
                'Shelf Life Left': max(row['predicted_shelf_life (days)']-t, 0)
            })
    anim_df = pd.DataFrame(anim_rows)
    # Use a vibrant color palette
    vibrant_palette = ['#f7971e','#ffd200','#21d4fd','#b721ff','#ff0844','#ffb199','#43cea2','#185a9d']
    fig_anim = px.line(
        anim_df,
        x='Day',
        y='Shelf Life Left',
        color='Product',
        animation_frame='Day',
        range_y=[0, anim_df['Shelf Life Left'].max()+1],
        title='Shelf Life Decay Animation',
        color_discrete_sequence=vibrant_palette
    )
    fig_anim.update_layout(
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(255,255,255,0.0)',
        font=dict(family='Segoe UI,Arial,sans-serif',size=16,color='#1e293b'),
        title_font=dict(size=24, color='#b721ff'),
        legend=dict(bgcolor='rgba(255,255,255,0.7)',bordercolor='#b721ff',borderwidth=1),
        margin=dict(l=20,r=20,t=60,b=20)
    )
    st.plotly_chart(fig_anim, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Interactive Map: Delivery Points ---
    st.markdown('---')
    st.subheader('Delivery Locations Map')
    # Simulate random lat/lon for demo (replace with real data if available)
    np.random.seed(42)
    input_df = st.session_state['predicted_items']
    if 'lat' not in input_df.columns or 'lon' not in input_df.columns:
        input_df['lat'] = 28.6 + np.random.uniform(-0.1, 0.1, len(input_df))
        input_df['lon'] = 77.2 + np.random.uniform(-0.1, 0.1, len(input_df))
    st.map(input_df[['lat', 'lon']])

    # --- Pydeck Geospatial Visualization ---
    st.subheader('Geospatial Risk Visualization')
    # Assign color column for pydeck (red for at-risk, blue otherwise)
    input_df['color'] = input_df['at_risk'].apply(lambda x: [255,0,0] if x else [0,128,255])
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=input_df,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=200,
        pickable=True
    )
    view_state = pdk.ViewState(
        latitude=input_df['lat'].mean(),
        longitude=input_df['lon'].mean(),
        zoom=10,
        pitch=0
    )
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{product_type} ({storage_conditions})\nRisk: {at_risk}"})
    st.pydeck_chart(r)

    # --- Downloadable Reports: Excel & PDF ---
    st.markdown('---')
    st.subheader('Downloadable Reports')
    input_df = st.session_state['predicted_items']
    # Excel
    excel_buffer = io.BytesIO()
    input_df.to_excel(excel_buffer, index=False)
    st.download_button(
        label='Download as Excel',
        data=excel_buffer.getvalue(),
        file_name='predicted_delivery_plan.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    # PDF (simple HTML to PDF)
    try:
        import pdfkit
        html = input_df.to_html(index=False)
        pdf_buffer = pdfkit.from_string(html, False)
        b64 = base64.b64encode(pdf_buffer).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="predicted_delivery_plan.pdf">Download as PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.info('PDF export requires pdfkit and wkhtmltopdf installed.')

# --- End of Dashboard ---
st.info("This dashboard helps optimize delivery routes for your selected perishable goods, reducing waste, improving sustainability, and providing actionable business insights.")
