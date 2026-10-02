import os
import streamlit as st
from crew.solar_crew import run_solar_analysis

st.set_page_config(page_title='SolarAI Consultant', page_icon='☀️', layout='wide')
st.title('☀️ SolarAI — Multi-Agent Solar Consultant')
st.caption('Electricity bills → 12-month consumption → load/solar sizing → panel/inverter → pricing → quotation')

st.header('1. Customer & Electricity Bills')
name = st.text_input('Customer name')
phone = st.text_input('Customer phone (optional)')
bills = st.file_uploader('Upload electricity bills (PDF/JPG/PNG). Upload up to 12 monthly bills.', type=['pdf','jpg','jpeg','png'], accept_multiple_files=True)

st.header('2. Solar Preferences')
panel_brand = st.selectbox('Preferred panel brand', ['Canadian Solar','LONGi','JA Solar','Other'])
panel_watt = st.number_input('Panel wattage (W)', min_value=300, max_value=1000, value=585, step=5)
inverter_brand = st.selectbox('Preferred inverter', ['Growatt','Solis','Huawei','GoodWe','Other'])

with st.expander('Advanced sizing assumptions'):
    peak_sun_hours = st.number_input('Peak sun hours/day', 3.0, 7.0, 5.0, 0.1)
    losses = st.slider('System losses (%)', 5, 35, 20)

if st.button('🚀 Analyze & Prepare Quotation', type='primary', disabled=not bills):
    api_key = st.secrets.get('GROQ_API_KEY', os.getenv('GROQ_API_KEY', ''))
    os.environ['GROQ_API_KEY'] = api_key
    with st.spinner('Multi-agent analysis in progress (reading bills, sizing, AI review)...'):
        result = run_solar_analysis(
            bills=bills,
            customer_name=name,
            panel_brand=panel_brand,
            panel_watt=panel_watt,
            inverter_brand=inverter_brand,
            peak_sun_hours=peak_sun_hours,
            losses=losses,
            groq_api_key=api_key,
        )
    st.session_state['result'] = result
    st.session_state.pop('quote', None)  # clear any old quotation when a new analysis runs

if 'result' in st.session_state:
    r = st.session_state['result']
    c = r['consumption']

    st.header('3. Analysis')
    if c['months_extracted'] < c['months_uploaded']:
        st.warning(f"Only {c['months_extracted']} of {c['months_uploaded']} bills were read. {c['note']}")
    else:
        st.info(c['note'])

    st.metric('Average Monthly Consumption', f"{c['average_monthly_kwh']:,.0f} kWh")
    st.metric('Annual Consumption', f"{c['annual_kwh']:,.0f} kWh")
    st.metric('Recommended Plant', f"{r['sizing']['recommended_kw']:.1f} kW")
    st.json(c)
    st.json(r['sizing'])

    st.subheader('AI Report')
    st.write(r['ai_summary'])

    st.header('4. Company Pricing')
    st.info('MVP: enter the approved current company price. The pricing agent is structured for WhatsApp/email integration in the next step.')
    rate = st.number_input('Approved system selling rate (PKR/W)', min_value=0.0, value=0.0, step=0.5)
    inverter_price = st.number_input('Approved inverter price (PKR)', min_value=0.0, value=0.0, step=1000.0)
    installation = st.number_input('Installation / BOS / other charges (PKR)', min_value=0.0, value=0.0, step=1000.0)

    if st.button('Generate Final Quotation'):
        from crew.solar_crew import build_quote
        quote = build_quote(r, rate, inverter_price, installation)
        st.session_state['quote'] = quote

if 'quote' in st.session_state:
    q = st.session_state['quote']
    st.header('5. Customer Quotation')
    st.table(q['items'])
    st.success(f"Estimated total: PKR {q['total']:,.0f}")
    st.download_button('Download quotation data', q['text'], file_name='solar_quotation.txt')
