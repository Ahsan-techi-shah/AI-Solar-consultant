import os

import pandas as pd
import streamlit as st

# Copy secrets into environment variables BEFORE the crew module is imported.
try:
    if 'GROQ_API_KEY' in st.secrets:
        os.environ['GROQ_API_KEY'] = str(st.secrets['GROQ_API_KEY'])
except Exception:
    pass

# Current Groq text model
os.environ['GROQ_TEXT_MODEL'] = "openai/gpt-oss-120b"

# Keep the vision model from Streamlit Secrets
try:
    if 'GROQ_VISION_MODEL' in st.secrets:
        os.environ['GROQ_VISION_MODEL'] = str(st.secrets['GROQ_VISION_MODEL'])
except Exception:
    pass

from crew.solar_crew import bill_agent, build_quote, run_solar_analysis
from utils.calculations import CITY_LATITUDES, SEASONS, apply_new_loads, consumption_stats

st.set_page_config(
    page_title='SolarAI Consultant',
    page_icon='☀️',
    layout='wide'
)

st.title('☀️ SolarAI — Multi-Agent Solar Consultant')
st.caption(
    'Bills → 12-month consumption → new loads → location & tilt → '
    'plant sizing → pricing → quotation'
)


def get_api_key():
    try:
        return st.secrets.get(
            'GROQ_API_KEY',
            os.getenv('GROQ_API_KEY', '')
        )
    except Exception:
        return os.getenv('GROQ_API_KEY', '')


def sizing_row(label, kwh, s):
    return {
        'Sizing basis': label,
        'Target kWh/month': round(kwh),
        'Plant (kW)': s['recommended_kw'],
        'Panels': s['panel_count'],
        'Installed (kWp)': s['installed_kwp']
    }


st.header('1. Customer & Electricity Bills')

name = st.text_input('Customer name')
phone = st.text_input('Customer phone (optional)')

bills = st.file_uploader(
    'Upload the electricity bill (PDF/JPG/PNG). One bill with a 12-month history table is enough; you can upload several.',
    type=['pdf', 'jpg', 'jpeg', 'png'],
    accept_multiple_files=True
)

if st.button('📄 Read bills', type='primary', disabled=not bills):

    api_key = get_api_key()

    if api_key:
        os.environ['GROQ_API_KEY'] = api_key

    with st.spinner('Reading bills...'):
        st.session_state['bill_data'] = bill_agent(
            bills,
            api_key
        )

    st.session_state['bill_version'] = (
        st.session_state.get('bill_version', 0) + 1
    )

    st.session_state.pop('result', None)
    st.session_state.pop('quote', None)


if 'bill_data' in st.session_state:

    bd = st.session_state['bill_data']
    version = st.session_state['bill_version']

    # ---------- 2. Monthly consumption ----------

    st.header('2. Monthly consumption — check and correct')

    if bd['files_read'] < bd['files_uploaded']:
        st.warning(
            f"Only {bd['files_read']} of {bd['files_uploaded']} files could be read. "
            f"{bd['note']}"
        )
    else:
        st.info(bd['note'])

    for _f in bd['files']:
        if _f.get('error'):
            st.error(f"{_f['file']}: {_f['error']}")

    st.caption(
        'Compare with the bill and fix any wrong value. '
        'You can add or delete rows (month format YYYY-MM).'
    )

    df = pd.DataFrame(
        bd['monthly'],
        columns=['month', 'units']
    )

    edited = st.data_editor(
        df,
        num_rows='dynamic',
        key=f'months_{version}',
        column_config={
            'month': st.column_config.TextColumn(
                'Month (YYYY-MM)'
            ),
            'units': st.column_config.NumberColumn(
                'Units (kWh)',
                min_value=0
            ),
        }
    )

    monthly = [
        {
            'month': str(r['month']).strip(),
            'units': float(r['units'])
        }
        for _, r in edited.iterrows()
        if pd.notna(r['month']) and pd.notna(r['units'])
    ]

    stats = consumption_stats(monthly)

    if stats['months_used']:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            'Average / month',
            f"{stats['average_monthly_kwh']:,.0f} kWh"
        )

        c2.metric(
            f"Maximum ({stats['max_month']})",
            f"{stats['max_kwh']:,.0f} kWh"
        )

        c3.metric(
            f"Minimum ({stats['min_month']})",
            f"{stats['min_kwh']:,.0f} kWh"
        )

        c4.metric(
            'Annual',
            f"{stats['annual_kwh']:,.0f} kWh"
        )

        if stats['months_used'] < 12:
            st.warning(
                f"Only {stats['months_used']} month(s) in the table. "
                "Add the missing months for a reliable average."
            )

    # ---------- 3. Location ----------

    st.header('3. Site location & panel orientation')

    cities = list(CITY_LATITUDES) + ['Other']

    default_city = (
        bd['city_hint']
        if bd['city_hint'] in CITY_LATITUDES
        else 'Peshawar'
    )

    city = st.selectbox(
        'Site city',
        cities,
        index=cities.index(default_city),
        key=f'city_{version}'
    )

    if bd['city_hint']:
        st.caption(
            f"City suggested from the bill: {bd['city_hint']}. "
            "Change it if the site is elsewhere."
        )

    if city in CITY_LATITUDES:
        latitude = CITY_LATITUDES[city]
    else:
        latitude = st.number_input(
            'Site latitude (°N)',
            5.0,
            40.0,
            30.0,
            0.1
        )

    # ---------- 4. New loads ----------

    st.header('4. Planned new loads (optional)')

    st.caption(
        'Anything the customer plans to add with the solar system, '
        'such as air conditioners. These raise the plant size. '
        'Example: Inverter AC 1.5 ton, Qty 1, 1500 W, 8 hours/day, '
        '30 days/month, Summer. Use "Duty %" for loads that do not draw '
        'full power all the time.'
    )

    loads_template = pd.DataFrame({
        'name': pd.Series(dtype='str'),
        'qty': pd.Series(dtype='float'),
        'watts': pd.Series(dtype='float'),
        'hours': pd.Series(dtype='float'),
        'days': pd.Series(dtype='float'),
        'duty': pd.Series(dtype='float'),
        'season': pd.Series(dtype='str'),
    })

    loads_edited = st.data_editor(
        loads_template,
        num_rows='dynamic',
        key=f'loads_{version}',
        column_config={
            'name': st.column_config.TextColumn(
                'Load (e.g. 1.5-ton inverter AC)'
            ),
            'qty': st.column_config.NumberColumn(
                'Qty',
                min_value=1,
                step=1,
                default=1
            ),
            'watts': st.column_config.NumberColumn(
                'Running watts (each)',
                min_value=0
            ),
            'hours': st.column_config.NumberColumn(
                'Hours/day',
                min_value=0,
                max_value=24
            ),
            'days': st.column_config.NumberColumn(
                'Days/month',
                min_value=1,
                max_value=31,
                default=30
            ),
            'duty': st.column_config.NumberColumn(
                'Duty %',
                min_value=1,
                max_value=100,
                default=100
            ),
            'season': st.column_config.SelectboxColumn(
                'Season',
                options=list(SEASONS),
                default='All year'
            ),
        }
    )

    new_loads = []

    for row in loads_edited.to_dict('records'):

        row = {
            k: (None if pd.isna(v) else v)
            for k, v in row.items()
        }

        if row.get('watts'):
            new_loads.append(row)

    adjusted, load_summary = apply_new_loads(
        monthly,
        new_loads
    )

    if load_summary['loads']:

        new_stats = consumption_stats(adjusted)

        st.table([
            {
                'Load': ld['name'],
                'Season': ld['season'],
                'kWh/month (when used)': ld['kwh_per_month'],
                'Connected (kW)': ld['connected_kw']
            }
            for ld in load_summary['loads']
        ])

        l1, l2, l3 = st.columns(3)

        l1.metric(
            'Added by new loads (avg)',
            f"{load_summary['avg_added_kwh_month']:,.0f} kWh/month"
        )

        l2.metric(
            'New average consumption',
            f"{new_stats['average_monthly_kwh']:,.0f} kWh/month"
        )

        l3.metric(
            'Added connected load',
            f"{load_summary['connected_kw']:.2f} kW"
        )

        st.caption(
            'Make sure the inverter can carry the added connected load, '
            'including air-conditioner start-up current.'
        )

    # ---------- 5. Preferences ----------

    st.header('5. Solar preferences')

    panel_brand = st.selectbox(
        'Preferred panel brand',
        ['Canadian Solar', 'LONGi', 'JA Solar', 'Other']
    )

    panel_watt = st.number_input(
        'Panel wattage (W)',
        min_value=300,
        max_value=1000,
        value=585,
        step=5
    )

    inverter_brand = st.selectbox(
        'Preferred inverter',
        ['Growatt', 'Solis', 'Huawei', 'GoodWe', 'Other']
    )

    with st.expander('Advanced sizing assumptions'):

        peak_sun_hours = st.number_input(
            'Peak sun hours/day',
            3.0,
            7.0,
            5.0,
            0.1
        )

        losses = st.slider(
            'System losses (%)',
            5,
            35,
            20
        )

    basis_label = st.radio(
        'Size the plant on',
        [
            'Annual average (net metering)',
            'Peak month (maximum units)'
        ]
    )

    sizing_basis = (
        'peak'
        if basis_label.startswith('Peak')
        else 'average'
    )

    if st.button(
        '🚀 Calculate system size & AI report',
        type='primary',
        disabled=not monthly
    ):

        with st.spinner(
            'Multi-agent analysis in progress...'
        ):

            st.session_state['result'] = run_solar_analysis(
                monthly=monthly,
                customer_name=name,
                city=city,
                latitude=latitude,
                panel_brand=panel_brand,
                panel_watt=panel_watt,
                inverter_brand=inverter_brand,
                peak_sun_hours=peak_sun_hours,
                losses=losses,
                sizing_basis=sizing_basis,
                new_loads=new_loads,
                groq_api_key=get_api_key()
            )

        st.session_state.pop('quote', None)


if 'result' in st.session_state:

    r = st.session_state['result']

    c = r['consumption']
    bc = r['base_consumption']
    o = r['orientation']
    opts = r['sizing_options']
    bopts = r['base_sizing_options']

    has_loads = bool(r['new_loads']['loads'])

    st.header('6. Analysis')

    m1, m2, m3 = st.columns(3)

    m1.metric(
        'Average monthly consumption' +
        (' (with new loads)' if has_loads else ''),
        f"{c['average_monthly_kwh']:,.0f} kWh"
    )

    m2.metric(
        'Annual consumption',
        f"{c['annual_kwh']:,.0f} kWh"
    )

    m3.metric(
        'Recommended plant',
        f"{r['sizing']['recommended_kw']:.1f} kW"
    )

    rows = [
        sizing_row(
            'Bills only — annual average',
            bc['average_monthly_kwh'],
            bopts['average']
        ),
        sizing_row(
            'Bills only — peak month',
            bc['max_kwh'],
            bopts['peak']
        )
    ]

    if has_loads:

        rows += [
            sizing_row(
                'With new loads — annual average',
                c['average_monthly_kwh'],
                opts['average']
            ),
            sizing_row(
                'With new loads — peak month',
                c['max_kwh'],
                opts['peak']
            )
        ]

    st.table(rows)

    tilt_note = (
        ' Tilt is 5° lower because consumption is summer-heavy.'
        if o['summer_biased']
        else ''
    )

    st.info(
        f"Orientation for {o['city']} "
        f"(latitude {o['latitude']}°N): face the panels true South "
        f"({o['azimuth_deg']}°), tilt about {o['tilt_deg']}°."
        f"{tilt_note}"
    )

    st.caption(
        'Rule-of-thumb tilt. Confirm with PVGIS / PVsyst for the exact '
        'roof, shading and local conditions.'
    )

    st.subheader('AI Report')
    st.write(r['ai_summary'])

    st.header('7. Company Pricing')

    st.info(
        'MVP: enter the approved current company price. '
        'The pricing agent is structured for WhatsApp/email integration '
        'in the next step.'
    )

    rate = st.number_input(
        'Approved system selling rate (PKR/W)',
        min_value=0.0,
        value=0.0,
        step=0.5
    )

    inverter_price = st.number_input(
        'Approved inverter price (PKR)',
        min_value=0.0,
        value=0.0,
        step=1000.0
    )

    installation = st.number_input(
        'Installation / BOS / other charges (PKR)',
        min_value=0.0,
        value=0.0,
        step=1000.0
    )

    if st.button('Generate Final Quotation'):
        st.session_state['quote'] = build_quote(
            r,
            rate,
            inverter_price,
            installation
        )


if 'quote' in st.session_state:

    q = st.session_state['quote']

    st.header('8. Customer Quotation')

    st.table(q['items'])

    st.success(
        f"Estimated total: PKR {q['total']:,.0f}"
    )

    st.download_button(
        'Download quotation data',
        q['text'],
        file_name='solar_quotation.txt'
    )
