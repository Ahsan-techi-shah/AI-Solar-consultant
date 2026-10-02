import base64
import io
import json
import os
import re

import pdfplumber
from PIL import Image, ImageOps

from utils.calculations import (MONTH_RE, apply_new_loads, consumption_stats, detect_city,
                                recommended_tilt, size_options)

try:
    from groq import Groq
except Exception:
    Groq = None

# Groq vision-capable model (reads bill images). Override with the GROQ_VISION_MODEL env var / Streamlit secret
# if Groq renames or retires it (check console.groq.com/docs/models).
VISION_MODEL = os.getenv('GROQ_VISION_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
TEXT_MODEL = os.getenv('GROQ_TEXT_MODEL', 'openai/gpt-oss-120b')  # text-only: used for the agents' reasoning/writing
CREW_MODEL = f'groq/{TEXT_MODEL}'  # CrewAI (LiteLLM) format: groq/<groq model id>
MAX_PDF_PAGES_FOR_VISION = 2
MAX_IMAGE_SIDE = 1600

VISION_PROMPT = (
    "This is a Pakistani electricity bill (DISCO such as PESCO, LESCO, IESCO, MEPCO, FESCO, K-Electric). "
    "Return ONLY a JSON object, no other text, with these keys:\n"
    '- "billing_month": month of THIS bill as YYYY-MM (for example "SEP 26" means "2026-09"), or null.\n'
    '- "units_kwh": units consumed in THIS bill (the UNITS value in the meter info box), or null. '
    "Never rupees and never meter readings.\n"
    '- "history": the BILL HISTORY table (usually the 12 previous months). One item per row: '
    '{"month": "YYYY-MM", "units": number}. A label like "Sep 25" means "2025-09". '
    "Use ONLY the UNITS column, never the bill or payment columns. Use an empty list if there is no table.\n"
    '- "city_text": the consumer\'s city or area as printed on the bill (no personal names), or null.\n'
    '- "disco": the electricity company name or abbreviation, or null.'
)


def extract_pdf_text(uploaded_file):
    data = uploaded_file.getvalue()
    if not uploaded_file.name.lower().endswith('.pdf'):
        return ''
    text = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or '')
    except Exception:
        pass
    return '\n'.join(text)


def find_kwh(text):
    patterns = [
        r'(?:units|unit|kwh|consumption|energy)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)',
        r'([0-9]+(?:\.[0-9]+)?)\s*(?:units|kwh)',
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return float(m.group(1))
    return None


def prepare_image(data):
    """Fix phone-photo rotation, shrink, and re-encode as JPEG so it fits Groq's request size limits."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert('RGB')
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def pdf_to_images(data, max_pages=MAX_PDF_PAGES_FOR_VISION):
    """Render the first pages of a (scanned) PDF to JPEG bytes."""
    images = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages[:max_pages]:
                pil = page.to_image(resolution=150).original
                buf = io.BytesIO()
                pil.convert('RGB').save(buf, format='JPEG', quality=85)
                images.append(buf.getvalue())
    except Exception:
        pass
    return images


def _valid_units(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if 0 < x < 100000 else None  # implausible for a single monthly bill otherwise


def _valid_month(m):
    return m if isinstance(m, str) and MONTH_RE.fullmatch(m) else None


def parse_bill_json(raw):
    """Pull the JSON object out of the model reply and validate every field."""
    m = re.search(r'\{.*\}', raw or '', re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    history = []
    for item in obj.get('history') or []:
        if isinstance(item, dict):
            month, units = _valid_month(item.get('month')), _valid_units(item.get('units'))
            if month and units is not None:
                history.append({'month': month, 'units': units})
    text = lambda v: v if isinstance(v, str) else None
    return {
        'billing_month': _valid_month(obj.get('billing_month')),
        'units_kwh': _valid_units(obj.get('units_kwh')),
        'history': history,
        'city_text': text(obj.get('city_text')),
        'disco': text(obj.get('disco')),
    }


def vision_extract(client, jpeg_bytes):
    b64 = base64.b64encode(jpeg_bytes).decode()
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': VISION_PROMPT},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            ],
        }],
        temperature=0,
        max_tokens=700,
    )
    return parse_bill_json(response.choices[0].message.content)


def read_bill_with_vision(client, images):
    """Read each image (page) and merge what was found."""
    merged = {'billing_month': None, 'units_kwh': None, 'history': [], 'city_text': None, 'disco': None}
    for img in images:
        r = vision_extract(client, img)
        if not r:
            continue
        for k in ('billing_month', 'units_kwh', 'city_text', 'disco'):
            merged[k] = merged[k] or r[k]
        merged['history'].extend(r['history'])
        if merged['units_kwh'] is not None and merged['history']:
            break
    return merged


def bill_agent(bills, api_key=''):
    """Read every uploaded bill: this month's units + the 12-month history table + a city hint."""
    client = Groq(api_key=api_key) if (api_key and Groq is not None) else None
    files, history_rows, current_rows = [], [], []
    city_text = disco = None
    skipped_no_key = 0

    for f in bills:
        name = f.name.lower()
        data = f.getvalue()
        is_pdf = name.endswith('.pdf')
        rec = {'file': f.name, 'billing_month': None, 'kwh': None, 'history_months': 0, 'method': None}
        parsed = None
        try:
            if client is not None and (is_pdf or name.endswith(IMAGE_EXTS)):
                images = pdf_to_images(data) if is_pdf else [prepare_image(data)]
                parsed = read_bill_with_vision(client, images)
                rec['method'] = 'vision'
            has_data = bool(parsed) and (parsed['units_kwh'] is not None or bool(parsed['history']))
            if is_pdf and not has_data:
                kwh = find_kwh(extract_pdf_text(f))  # text-PDF fallback: current month only
                if kwh is not None:
                    parsed = {'billing_month': None, 'units_kwh': kwh, 'history': [], 'city_text': None, 'disco': None}
                    rec['method'] = 'pdf_text'
        except Exception as e:
            rec['error'] = str(e)

        has_data = bool(parsed) and (parsed['units_kwh'] is not None or bool(parsed['history']))
        if has_data:
            rec.update(billing_month=parsed['billing_month'], kwh=parsed['units_kwh'], history_months=len(parsed['history']))
            history_rows += parsed['history']
            if parsed['units_kwh'] is not None:
                current_rows.append({'month': parsed['billing_month'] or f'unknown:{f.name}', 'units': parsed['units_kwh']})
            city_text = city_text or parsed['city_text']
            disco = disco or parsed['disco']
        elif client is None:
            skipped_no_key += 1
        files.append(rec)

    stats = consumption_stats(history_rows + current_rows)  # current-bill values override history values
    files_read = sum(1 for r in files if r['kwh'] is not None or r['history_months'])
    notes = []
    if skipped_no_key:
        notes.append(f'{skipped_no_key} bill(s) need the vision model but GROQ_API_KEY is missing.')
    if files_read < len(files):
        notes.append('Some files could not be read. Add the months manually in the table below.')
    if 0 < stats['months_used'] < 12:
        notes.append(f"Only {stats['months_used']} month(s) found. Add the missing months for an accurate 12-month average.")
    notes.append('Values read from photos are AI readings: compare them with the bill before continuing.')

    return {
        'files': files,
        'files_uploaded': len(files),
        'files_read': files_read,
        'monthly': stats['monthly'],
        'city_hint': detect_city(city_text) or detect_city(disco),
        'note': ' '.join(notes),
    }


def _facts(base_stats, stats, sizing, orientation, loads, panel_brand, inverter_brand):
    text = f'average monthly consumption from the bills {base_stats["average_monthly_kwh"]:.0f} kWh, '
    if loads['loads']:
        names = ', '.join(ld['name'] for ld in loads['loads'])
        text += (
            f'planned new loads ({names}) add about {loads["avg_added_kwh_month"]:.0f} kWh/month on average '
            f'(connected load {loads["connected_kw"]} kW), giving {stats["average_monthly_kwh"]:.0f} kWh/month in total, '
        )
    text += (
        f'highest month {stats["max_kwh"]:.0f} kWh ({stats["max_month"]}), lowest month {stats["min_kwh"]:.0f} kWh ({stats["min_month"]}), '
        f'recommended plant {sizing["recommended_kw"]} kW with {sizing["panel_count"]} panels, '
        f'site {orientation["city"]}, panels facing true South at about {orientation["tilt_deg"]} degrees tilt, '
        f'panel {panel_brand}, inverter {inverter_brand}'
    )
    return text


def llm_agent_summary(api_key, base_stats, stats, sizing, orientation, loads, panel_brand, inverter_brand):
    """Direct Groq SDK call (no CrewAI). Used as a fallback if CrewAI is unavailable."""
    if not api_key or Groq is None:
        return 'AI summary skipped: add GROQ_API_KEY in Streamlit Secrets.'
    try:
        client = Groq(api_key=api_key)
        prompt = (
            'You are a solar consultant. Write a concise customer-facing summary (under 150 words) using ONLY these '
            'verified numbers. Do not invent prices or engineering facts. '
            + _facts(base_stats, stats, sizing, orientation, loads, panel_brand, inverter_brand)
        )
        chat_completion = client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=TEXT_MODEL,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f'AI summary unavailable: {e}'


def run_crew_report(api_key, base_stats, stats, sizing, options, basis, orientation, loads, panel_brand, inverter_brand):
    """CrewAI multi-agent report: bill auditor -> sizing reviewer -> consultant (customer summary)."""
    if not api_key:
        return 'AI report skipped: add GROQ_API_KEY in Streamlit Secrets.'
    try:
        os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
        from crewai import Agent, Crew, LLM, Process, Task

        llm = LLM(model=CREW_MODEL, api_key=api_key, temperature=0.1, max_tokens=800)

        auditor = Agent(
            role='Electricity Bill Auditor',
            goal='Check the monthly kWh history for gaps, duplicates, misreads and the seasonal pattern.',
            backstory='You review electricity bill data for a solar company and only report what the data shows.',
            llm=llm, allow_delegation=False, verbose=False,
        )
        reviewer = Agent(
            role='PV Sizing Reviewer',
            goal='Sanity-check the calculated system size against the customer consumption and planned new loads. Never change the numbers.',
            backstory='You are a solar design engineer who checks that plant size, panel count and expected generation are consistent.',
            llm=llm, allow_delegation=False, verbose=False,
        )
        consultant = Agent(
            role='Solar Consultant',
            goal='Write a short, clear customer-facing summary using only the verified numbers provided.',
            backstory='You advise residential and commercial customers in Pakistan. You never invent prices or engineering facts.',
            llm=llm, allow_delegation=False, verbose=False,
        )

        audit_task = Task(
            description=(
                'Review this monthly consumption (kWh) taken from the customer bill history. Flag data-quality problems '
                '(missing months in the 12-month window, duplicates, implausible jumps that look like misreads) and '
                'describe the seasonal pattern. '
                f'Monthly data: {json.dumps(base_stats["monthly"])}. '
                f'Maximum: {base_stats["max_kwh"]} kWh in {base_stats["max_month"]}. '
                f'Minimum: {base_stats["min_kwh"]} kWh in {base_stats["min_month"]}.'
            ),
            expected_output='A short bullet list of data-quality issues and the seasonal pattern.',
            agent=auditor,
        )
        review_task = Task(
            description=(
                'Check that the sizing is consistent with the consumption and the planned new loads. Do NOT recalculate or change '
                'numbers; only flag concerns (for example peak-month sizing oversizing the plant for most of the year, unrealistic '
                'hours/day or season for a new load, or the inverter needing to carry the added connected load). '
                f'Average monthly consumption from bills: {base_stats["average_monthly_kwh"]} kWh. '
                f'Planned new loads: {json.dumps(loads)}. '
                f'Average including new loads: {stats["average_monthly_kwh"]} kWh. '
                f'Sizing options: {json.dumps(options)}. Chosen basis: {basis}. Orientation: {json.dumps(orientation)}.'
            ),
            expected_output='A short bullet list of concerns, or "Sizing looks consistent".',
            agent=reviewer,
            context=[audit_task],
        )
        summary_task = Task(
            description=(
                'Write a concise customer-facing summary (under 150 words). Use ONLY these numbers: '
                + _facts(base_stats, stats, sizing, orientation, loads, panel_brand, inverter_brand)
                + '. Mention any data-quality issues or concerns from the audit and the sizing review. Do not state prices.'
            ),
            expected_output='A plain-text summary for the customer.',
            agent=consultant,
            context=[audit_task, review_task],
        )

        crew = Crew(agents=[auditor, reviewer, consultant], tasks=[audit_task, review_task, summary_task],
                    process=Process.sequential, verbose=False)
        result = crew.kickoff()
        return getattr(result, 'raw', str(result))
    except Exception as e:
        # CrewAI missing or failed (e.g. dependency/build issue) -> fall back to a single direct Groq call
        fallback = llm_agent_summary(api_key, base_stats, stats, sizing, orientation, loads, panel_brand, inverter_brand)
        return f'{fallback}\n\n(CrewAI unavailable: {e})'


def run_solar_analysis(monthly, customer_name, city, latitude, panel_brand, panel_watt, inverter_brand,
                       peak_sun_hours, losses, sizing_basis='average', new_loads=None, groq_api_key=''):
    base_stats = consumption_stats(monthly)
    adjusted, loads = apply_new_loads(monthly, new_loads)
    stats = consumption_stats(adjusted)  # bills + planned new loads
    base_options = size_options(base_stats, peak_sun_hours, losses, panel_watt)
    options = size_options(stats, peak_sun_hours, losses, panel_watt)
    sizing = options['peak' if sizing_basis == 'peak' else 'average']
    orientation = {
        'city': city,
        'latitude': latitude,
        'azimuth_deg': 180,
        'tilt_deg': recommended_tilt(latitude, stats['summer_heavy']),
        'summer_biased': stats['summer_heavy'],
    }
    summary = run_crew_report(groq_api_key, base_stats, stats, sizing, options, sizing_basis, orientation, loads,
                              panel_brand, inverter_brand)
    return {
        'customer': {'name': customer_name},
        'consumption': stats,
        'base_consumption': base_stats,
        'new_loads': loads,
        'sizing': sizing,
        'sizing_options': options,
        'base_sizing_options': base_options,
        'sizing_basis': sizing_basis,
        'orientation': orientation,
        'selection': {'panel_brand': panel_brand, 'panel_watt': panel_watt, 'inverter_brand': inverter_brand},
        'ai_summary': summary,
    }


def build_quote(result, rate, inverter_price, installation):
    kw = result['sizing']['recommended_kw']
    panel_w = result['selection']['panel_watt']
    count = result['sizing']['panel_count']
    panel_cost = kw * 1000 * rate
    total = panel_cost + inverter_price + installation
    items = [
        {'Item': result['selection']['panel_brand'], 'Specification': f'{panel_w} W', 'Quantity': count, 'Amount PKR': round(panel_cost)},
        {'Item': result['selection']['inverter_brand'], 'Specification': 'Approved inverter', 'Quantity': 1, 'Amount PKR': round(inverter_price)},
        {'Item': 'Installation / BOS / other', 'Specification': '', 'Quantity': 1, 'Amount PKR': round(installation)},
    ]
    text = f"Customer: {result['customer']['name']}\nRecommended system: {kw:.1f} kW\nTotal: PKR {total:,.0f}\n"
    return {'items': items, 'total': total, 'text': text}
