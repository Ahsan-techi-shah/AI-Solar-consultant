import base64
import io
import json
import os
import re
from statistics import mean

import pdfplumber
from PIL import Image, ImageOps

from utils.calculations import size_solar_system

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
    "This is an electricity bill (for example from a DISCO such as PESCO, LESCO, IESCO or K-Electric). "
    "Extract two values:\n"
    "- billing_month: the billing month of THIS bill as YYYY-MM, or null if not visible.\n"
    "- units_kwh: the units (kWh) consumed in THIS billing month. Do NOT return the amount in rupees, "
    "the meter reading, or any value from the previous-months history table. Use null if not visible.\n"
    'Respond with ONLY a JSON object, no other text: {"billing_month": "YYYY-MM" or null, "units_kwh": number or null}'
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


def parse_vision_json(raw):
    """Pull the JSON object out of the model reply and validate it."""
    m = re.search(r'\{.*\}', raw or '', re.S)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, None
    units = obj.get('units_kwh')
    try:
        units = float(units)
    except (TypeError, ValueError):
        units = None
    if units is not None and not (0 < units < 100000):
        units = None  # implausible reading for a single monthly bill
    month = obj.get('billing_month')
    month = month if isinstance(month, str) and re.fullmatch(r'\d{4}-\d{2}', month) else None
    return units, month


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
        max_tokens=200,
    )
    return parse_vision_json(response.choices[0].message.content)


def read_bill_with_vision(client, images):
    """Try each image until one yields a units value."""
    for img in images:
        units, month = vision_extract(client, img)
        if units is not None:
            return units, month
    return None, None


def bill_agent(bills, api_key=''):
    client = Groq(api_key=api_key) if (api_key and Groq is not None) else None
    records = []
    skipped_no_key = 0

    for f in bills:
        name = f.name.lower()
        data = f.getvalue()
        rec = {'file': f.name, 'kwh': None, 'billing_month': None, 'method': None, 'text_found': False}

        try:
            if name.endswith('.pdf'):
                text = extract_pdf_text(f)
                rec['text_found'] = bool(text.strip())
                kwh = find_kwh(text)
                if kwh is not None:
                    rec.update(kwh=kwh, method='pdf_text')
                elif client is not None:
                    # Scanned PDF (no text layer) or regex miss -> read the rendered pages with vision
                    kwh, month = read_bill_with_vision(client, pdf_to_images(data))
                    if kwh is not None:
                        rec.update(kwh=kwh, billing_month=month, method='vision_pdf')
                else:
                    skipped_no_key += 1
            elif name.endswith(IMAGE_EXTS):
                if client is None:
                    skipped_no_key += 1
                else:
                    kwh, month = read_bill_with_vision(client, [prepare_image(data)])
                    if kwh is not None:
                        rec.update(kwh=kwh, billing_month=month, method='vision_image')
        except Exception as e:
            rec['error'] = str(e)

        records.append(rec)

    values = [x['kwh'] for x in records if x['kwh'] is not None]
    notes = []
    if skipped_no_key:
        notes.append(f'{skipped_no_key} bill(s) need the vision model but GROQ_API_KEY is missing.')
    if len(values) < len(records):
        notes.append('Some bills could not be read. Check values manually or upload a clearer copy.')
    notes.append('Vision-extracted values are AI readings: verify them against the bills before quoting.')

    return {
        'months_uploaded': len(records),
        'months_extracted': len(values),
        'monthly_records': records,
        'average_monthly_kwh': mean(values) if values else 0,
        'annual_kwh': sum(values) if values else 0,
        'note': ' '.join(notes),
    }


def llm_agent_summary(api_key, consumption, sizing, panel_brand, inverter_brand):
    """Direct Groq SDK call (no CrewAI). Used as a fallback if CrewAI is unavailable."""
    if not api_key or Groq is None:
        return 'AI summary skipped: add GROQ_API_KEY in Streamlit Secrets.'
    try:
        client = Groq(api_key=api_key)
        prompt = (
            'You are a solar consultant. Write a concise customer-facing summary (under 150 words) using ONLY these '
            f'verified numbers. Do not invent prices or engineering facts. Consumption: {consumption["average_monthly_kwh"]:.0f} kWh/month. '
            f'Recommended plant: {sizing["recommended_kw"]} kW, {sizing["panel_count"]} panels. Panel: {panel_brand}. Inverter: {inverter_brand}.'
        )
        chat_completion = client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=TEXT_MODEL,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f'AI summary unavailable: {e}'


def run_crew_report(api_key, consumption, sizing, panel_brand, inverter_brand):
    """CrewAI multi-agent report: bill auditor -> sizing reviewer -> consultant (customer summary)."""
    if not api_key:
        return 'AI report skipped: add GROQ_API_KEY in Streamlit Secrets.'
    try:
        os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
        from crewai import Agent, Crew, LLM, Process, Task

        llm = LLM(model=CREW_MODEL, api_key=api_key, temperature=0.1, max_tokens=800)

        auditor = Agent(
            role='Electricity Bill Auditor',
            goal='Check extracted monthly kWh values for gaps, duplicates and outliers.',
            backstory='You review electricity bill data for a solar company and only report what the data shows.',
            llm=llm, allow_delegation=False, verbose=False,
        )
        reviewer = Agent(
            role='PV Sizing Reviewer',
            goal='Sanity-check the calculated system size against the customer consumption. Never change the numbers.',
            backstory='You are a solar design engineer who checks that panel count, plant size and expected generation are consistent.',
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
                'Review these extracted bill records and list any data-quality problems '
                '(missing months, duplicate months, values far from the rest, bills not read). '
                f'Records: {json.dumps(consumption["monthly_records"])}'
            ),
            expected_output='A short bullet list of data-quality issues, or "No issues found".',
            agent=auditor,
        )
        review_task = Task(
            description=(
                'Check that these calculated values are consistent with each other and with the consumption. '
                'Do NOT recalculate or change them; only flag concerns. '
                f'Average monthly consumption: {consumption["average_monthly_kwh"]:.0f} kWh. Sizing: {json.dumps(sizing)}.'
            ),
            expected_output='A short bullet list of concerns, or "Sizing looks consistent".',
            agent=reviewer,
            context=[audit_task],
        )
        summary_task = Task(
            description=(
                'Write a concise customer-facing summary (under 150 words). Use ONLY these numbers: '
                f'average monthly consumption {consumption["average_monthly_kwh"]:.0f} kWh, '
                f'recommended plant {sizing["recommended_kw"]} kW, {sizing["panel_count"]} panels, '
                f'panel {panel_brand}, inverter {inverter_brand}. '
                'Mention any data-quality issues or concerns from the audit and the sizing review. Do not state prices.'
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
        fallback = llm_agent_summary(api_key, consumption, sizing, panel_brand, inverter_brand)
        return f'{fallback}\n\n(CrewAI unavailable: {e})'


def run_solar_analysis(bills, customer_name, panel_brand, panel_watt, inverter_brand, peak_sun_hours, losses, groq_api_key=''):
    consumption = bill_agent(bills, groq_api_key)
    sizing = size_solar_system(consumption['average_monthly_kwh'], peak_sun_hours, losses, panel_watt)
    summary = run_crew_report(groq_api_key, consumption, sizing, panel_brand, inverter_brand)
    return {
        'customer': {'name': customer_name},
        'consumption': consumption,
        'sizing': sizing,
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
