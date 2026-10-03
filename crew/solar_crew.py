import base64
import io
import json
import os
import re

import pdfplumber
from PIL import Image, ImageOps

from utils.calculations import (
    MONTH_RE,
    apply_new_loads,
    consumption_stats,
    detect_city,
    recommended_tilt,
    size_options,
)

try:
    from groq import Groq
except Exception:
    Groq = None


# =========================================================
# GROQ MODELS
# =========================================================

# Current Groq multimodal model for reading electricity-bill photos.
# It supports text + image input and JSON mode.
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

# Do NOT use the retired Llama 4 Scout/Maverick IDs.
VISION_FALLBACK_MODELS = []

# Text/reasoning model used by CrewAI.
TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")

# CrewAI / LiteLLM format.
CREW_MODEL = TEXT_MODEL


# =========================================================
# FILE / IMAGE SETTINGS
# =========================================================

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

MAX_PDF_PAGES_FOR_VISION = 2
MAX_IMAGE_SIDE = 2400
MAX_IMAGE_BYTES = 2_800_000


# =========================================================
# BILL VISION PROMPT
# =========================================================

VISION_PROMPT = (
    "This is a Pakistani electricity bill (DISCO such as PESCO, LESCO, IESCO, MEPCO, FESCO, K-Electric). "
    "Read the bill carefully and return ONLY a JSON object with these keys:\n"
    '- "billing_month": month of THIS bill as YYYY-MM. Example: "SEP 26" = "2026-09". Use null if unclear.\n'
    '- "units_kwh": units consumed in THIS bill. This is the UNITS value in the meter information box. '
    "Never return rupees and never return the previous/present meter reading.\n"
    '- "history": the BILL HISTORY table. One item per row: '
    '{"month": "YYYY-MM", "units": number}. '
    "For example, Sep 25 = 2025-09. Use ONLY the UNITS column, never the bill amount or payment columns. "
    "Use an empty list if there is no history table.\n"
    '- "city_text": consumer city/area printed on the bill, if visible. Do not return personal names.\n'
    '- "disco": electricity company name or abbreviation, if visible.\n'
    "Double-check the numbers before returning JSON."
)


# =========================================================
# PDF / IMAGE HELPERS
# =========================================================

def extract_pdf_text(uploaded_file):
    data = uploaded_file.getvalue()

    if not uploaded_file.name.lower().endswith(".pdf"):
        return ""

    text = []

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
    except Exception:
        pass

    return "\n".join(text)


def find_kwh(text):
    patterns = [
        r"(?:units|unit|kwh|consumption|energy)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|kwh)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text or "", re.I)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                pass

    return None


def prepare_image(data):
    """Rotate phone photos correctly, resize them, and encode as JPEG."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")

    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    quality = 85

    # Keep the image comfortably below the API image-size limit.
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)

        if len(buf.getvalue()) <= MAX_IMAGE_BYTES or quality <= 55:
            return buf.getvalue()

        quality -= 5


def pdf_to_images(data, max_pages=MAX_PDF_PAGES_FOR_VISION):
    """Render the first pages of a scanned PDF to JPEG bytes."""
    images = []

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages[:max_pages]:
                pil = page.to_image(resolution=150).original

                buf = io.BytesIO()
                pil.convert("RGB").save(
                    buf,
                    format="JPEG",
                    quality=85,
                    optimize=True,
                )

                images.append(buf.getvalue())

    except Exception:
        pass

    return images


# =========================================================
# BILL JSON VALIDATION
# =========================================================

def _valid_units(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if 0 < value < 100000 else None


def _valid_month(month):
    return (
        month
        if isinstance(month, str) and MONTH_RE.fullmatch(month)
        else None
    )


def parse_bill_json(raw):
    """Extract and validate the JSON returned by the vision model."""
    if not raw:
        return None

    match = re.search(r"\{.*\}", raw, re.S)

    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    history = []

    for item in obj.get("history") or []:
        if isinstance(item, dict):
            month = _valid_month(item.get("month"))
            units = _valid_units(item.get("units"))

            if month and units is not None:
                history.append(
                    {
                        "month": month,
                        "units": units,
                    }
                )

    def text_value(value):
        return value if isinstance(value, str) else None

    return {
        "billing_month": _valid_month(obj.get("billing_month")),
        "units_kwh": _valid_units(obj.get("units_kwh")),
        "history": history,
        "city_text": text_value(obj.get("city_text")),
        "disco": text_value(obj.get("disco")),
    }


# =========================================================
# GROQ VISION
# =========================================================

def vision_extract(client, jpeg_bytes, errors=None):
    """
    Read one electricity-bill image with Groq vision.

    The current model is qwen/qwen3.8-27b.
    """
    if errors is None:
        errors = []

    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

    models = [VISION_MODEL]

    for model in VISION_FALLBACK_MODELS:
        if model not in models:
            models.append(model)

    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": VISION_PROMPT,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0,
                max_completion_tokens=700,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            result = parse_bill_json(raw)

            if result and (
                result["units_kwh"] is not None
                or result["history"]
            ):
                return result

            errors.append(
                f"{model}: model returned no usable units/history."
            )

        except Exception as exc:
            errors.append(
                f"{model}: {str(exc)[:500]}"
            )

    return None


def read_bill_with_vision(client, images, errors=None):
    """Read bill image/page(s) and merge the extracted information."""
    if errors is None:
        errors = []

    merged = {
        "billing_month": None,
        "units_kwh": None,
        "history": [],
        "city_text": None,
        "disco": None,
    }

    for image in images:
        result = vision_extract(
            client,
            image,
            errors,
        )

        if not result:
            continue

        for key in (
            "billing_month",
            "units_kwh",
            "city_text",
            "disco",
        ):
            if merged[key] is None:
                merged[key] = result[key]

        merged["history"].extend(result["history"])

        # A bill with current units + history has enough information.
        if (
            merged["units_kwh"] is not None
            and merged["history"]
        ):
            break

    # Remove duplicate history months.
    unique = {}

    for row in merged["history"]:
        unique[row["month"]] = row["units"]

    merged["history"] = [
        {
            "month": month,
            "units": units,
        }
        for month, units in sorted(unique.items())
    ]

    return merged


# =========================================================
# BILL AGENT
# =========================================================

def bill_agent(bills, api_key=""):
    """
    Read uploaded electricity bills.

    Returns the structure expected by app.py:
    files_uploaded, files_read, monthly, city_hint, note, files.
    """
    client = (
        Groq(api_key=api_key)
        if api_key and Groq is not None
        else None
    )

    files = []
    history_rows = []
    current_rows = []

    city_text = None
    disco = None

    skipped_no_key = 0

    for uploaded_file in bills:
        filename = uploaded_file.name
        name = filename.lower()
        data = uploaded_file.getvalue()

        is_pdf = name.endswith(".pdf")
        is_image = name.endswith(IMAGE_EXTS)

        record = {
            "file": filename,
            "billing_month": None,
            "kwh": None,
            "history_months": 0,
            "method": None,
        }

        parsed = None
        errors = []

        try:
            if client is not None and (is_pdf or is_image):

                if is_pdf:
                    images = pdf_to_images(data)
                else:
                    images = [prepare_image(data)]

                if images:
                    parsed = read_bill_with_vision(
                        client,
                        images,
                        errors,
                    )

                    record["method"] = "vision"

            has_data = bool(parsed) and (
                parsed["units_kwh"] is not None
                or bool(parsed["history"])
            )

            # Text PDF fallback.
            if is_pdf and not has_data:
                kwh = find_kwh(extract_pdf_text(uploaded_file))

                if kwh is not None:
                    parsed = {
                        "billing_month": None,
                        "units_kwh": kwh,
                        "history": [],
                        "city_text": None,
                        "disco": None,
                    }

                    record["method"] = "pdf_text"

                    has_data = True

        except Exception as exc:
            errors.append(str(exc))

        has_data = bool(parsed) and (
            parsed["units_kwh"] is not None
            or bool(parsed["history"])
        )

        if has_data:
            record.update(
                {
                    "billing_month": parsed["billing_month"],
                    "kwh": parsed["units_kwh"],
                    "history_months": len(parsed["history"]),
                }
            )

            history_rows.extend(parsed["history"])

            if parsed["units_kwh"] is not None:
                current_rows.append(
                    {
                        "month": (
                            parsed["billing_month"]
                            or f"unknown:{filename}"
                        ),
                        "units": parsed["units_kwh"],
                    }
                )

            city_text = city_text or parsed["city_text"]
            disco = disco or parsed["disco"]

        elif client is None:
            skipped_no_key += 1

        if errors:
            record["error"] = " | ".join(errors)

        files.append(record)

    # Combine history + current readings.
    combined = history_rows + current_rows

    # Remove duplicate month values, preferring the current bill.
    monthly_map = {}

    for row in history_rows:
        monthly_map[row["month"]] = row["units"]

    for row in current_rows:
        month = row["month"]

        if not month.startswith("unknown:"):
            monthly_map[month] = row["units"]

    monthly = [
        {
            "month": month,
            "units": units,
        }
        for month, units in sorted(monthly_map.items())
    ]

    # If only an unknown current bill was found, preserve it.
    if not monthly and current_rows:
        monthly = current_rows

    try:
        stats = consumption_stats(monthly)
    except Exception:
        stats = {
            "monthly": monthly,
            "months_used": len(monthly),
        }

    files_read = sum(
        1
        for record in files
        if record["kwh"] is not None
        or record["history_months"] > 0
    )

    notes = []

    if skipped_no_key:
        notes.append(
            f"{skipped_no_key} bill(s) need the vision model but GROQ_API_KEY is missing."
        )

    if files_read < len(files):
        notes.append(
            "Some files could not be read. Add the months manually in the table below."
        )

    months_used = stats.get("months_used", len(monthly))

    if 0 < months_used < 12:
        notes.append(
            f"Only {months_used} month(s) found. Add the missing months for an accurate 12-month average."
        )

    notes.append(
        "Values read from photos are AI readings: compare them with the bill before continuing."
    )

    city_hint = None

    try:
        if city_text:
            city_hint = detect_city(city_text)
    except Exception:
        city_hint = None

    if not city_hint:
        try:
            if disco:
                city_hint = detect_city(disco)
        except Exception:
            city_hint = None

    return {
        "files": files,
        "files_uploaded": len(files),
        "files_read": files_read,
        "monthly": monthly,
        "city_hint": city_hint,
        "note": " ".join(notes),
    }


# =========================================================
# AI REPORT HELPERS
# =========================================================

def _facts(
    base_stats,
    stats,
    sizing,
    orientation,
    loads,
    panel_brand,
    inverter_brand,
):
    text = (
        f'average monthly consumption from the bills '
        f'{base_stats["average_monthly_kwh"]:.0f} kWh, '
    )

    if loads["loads"]:
        names = ", ".join(
            load["name"]
            for load in loads["loads"]
        )

        text += (
            f'planned new loads ({names}) add about '
            f'{loads["avg_added_kwh_month"]:.0f} kWh/month on average '
            f'(connected load {loads["connected_kw"]:.2f} kW), '
            f'giving {stats["average_monthly_kwh"]:.0f} kWh/month in total, '
        )

    text += (
        f'highest month {stats["max_kwh"]:.0f} kWh '
        f'({stats["max_month"]}), '
        f'lowest month {stats["min_kwh"]:.0f} kWh '
        f'({stats["min_month"]}), '
        f'recommended plant {sizing["recommended_kw"]} kW '
        f'with {sizing["panel_count"]} panels, '
        f'site {orientation["city"]}, '
        f'panels facing true South at about '
        f'{orientation["tilt_deg"]} degrees tilt, '
        f'panel {panel_brand}, '
        f'inverter {inverter_brand}'
    )

    return text


def llm_agent_summary(
    api_key,
    base_stats,
    stats,
    sizing,
    orientation,
    loads,
    panel_brand,
    inverter_brand,
):
    """Direct Groq call used if CrewAI is unavailable."""
    if not api_key or Groq is None:
        return (
            "AI summary skipped: add GROQ_API_KEY "
            "in Streamlit Secrets."
        )

    try:
        client = Groq(api_key=api_key)

        prompt = (
            "You are a solar consultant. Write a concise "
            "customer-facing summary under 150 words using ONLY "
            "the verified numbers below. Do not invent prices "
            "or engineering facts.\n\n"
            + _facts(
                base_stats,
                stats,
                sizing,
                orientation,
                loads,
                panel_brand,
                inverter_brand,
            )
        )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=TEXT_MODEL,
            temperature=0.2,
            max_completion_tokens=500,
        )

        return response.choices[0].message.content

    except Exception as exc:
        return f"AI summary unavailable: {exc}"


def run_crew_report(
    api_key,
    base_stats,
    stats,
    sizing,
    options,
    basis,
    orientation,
    loads,
    panel_brand,
    inverter_brand,
):
    """Run the CrewAI multi-agent solar report."""
    if not api_key:
        return (
            "AI report skipped: add GROQ_API_KEY "
            "in Streamlit Secrets."
        )

    try:
        os.environ.setdefault(
            "OTEL_SDK_DISABLED",
            "true",
        )

        from crewai import (
            Agent,
            Crew,
            LLM,
            Process,
            Task,
        )

        llm = LLM(
            model=CREW_MODEL,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            custom_openai=True,
            temperature=0.1,
            max_tokens=800,
        )

        auditor = Agent(
            role="Electricity Bill Auditor",
            goal=(
                "Check monthly kWh history for gaps, duplicates, "
                "misreads and seasonal patterns."
            ),
            backstory=(
                "You review electricity bill data for a solar "
                "company and only report what the data shows."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )

        reviewer = Agent(
            role="PV Sizing Reviewer",
            goal=(
                "Sanity-check the calculated system size against "
                "customer consumption and planned new loads. "
                "Never change the supplied numbers."
            ),
            backstory=(
                "You are a solar design engineer who checks that "
                "plant size, panel count and expected generation "
                "are internally consistent."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )

        consultant = Agent(
            role="Solar Consultant",
            goal=(
                "Write a short, clear customer-facing summary "
                "using only verified numbers."
            ),
            backstory=(
                "You advise solar customers in Pakistan. "
                "You never invent prices or engineering facts."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )

        audit_task = Task(
            description=(
                "Review this monthly consumption data from the "
                "customer's bill history. Flag missing months, "
                "duplicates and suspicious readings, then describe "
                "the seasonal pattern.\n\n"
                f'Monthly data: {json.dumps(base_stats["monthly"])}\n'
                f'Maximum: {base_stats["max_kwh"]} kWh '
                f'in {base_stats["max_month"]}.\n'
                f'Minimum: {base_stats["min_kwh"]} kWh '
                f'in {base_stats["min_month"]}.'
            ),
            expected_output=(
                "A short bullet list of data-quality issues "
                "and the seasonal pattern."
            ),
            agent=auditor,
        )

        review_task = Task(
            description=(
                "Check that the sizing is consistent with "
                "consumption and planned new loads. Do not "
                "recalculate or change numbers. Flag only "
                "engineering/data concerns.\n\n"
                f'Average bill consumption: '
                f'{base_stats["average_monthly_kwh"]} kWh.\n'
                f'Planned new loads: {json.dumps(loads)}\n'
                f'Average including new loads: '
                f'{stats["average_monthly_kwh"]} kWh.\n'
                f'Sizing options: {json.dumps(options)}\n'
                f'Chosen basis: {basis}\n'
                f'Orientation: {json.dumps(orientation)}'
            ),
            expected_output=(
                'A short bullet list of concerns, or '
                '"Sizing looks consistent".'
            ),
            agent=reviewer,
            context=[audit_task],
        )

        summary_task = Task(
            description=(
                "Write a concise customer-facing summary under "
                "150 words. Use ONLY the verified numbers below. "
                "Mention relevant audit/sizing concerns. Do not "
                "state prices.\n\n"
                + _facts(
                    base_stats,
                    stats,
                    sizing,
                    orientation,
                    loads,
                    panel_brand,
                    inverter_brand,
                )
            ),
            expected_output=(
                "A plain-text customer-facing solar summary."
            ),
            agent=consultant,
            context=[
                audit_task,
                review_task,
            ],
        )

        crew = Crew(
            agents=[
                auditor,
                reviewer,
                consultant,
            ],
            tasks=[
                audit_task,
                review_task,
                summary_task,
            ],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()

        return getattr(
            result,
            "raw",
            str(result),
        )

    except Exception as exc:
        # If CrewAI itself fails, still provide a useful direct
        # Groq summary rather than crashing the Streamlit app.
        fallback = llm_agent_summary(
            api_key,
            base_stats,
            stats,
            sizing,
            orientation,
            loads,
            panel_brand,
            inverter_brand,
        )

        return (
            f"{fallback}\n\n"
            f"(CrewAI fallback: {exc})"
        )


# =========================================================
# SOLAR ANALYSIS
# =========================================================

def run_solar_analysis(
    monthly,
    customer_name,
    city,
    latitude,
    panel_brand,
    panel_watt,
    inverter_brand,
    peak_sun_hours,
    losses,
    sizing_basis="average",
    new_loads=None,
    groq_api_key="",
):
    base_stats = consumption_stats(monthly)

    adjusted, loads = apply_new_loads(
        monthly,
        new_loads or [],
    )

    stats = consumption_stats(adjusted)

    base_options = size_options(
        base_stats,
        peak_sun_hours,
        losses,
        panel_watt,
    )

    options = size_options(
        stats,
        peak_sun_hours,
        losses,
        panel_watt,
    )

    selected_key = (
        "peak"
        if sizing_basis == "peak"
        else "average"
    )

    sizing = options[selected_key]

    orientation = {
        "city": city,
        "latitude": latitude,
        "azimuth_deg": 180,
        "tilt_deg": recommended_tilt(
            latitude,
            stats["summer_heavy"],
        ),
        "summer_biased": stats["summer_heavy"],
    }

    summary = run_crew_report(
        groq_api_key,
        base_stats,
        stats,
        sizing,
        options,
        sizing_basis,
        orientation,
        loads,
        panel_brand,
        inverter_brand,
    )

    return {
        "customer": {
            "name": customer_name,
        },
        "consumption": stats,
        "base_consumption": base_stats,
        "new_loads": loads,
        "sizing": sizing,
        "sizing_options": options,
        "base_sizing_options": base_options,
        "sizing_basis": sizing_basis,
        "orientation": orientation,
        "selection": {
            "panel_brand": panel_brand,
            "panel_watt": panel_watt,
            "inverter_brand": inverter_brand,
        },
        "ai_summary": summary,
    }


# =========================================================
# QUOTATION
# =========================================================

def build_quote(
    result,
    rate,
    inverter_price,
    installation,
):
    kw = result["sizing"]["recommended_kw"]
    panel_w = result["selection"]["panel_watt"]
    count = result["sizing"]["panel_count"]

    panel_cost = kw * 1000 * rate

    total = (
        panel_cost
        + inverter_price
        + installation
    )

    items = [
        {
            "Item": result["selection"]["panel_brand"],
            "Specification": f"{panel_w} W",
            "Quantity": count,
            "Amount PKR": round(panel_cost),
        },
        {
            "Item": result["selection"]["inverter_brand"],
            "Specification": "Approved inverter",
            "Quantity": 1,
            "Amount PKR": round(inverter_price),
        },
        {
            "Item": "Installation / BOS / other",
            "Specification": "",
            "Quantity": 1,
            "Amount PKR": round(installation),
        },
    ]

    text = (
        f'Customer: {result["customer"]["name"]}\n'
        f"Recommended system: {kw:.1f} kW\n"
        f"Total: PKR {total:,.0f}\n"
    )

    return {
        "items": items,
        "total": total,
        "text": text,
    }
