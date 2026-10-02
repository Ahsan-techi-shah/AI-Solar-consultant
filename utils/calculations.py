import re
from statistics import mean

# Approximate latitudes (degrees North). Choose "Other" in the app to type your own.
CITY_LATITUDES = {
    'Peshawar': 34.01, 'Mardan': 34.20, 'Abbottabad': 34.15,
    'Islamabad': 33.68, 'Rawalpindi': 33.60, 'Lahore': 31.55,
    'Faisalabad': 31.42, 'Multan': 30.20, 'Quetta': 30.18, 'Karachi': 24.86,
}
DISCO_DEFAULT_CITY = {
    'pesco': 'Peshawar', 'iesco': 'Islamabad', 'lesco': 'Lahore', 'fesco': 'Faisalabad',
    'mepco': 'Multan', 'qesco': 'Quetta', 'k-electric': 'Karachi', 'k electric': 'Karachi',
}
MONTH_RE = re.compile(r'\d{4}-(0[1-9]|1[0-2])')


def detect_city(text):
    """Guess the site city from text printed on the bill (city/address line first, company name as fallback)."""
    t = (text or '').lower()
    for city in CITY_LATITUDES:
        if city.lower() in t:
            return city
    if re.search(r'\bpesh\b', t):
        return 'Peshawar'
    for disco, city in DISCO_DEFAULT_CITY.items():
        if disco in t:
            return city
    return None


def recommended_tilt(latitude, summer_biased=False):
    """Fixed-tilt rule of thumb for panels facing true South: about 0.87 x latitude for best annual yield,
    5 degrees lower when consumption is summer-heavy. Confirm with PVGIS / PVsyst for the real site."""
    tilt = round(latitude * 0.87)
    if summer_biased:
        tilt -= 5
    return max(tilt, 10)


def consumption_stats(monthly):
    """monthly: list of {'month': 'YYYY-MM', 'units': kWh}. Uses the latest 12 distinct months."""
    by_key = {}
    for r in monthly or []:
        key, units = r.get('month'), r.get('units')
        if not key or units is None:
            continue
        try:
            units = float(units)
        except (TypeError, ValueError):
            continue
        if units < 0:
            continue
        by_key[str(key).strip()] = units  # a later row for the same month replaces an earlier one

    keys = sorted(by_key)[-12:]
    if not keys:
        return {'months_used': 0, 'period': '', 'monthly': [], 'average_monthly_kwh': 0, 'annual_kwh': 0,
                'max_kwh': 0, 'max_month': None, 'min_kwh': 0, 'min_month': None, 'summer_heavy': False}

    vals = [by_key[k] for k in keys]
    avg = mean(vals)
    real = [k for k in keys if MONTH_RE.fullmatch(k)]
    summer = [by_key[k] for k in real if k[5:7] in ('06', '07', '08', '09')]
    return {
        'months_used': len(vals),
        'period': f'{real[0]} to {real[-1]}' if real else '',
        'monthly': [{'month': k, 'units': by_key[k]} for k in keys],
        'average_monthly_kwh': round(avg, 1),
        'annual_kwh': round(sum(vals) if len(vals) == 12 else avg * 12, 1),
        'max_kwh': max(vals), 'max_month': max(keys, key=lambda k: by_key[k]),
        'min_kwh': min(vals), 'min_month': min(keys, key=lambda k: by_key[k]),
        'summer_heavy': bool(summer) and mean(summer) > 1.3 * avg,
    }


def size_solar_system(avg_monthly_kwh, peak_sun_hours=5.0, losses=20, panel_w=585):
    if avg_monthly_kwh <= 0:
        return {'recommended_kw': 0.0, 'panel_count': 0, 'panel_w': panel_w, 'installed_kwp': 0.0,
                'estimated_monthly_generation_kwh': 0.0}
    performance = 1 - losses / 100
    monthly_kwh_per_kw = peak_sun_hours * 30 * performance
    required_kw = avg_monthly_kwh / monthly_kwh_per_kw
    recommended_kw = max(0.5, round(required_kw * 2) / 2)
    if recommended_kw < required_kw:
        recommended_kw += 0.5
    panel_count = int((recommended_kw * 1000 + panel_w - 1) // panel_w)
    installed_kwp = panel_count * panel_w / 1000
    return {
        'recommended_kw': round(recommended_kw, 1),
        'panel_count': panel_count,
        'panel_w': panel_w,
        'installed_kwp': round(installed_kwp, 2),
        'estimated_monthly_generation_kwh': round(installed_kwp * monthly_kwh_per_kw, 1),
        'assumptions': {'peak_sun_hours': peak_sun_hours, 'system_losses_percent': losses},
    }


def size_options(stats, peak_sun_hours=5.0, losses=20, panel_w=585):
    """Two sizing bases: the 12-month average (annual offset) and the peak month."""
    return {
        'average': size_solar_system(stats['average_monthly_kwh'], peak_sun_hours, losses, panel_w),
        'peak': size_solar_system(stats['max_kwh'], peak_sun_hours, losses, panel_w),
    }


# ---------- Planned new loads (AC etc.) ----------
SEASONS = {
    'All year': set(range(1, 13)),
    'Summer (Apr-Oct)': {4, 5, 6, 7, 8, 9, 10},
    'Winter (Nov-Mar)': {11, 12, 1, 2, 3},
}


def _num(v, default=0.0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if x != x else x  # NaN -> default


def load_kwh_per_month(load):
    """kWh added in every month the load is used = qty x watts x hours/day x days/month x duty% / 1000."""
    qty = _num(load.get('qty'), 1.0)
    watts = _num(load.get('watts'))
    hours = _num(load.get('hours'))
    days = _num(load.get('days'), 30.0)
    duty = _num(load.get('duty'), 100.0)
    return qty * watts * hours * days * (duty / 100.0) / 1000.0


def apply_new_loads(monthly, loads):
    """Add planned new loads to each month of the bill history.
    loads: [{'name','qty','watts','hours','days','duty','season'}].
    Returns (new_monthly, summary)."""
    clean = []
    for ld in loads or []:
        kwh = load_kwh_per_month(ld)
        if kwh > 0:
            season = ld.get('season') if ld.get('season') in SEASONS else 'All year'
            clean.append({
                'name': str(ld.get('name') or 'New load'),
                'season': season,
                'kwh_per_month': round(kwh, 1),
                'connected_kw': round(_num(ld.get('qty'), 1.0) * _num(ld.get('watts')) / 1000.0, 2),
            })

    base = consumption_stats(monthly)['monthly']
    new_monthly = []
    for r in base:
        month_no = int(r['month'][5:7]) if MONTH_RE.fullmatch(r['month']) else None
        add = 0.0
        for ld in clean:
            in_season = (ld['season'] == 'All year') if month_no is None else (month_no in SEASONS[ld['season']])
            if in_season:
                add += ld['kwh_per_month']
        new_monthly.append({'month': r['month'], 'units': round(r['units'] + add, 1)})

    added_total = sum(n['units'] - b['units'] for n, b in zip(new_monthly, base))
    summary = {
        'loads': clean,
        'connected_kw': round(sum(ld['connected_kw'] for ld in clean), 2),
        'avg_added_kwh_month': round(added_total / len(base), 1) if base else 0.0,
    }
    return new_monthly, summary
