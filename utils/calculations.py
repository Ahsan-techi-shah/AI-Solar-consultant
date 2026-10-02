def size_solar_system(avg_monthly_kwh, peak_sun_hours=5.0, losses=20, panel_w=585):
    if avg_monthly_kwh <= 0:
        return {'recommended_kw': 0.0, 'panel_count': 0, 'panel_w': panel_w, 'estimated_monthly_generation_kwh': 0.0}
    performance = 1 - losses / 100
    monthly_kwh_per_kw = peak_sun_hours * 30 * performance
    required_kw = avg_monthly_kwh / monthly_kwh_per_kw
    recommended_kw = max(0.5, round(required_kw * 2) / 2)
    if recommended_kw < required_kw:
        recommended_kw += 0.5
    panel_count = int((recommended_kw * 1000 + panel_w - 1) // panel_w)
    return {
        'recommended_kw': round(recommended_kw, 1),
        'panel_count': panel_count,
        'panel_w': panel_w,
        'estimated_monthly_generation_kwh': round(recommended_kw * monthly_kwh_per_kw, 1),
        'assumptions': {'peak_sun_hours': peak_sun_hours, 'system_losses_percent': losses}
    }
