"""Strict calendar/driver helpers for the isolated085 copy. Standard library only."""
from datetime import datetime, timedelta
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]

def calendar_date_from_state(state):
    """Read the live DSSAT calendar. Never infer dates from planting/DAP."""
    if not isinstance(state, dict) or 'yrdoy' not in state:
        raise ValueError('DSSAT raw state has no yrdoy; DAP fallback forbidden')
    value = state['yrdoy']
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError('Invalid yrdoy')
    code = int(value)
    if float(value) != code:
        raise ValueError('Noninteger yrdoy')
    year, doy = divmod(code, 1000)
    if year < 1900 or year > 2200 or doy < 1:
        raise ValueError('Invalid yrdoy calendar range')
    date = datetime.strptime(str(code), '%Y%j').date()
    if date.year != year:
        raise ValueError('yrdoy spills into next year')
    return date

def read_actual_wth(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Driver path outside project')
    records = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts[0]) != 7 or len(parts) < 5:
            raise ValueError('Expected YYYYDDD SRAD TMAX TMIN RAIN driver columns')
        date = calendar_date_from_state({'yrdoy': int(parts[0])})
        if date in records:
            raise ValueError('Duplicate WTH date')
        values = [float(v) for v in parts[1:5]]
        if not all(math.isfinite(v) and v != -99 for v in values):
            raise ValueError('Missing/nonfinite WTH weather')
        records[date] = dict(zip(['srad', 'tmax', 'tmin', 'rain'], values))
    if not records:
        raise ValueError('No driver records')
    return records

def next_calendar_window(state, records):
    """Calendar+1..+7, pending independent interpretation of the action phase."""
    date = calendar_date_from_state(state)
    days = [date + timedelta(days=i) for i in range(1,8)]
    if any(d.year != date.year or d not in records for d in days):
        raise ValueError('Incomplete/cross-year seven-day driver window')
    return [(d, records[d]) for d in days]
