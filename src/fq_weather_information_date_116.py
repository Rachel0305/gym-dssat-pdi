"""FQ-only DSSAT zero-radiation identity check; no forecast/input changes."""
import math
from datetime import timedelta
from sy_forecast_calendar_driver_085 import calendar_date_from_state

def expected_observed_weather(record):
    values=dict(record)
    # Reproduced with the project's current DSSAT: WTH SRAD=0 -> state SRAD=0.1.
    if values['srad']==0.0:
        values['srad']=0.1
    return values

def last_observed_weather_date(state,driver_records):
    date=calendar_date_from_state(state)-timedelta(days=1)
    if date not in driver_records:
        raise ValueError('No driver weather for last observed day')
    expected=expected_observed_weather(driver_records[date])
    for key in ['rain','srad','tmax','tmin']:
        value=float(state.get(key,float('nan')))
        if not math.isfinite(value) or abs(value-expected[key])>1e-4:
            raise ValueError(f'FQ weather identity mismatch: {date} {key}, state={value}, expected={expected[key]}')
    return date
