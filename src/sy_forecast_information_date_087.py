"""Separate information date from DSSAT's upcoming action calendar."""
from datetime import timedelta
from sy_forecast_calendar_driver_085 import calendar_date_from_state

def last_observed_weather_date(state,driver_records):
    action_date=calendar_date_from_state(state)
    observed_date=action_date-timedelta(days=1)
    if observed_date not in driver_records:
        raise ValueError('No driver weather for last observed day')
    for key in ['rain','srad','tmax','tmin']:
        if key not in state or abs(float(state[key])-driver_records[observed_date][key])>1e-4:
            raise ValueError(f'Raw weather is not confirmed as previous-day weather:{key}')
    return observed_date
