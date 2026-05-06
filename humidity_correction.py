# humidity_correction.py
# ---------------------------------------------------------
# Real-time PM10 humidity + temperature correction module
# Fixed parameters fitted from Bristol public data
# ---------------------------------------------------------

from typing import Optional


# ---------------------------------------------------------
# Fixed fitted parameters
# ---------------------------------------------------------
RH0 = 40.0
T0 = 20.0

# Fitted from your previous Bristol model
A_HUMIDITY = 0.005251833375098682
B_HUMIDITY = 1.5808939118967377e-05


def clamp_rh(rh: float) -> float:
    """
    Clamp relative humidity into physical range [0, 100].
    """
    rh = float(rh)
    if rh < 0.0:
        return 0.0
    if rh > 100.0:
        return 100.0
    return rh


def humidity_gain_formula(
    rh: float,
    a: float = A_HUMIDITY,
    b: float = B_HUMIDITY,
    rh0: float = RH0,
) -> float:
    """
    Humidity gain model:
        gain(RH) = 1,                         for RH <= rh0
        gain(RH) = 1 + a*x + b*x^2,         for RH > rh0
        where x = RH - rh0

    The gain is constrained to be >= 1.
    """
    rh = clamp_rh(rh)

    if rh <= rh0:
        return 1.0

    x = rh - rh0
    gain = 1.0 + a * x + b * x * x

    if gain < 1.0:
        gain = 1.0

    return gain


def temperature_factor(temperature: float, T0: float = T0) -> float:
    """
    Temperature multiplicative factor:
        f_T(T) = 1 - min(0.20, 0.0075 * |T - T0|)

    Meaning:
    - no correction at T0 = 20掳C
    - larger deviation from 20掳C increases correction
    - maximum correction contribution is limited to 20%
    """
    temperature = float(temperature)
    factor = 1.0 - min(0.20, 0.0075 * abs(temperature - T0))

    # Safety clamp, although formula should already keep it positive
    if factor <= 0.0:
        factor = 0.01

    return factor


def correct_pm10_realtime(
    rh: float,
    temperature: float,
    pm10_raw: float,
    a: float = A_HUMIDITY,
    b: float = B_HUMIDITY,
    rh0: float = RH0,
    T0: float = T0,
) -> float:
    """
    Final real-time correction formula:
        PM10_corrected = PM10_raw / (humidity_gain * temperature_factor)
    """
    pm10_raw = float(pm10_raw)

    if pm10_raw < 0.0:
        pm10_raw = 0.0

    humidity_gain = humidity_gain_formula(rh=rh, a=a, b=b, rh0=rh0)
    temp_factor = temperature_factor(temperature=temperature, T0=T0)

    corrected = pm10_raw / (humidity_gain * temp_factor)

    if corrected < 0.0:
        corrected = 0.0

    return corrected


def correct_pm10_with_fixed_params(
    rh: float,
    temperature: float,
    pm10_raw: float,
) -> float:
    """
    Convenience wrapper using the fixed fitted parameters.
    """
    return correct_pm10_realtime(
        rh=rh,
        temperature=temperature,
        pm10_raw=pm10_raw,
        a=A_HUMIDITY,
        b=B_HUMIDITY,
        rh0=RH0,
        T0=T0,
    )


def correct_pm10_with_fixed_params_rounded(
    rh: float,
    temperature: float,
    pm10_raw: float,
    ndigits: int = 2,
) -> float:
    """
    Same as correct_pm10_with_fixed_params(), but rounded for direct display/storage.
    """
    value = correct_pm10_with_fixed_params(
        rh=rh,
        temperature=temperature,
        pm10_raw=pm10_raw,
    )
    return round(value, ndigits)


def get_correction_parameters() -> dict:
    """
    Return the currently used fixed correction parameters.
    Useful for logging/debugging/reporting.
    """
    return {
        "rh0": RH0,
        "T0": T0,
        "a_humidity": A_HUMIDITY,
        "b_humidity": B_HUMIDITY,
    }


def explain_formula() -> str:
    """
    Return a readable text description of the deployed formula.
    """
    return (
        "For RH <= 40.0: humidity_gain = 1.0\n"
        "For RH > 40.0: humidity_gain = "
        f"1 + {A_HUMIDITY}*(RH - 40.0) + {B_HUMIDITY}*(RH - 40.0)^2\n"
        "temperature_factor = 1 - min(0.20, 0.0075 * abs(T - 20.0))\n"
        "PM10_corrected = PM10_raw / (humidity_gain * temperature_factor)"
    )


if __name__ == "__main__":
    # Simple local test
    example_rh = 65.0
    example_temp = 40.0
    example_pm10 = 80.0

    corrected = correct_pm10_with_fixed_params_rounded(
        rh=example_rh,
        temperature=example_temp,
        pm10_raw=example_pm10,
        ndigits=2,
    )

    print("=== PM10 correction test ===")
    print("Parameters:", get_correction_parameters())
    print(explain_formula())
    print(f"RH = {example_rh}")
    print(f"Temperature = {example_temp}")
    print(f"PM10_raw = {example_pm10}")
    print(f"PM10_corrected = {corrected}")
