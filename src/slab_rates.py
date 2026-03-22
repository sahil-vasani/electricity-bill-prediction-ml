# State-wise slab definitions (simplified realistic values)

STATE_SLABS = {
    "Gujarat": [
        (50, 3.0),
        (200, 4.5),
        (400, 6.0),
        (9999, 7.0)
    ],
    "Maharashtra": [
        (100, 4.0),
        (300, 6.0),
        (500, 8.0),
        (9999, 10.0)
    ],
    "Delhi": [
        (200, 3.0),
        (400, 5.5),
        (800, 7.0),
        (9999, 8.5)
    ],
    "Karnataka": [
        (100, 4.5),
        (300, 6.5),
        (500, 8.0),
        (9999, 9.5)
    ],
    "Tamil Nadu": [
        (100, 3.5),
        (300, 5.0),
        (500, 6.5),
        (9999, 7.5)
    ],
    "Rajasthan": [
        (100, 4.0),
        (300, 6.5),
        (500, 8.0),
        (9999, 9.0)
    ],
    "Uttar Pradesh": [
        (100, 5.0),
        (300, 6.5),
        (500, 8.0),
        (9999, 9.5)
    ],
    "West Bengal": [
        (100, 4.0),
        (300, 5.5),
        (500, 7.0),
        (9999, 8.5)
    ],
    "Madhya Pradesh": [
        (100, 4.5),
        (300, 6.0),
        (500, 7.5),
        (9999, 9.0)
    ],
    "Telangana": [
        (100, 4.0),
        (300, 6.0),
        (500, 8.0),
        (9999, 9.5)
    ],
}


def calculate_slab_bill(units, state):
    slabs = STATE_SLABS.get(state, STATE_SLABS["Gujarat"])

    remaining_units = units
    total_bill = 0
    prev_limit = 0

    for limit, rate in slabs:
        slab_units = min(remaining_units, limit - prev_limit)
        total_bill += slab_units * rate
        remaining_units -= slab_units
        prev_limit = limit

        if remaining_units <= 0:
            break

    return total_bill