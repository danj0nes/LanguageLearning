def validate_int_bound(
    bound, min_allowed: int, max_allowed: int, default: int, label="bound"
):
    if bound is None:
        return default

    if not isinstance(bound, int):
        print(f"Error: {label} is not an integer, using default.")
        return default

    if bound < min_allowed or bound > max_allowed:
        print(
            f"Error: {label} is out of range ({min_allowed}-{max_allowed}), using default."
        )
        return default

    return bound


def validate_int_bounds(
    l_bound, u_bound, min_allowed: int, max_allowed: int, l_default: int, u_default: int
):
    new_l_bound = validate_int_bound(
        l_bound, min_allowed, max_allowed, l_default, label="lower bound"
    )
    new_u_bound = validate_int_bound(
        u_bound, min_allowed, max_allowed, u_default, label="upper bound"
    )

    if new_l_bound > new_u_bound:
        print(
            "Error: upper bound must be greater than or equal to lower bound, using defaults."
        )
        return l_default, u_default

    return new_l_bound, new_u_bound


def simple_validate_int_bound(bound, default: int, label: str = "bound"):
    if bound is None:
        return default

    if not isinstance(bound, int):
        print(f"Error: {label} is not an integer, using default.")
        return default

    return bound
