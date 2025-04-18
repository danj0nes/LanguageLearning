def validate_int_bound(bound, default: int, label: str = "bound"):
    if bound is None:
        print(f"Error: {label} is None, using default.")
        return default

    if not isinstance(bound, int):
        print(f"Error: {label} is not an integer, using default.")
        return default

    return bound
