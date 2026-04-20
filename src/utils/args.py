import argparse


def str_to_bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise argparse.ArgumentTypeError(
        f"Boolean value expected ('true'/'false'), got {value!r}"
    )


def positive_int(raw_value: str) -> int:
    parsed_value = int(raw_value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return parsed_value


def positive_float(raw_value: str) -> float:
    parsed_value = float(raw_value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive float")
    return parsed_value
