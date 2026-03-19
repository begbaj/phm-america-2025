from typing import Any

def print_title(msg: str):
    print("=" * 60)
    print(msg.upper())
    print("=" * 60)


def print_line(msg: str):
    print(f"    {msg}")


def print_data(msg: str, data_val: Any):
    data_val = str(data_val)
    print_line(f"{msg.ljust(30)}{data_val.rjust(20)}")
