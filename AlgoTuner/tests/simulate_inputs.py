#!/usr/bin/env python3
import sys
from typing import List
from pathlib import Path

from interfaces.commands.handlers import CommandHandlers
from interfaces.core.base_interface import BaseLLMInterface

INPUT_SEPARATOR = "[INPUT_SEPARATOR]"


def read_inputs(file_path: str) -> List[str]:
    """Read inputs from a file, separated by INPUT_SEPARATOR."""
    with open(file_path, "r") as f:
        content = f.read()
        # Split by separator and filter out empty inputs
        inputs = [inp.strip() for inp in content.split(INPUT_SEPARATOR)]
        return [inp for inp in inputs if inp]


def simulate_inputs(inputs: List[str]):
    """
    Simulates sending a series of inputs to the system, one at a time,
    waiting for each response before sending the next input.
    """
    # Create a real interface instance - this will process inputs like in production
    interface = BaseLLMInterface()
    handlers = CommandHandlers(interface)

    print("\n=== Starting Input Simulation ===\n")

    for i, user_input in enumerate(inputs, 1):
        print(f"\n--- Input {i} ---")
        print(f"User: {user_input}")

        # Process the input through the interface
        response = interface.get_response(user_input)
        print(f"Assistant: {response}")

    print("\n=== Simulation Complete ===")


def main():
    if len(sys.argv) != 2:
        print("Usage: python simulate_inputs.py <input_file>")
        print(f"Inputs in the file should be separated by {INPUT_SEPARATOR}")
        sys.exit(1)

    input_file = sys.argv[1]
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)

    inputs = read_inputs(input_file)
    if not inputs:
        print("Error: No valid inputs found in file")
        sys.exit(1)

    simulate_inputs(inputs)


if __name__ == "__main__":
    main()
