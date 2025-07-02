import logging
from typing import List, Dict, Union
from AlgoTuner.utils.message_writer import MessageWriter


class DummyLLM:
    """A dummy LLM implementation that calculates costs based on character count."""

    # Cost per character (both input and output)
    COST_PER_CHAR = 0.01  # $0.01 per character

    def __init__(self, model_name: str = "dummy", api_key: str = None, **kwargs):
        self.model_name = model_name
        self.message_writer = MessageWriter()
        logging.info(self.message_writer.format_system_message("DummyLLM initialized"))

    def query(self, messages: List[Dict[str, str]]) -> Dict[str, Union[str, float]]:
        """Process messages and return response with cost calculation."""
        # Calculate input cost based on total characters in messages
        input_chars = sum(len(msg["content"]) for msg in messages)
        input_cost = input_chars * self.COST_PER_CHAR

        # Generate a simple response
        response = "This is a dummy response from DummyLLM."
        output_chars = len(response)
        output_cost = output_chars * self.COST_PER_CHAR

        # Total cost is sum of input and output costs
        total_cost = input_cost + output_cost

        logging.info(
            f"DummyLLM cost calculation: Input chars={input_chars}, Output chars={output_chars}, Total cost=${total_cost:.4f}"
        )

        return {"message": response, "cost": total_cost}
