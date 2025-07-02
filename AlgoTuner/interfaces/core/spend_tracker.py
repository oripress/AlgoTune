import logging
from threading import Lock
from typing import Dict, Any, Optional

from AlgoTuner.interfaces.core.base_interface import BaseLLMInterface
from AlgoTuner.utils.message_writer import MessageWriter


class SpendTracker:
    """Tracks and manages spending for LLM usage."""

    def __init__(self, interface: BaseLLMInterface):
        self.interface = interface
        self.spend_lock = Lock()
        self.message_writer = MessageWriter()

    def update_spend(self, amount: float) -> Dict[str, Any]:
        """
        Update the total spend and check against limits.
        Returns a dict with status and any relevant messages.
        """
        with self.spend_lock:
            self.interface.state.spend += amount

            if self.interface.state.spend >= self.interface.spend_limit:
                msg = f"Spend limit of ${self.interface.spend_limit:.4f} reached. Current spend: ${self.interface.state.spend:.4f}"
                logging.warning(msg)
                return {
                    "success": False,
                    "error": "spend_limit_reached",
                    "message": msg,
                    "spend": self.interface.state.spend,
                }

            if self.interface.state.messages_sent >= self.interface.total_messages:
                msg = f"Message limit of {self.interface.total_messages} reached. Messages sent: {self.interface.state.messages_sent}"
                logging.warning(msg)
                return {
                    "success": False,
                    "error": "message_limit_reached",
                    "message": msg,
                    "spend": self.interface.state.spend,
                }

            return {"success": True, "spend": self.interface.state.spend}

    def get_spend_status(self) -> Dict[str, float]:
        """Get current spending status."""
        with self.spend_lock:
            return {
                "current_spend": self.interface.state.spend,
                "spend_limit": self.interface.spend_limit,
                "remaining": max(0, self.interface.spend_limit - self.interface.state.spend),
                "messages_sent": self.interface.state.messages_sent,
                "messages_limit": self.interface.total_messages,
                "messages_remaining": max(
                    0, self.interface.total_messages - self.interface.state.messages_sent
                ),
            }

    def reset_spend(self) -> None:
        """Reset spending metrics."""
        with self.spend_lock:
            self.interface.state.spend = 0
            self.interface.state.messages_sent = 0

    def format_spend_status(self) -> str:
        """Format current spending status as a user-friendly string."""
        status = self.get_spend_status()
        return self.message_writer.format_budget_status(
            spend=status["current_spend"],
            remaining=status["remaining"],
            messages_sent=status["messages_sent"],
            messages_remaining=status["messages_remaining"],
        )

    def check_limits(self) -> Optional[str]:
        """
        Check if any limits have been reached.
        Returns a formatted error message if limits are reached, None otherwise.
        """
        with self.spend_lock:
            if error := self._check_limit():
                return error

            # Check message limit
            if (
                self.interface.total_messages > 0
                and self.interface.state.messages_sent >= self.interface.total_messages
            ):
                return f"Message limit of {self.interface.total_messages} reached"

            return None

    def _check_limit(self) -> Optional[str]:
        """
        Check if spending has exceeded the limit.
        Returns an error message if the limit has been reached, otherwise None.
        """
        if not self.interface:
            return None

        if self.interface.spend_limit <= 0:
            return None  # No limit set

        if self.interface.state.spend >= self.interface.spend_limit:
            msg = f"Spend limit of ${self.interface.spend_limit:.4f} reached. Current spend: ${self.interface.state.spend:.4f}"
            logging.warning(msg)
            return msg

        return None
