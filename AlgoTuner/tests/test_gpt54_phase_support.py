from types import SimpleNamespace
import unittest
from unittest.mock import patch

from AlgoTuner.interfaces.core.base_interface import InterfaceState
from AlgoTuner.interfaces.core.message_handler import MessageHandler
from AlgoTuner.interfaces.llm_interface import LLMInterface
from AlgoTuner.models.lite_llm_model import LiteLLMModel


def _make_handler(model_name: str = "openrouter/openai/gpt-5.4") -> tuple[MessageHandler, SimpleNamespace]:
    interface = SimpleNamespace(state=InterfaceState(), model_name=model_name)
    return MessageHandler(interface), interface


class GPT54PhaseSupportTests(unittest.TestCase):
    def test_add_message_preserves_assistant_phase_only(self):
        handler, interface = _make_handler()

        handler.add_message("assistant", {"message": "Draft command", "phase": "commentary"})
        handler.add_message("user", {"message": "Proceed", "phase": "commentary"})

        self.assertEqual(
            interface.state.messages[0],
            {
                "role": "assistant",
                "content": "Draft command",
                "phase": "commentary",
            },
        )
        self.assertEqual(interface.state.messages[1], {"role": "user", "content": "Proceed"})

    def test_prepare_truncated_history_keeps_phase_on_truncated_assistant_messages(self):
        handler, _ = _make_handler()
        full_history = [{"role": "system", "content": "System prompt"}]

        for i in range(1, 7):
            full_history.append({"role": "user", "content": f"user {i}"})
            assistant_message = {
                "role": "assistant",
                "content": f"assistant {i}",
                "phase": "commentary",
            }
            if i == 1:
                assistant_message["content"] = "A" * 150
            full_history.append(assistant_message)

        prepared = handler._prepare_truncated_history(full_history, token_limit=100000)
        truncated_assistant = next(
            msg
            for msg in prepared["messages"]
            if msg["role"] == "assistant" and msg["content"].endswith("...")
        )

        self.assertEqual(truncated_assistant["phase"], "commentary")
        self.assertEqual(len(truncated_assistant["content"]), 103)

    def test_litellm_model_query_extracts_phase(self):
        model = LiteLLMModel(model_name="openrouter/openai/gpt-5.4", api_key="test-key")

        def fake_completion(**kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "edit: solver.py\nlines: 1-1\n---\npass\n---",
                            "phase": "commentary",
                        }
                    }
                ],
                "cost": 0.25,
            }

        with patch("AlgoTuner.models.lite_llm_model.litellm.completion", fake_completion):
            response = model.query(
                [
                    {"role": "system", "content": "You are an agent."},
                    {"role": "user", "content": "Make an edit."},
                ]
            )

        self.assertTrue(response["message"].startswith("edit: solver.py"))
        self.assertEqual(response["phase"], "commentary")

    def test_llm_interface_get_response_returns_phase_metadata(self):
        state = SimpleNamespace(
            messages=[
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Continue"},
            ],
            spend=0.0,
        )
        summary = {
            "kept_essential_indices": set(),
            "included_older_indices": set(),
            "content_truncated_older_indices": set(),
            "dropped_older_indices": set(),
            "final_token_count": 12,
        }
        interface = SimpleNamespace(
            state=state,
            model_config=SimpleNamespace(context_length=4096),
            message_handler=SimpleNamespace(
                _prepare_truncated_history=lambda messages, token_limit: {
                    "messages": messages,
                    "summary": summary,
                }
            ),
            model=SimpleNamespace(
                query=lambda messages: {
                    "message": "edit: solver.py\nlines: 1-1\n---\npass\n---",
                    "phase": "commentary",
                    "cost": 0.5,
                }
            ),
            check_limits=lambda: None,
            update_spend=lambda cost: state.__setattr__("spend", state.spend + cost) or True,
            _content_policy_violation=False,
        )

        response = LLMInterface.get_response(interface)

        self.assertEqual(
            response,
            {
                "message": "edit: solver.py\nlines: 1-1\n---\npass\n---",
                "phase": "commentary",
            },
        )
        self.assertEqual(state.spend, 0.5)
