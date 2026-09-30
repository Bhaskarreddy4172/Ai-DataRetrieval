"""Unit tests for Ollama Provider configuration, health tracking, retry policy, and fallback hierarchy."""

import unittest
from unittest.mock import MagicMock, patch
import requests

from app.config import settings
from app.ai.ollama_client import OllamaClient
from app.llm.ollama_provider import OllamaProvider
from app.llm.schemas import ToolDefinition, LLMResponse


class TestOllamaConfigurationAndFallback(unittest.TestCase):
    """Test suite verifying Ollama configuration, health check enum statuses, retries, and fallback behavior."""

    def test_01_configuration_settings(self):
        """Verify centralized Ollama settings loaded correctly in app/config.py."""
        self.assertTrue(hasattr(settings, "OLLAMA_BASE_URL"))
        self.assertTrue(hasattr(settings, "OLLAMA_MODEL"))
        self.assertTrue(hasattr(settings, "OLLAMA_TIMEOUT"))
        self.assertTrue(hasattr(settings, "OLLAMA_MAX_RETRIES"))
        self.assertTrue(hasattr(settings, "OLLAMA_TEMPERATURE"))
        self.assertTrue(hasattr(settings, "TOOL_TIMEOUT"))
        self.assertTrue(hasattr(settings, "AGENT_TIMEOUT"))

        self.assertIsInstance(settings.OLLAMA_MAX_RETRIES, int)
        self.assertGreaterEqual(settings.OLLAMA_MAX_RETRIES, 1)

    @patch("requests.get")
    def test_02_health_check_ready(self, mock_get):
        """Verify check_health status is READY when model exists."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "llama3.1:8b"}]
        }
        mock_get.return_value = mock_resp

        client = OllamaClient(model="llama3.1:8b")
        health = client.check_health(force_refresh=True)

        self.assertEqual(health["status"], "READY")
        self.assertTrue(health["available"])
        self.assertTrue(health["service_running"])

    @patch("requests.get")
    def test_03_health_check_model_not_found(self, mock_get):
        """Verify check_health status is MODEL_NOT_FOUND when model is missing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "mistral:latest"}]
        }
        mock_get.return_value = mock_resp

        client = OllamaClient(model="nonexistent-model:99b")
        health = client.check_health(force_refresh=True)

        self.assertEqual(health["status"], "MODEL_NOT_FOUND")
        self.assertFalse(health["available"])
        self.assertTrue(health["service_running"])

    @patch("requests.get")
    def test_04_health_check_unavailable(self, mock_get):
        """Verify check_health status is UNAVAILABLE on ConnectionError."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = OllamaClient()
        health = client.check_health(force_refresh=True)

        self.assertEqual(health["status"], "UNAVAILABLE")
        self.assertFalse(health["available"])

    @patch("requests.get")
    def test_05_health_check_timeout(self, mock_get):
        """Verify check_health status is TIMEOUT on Timeout exception."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        client = OllamaClient()
        health = client.check_health(force_refresh=True)

        self.assertEqual(health["status"], "TIMEOUT")
        self.assertFalse(health["available"])

    @patch("requests.post")
    def test_06_generate_retry_policy(self, mock_post):
        """Verify generate retries up to max_retries on requests failure."""
        mock_post.side_effect = [
            requests.exceptions.Timeout("Attempt 1 timed out"),
            requests.exceptions.Timeout("Attempt 2 timed out"),
            MagicMock(status_code=200, json=lambda: {"response": "Success response"}),
        ]

        client = OllamaClient(max_retries=2)
        res = client.generate("Test prompt")

        self.assertEqual(res, "Success response")
        self.assertEqual(mock_post.call_count, 3)

    def test_07_status_messages(self):
        """Verify standardized user-facing status messages in OllamaProvider."""
        provider = OllamaProvider()

        msg_unavail = provider.get_status_message("UNAVAILABLE")
        self.assertIn("temporarily unavailable", msg_unavail.lower())

        msg_not_found = provider.get_status_message("MODEL_NOT_FOUND")
        self.assertIn("model is currently unavailable", msg_not_found.lower())

        msg_err = provider.get_status_message("ERROR")
        self.assertIn("couldn't process that request", msg_err.lower())

        msg_missing = provider.get_status_message("MISSING_DATA")
        self.assertIn("not available in the uploaded dataset", msg_missing.lower())

    @patch("app.ai.ollama_client.ollama_client.check_health")
    def test_08_deterministic_fallback_when_ollama_offline(self, mock_health):
        """Verify OllamaProvider falls back to deterministic planning when offline."""
        mock_health.return_value = {"available": False, "status": "UNAVAILABLE"}

        provider = OllamaProvider()
        tools = [
            ToolDefinition(
                name="aggregate_dataset",
                description="Aggregates dataset column",
                parameters={"type": "object", "properties": {"operation": {"type": "string"}}},
            )
        ]
        schema = {"columns": [{"name": "salary", "type": "number"}]}

        response = provider.select_tool(
            question="What is the maximum salary?",
            available_tools=tools,
            dataset_schema=schema,
        )

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].tool_name, "aggregate_dataset")
        self.assertEqual(response.tool_calls[0].arguments.get("operation"), "MAX")

    @patch("app.ai.ollama_client.ollama_client.check_health")
    @patch("app.ai.ollama_client.ollama_client.generate")
    def test_09_tool_selection_when_ollama_ready(self, mock_generate, mock_health):
        """Verify OllamaProvider parses structured tool call when Ollama is ready."""
        mock_health.return_value = {"available": True, "status": "READY"}
        mock_generate.return_value = '{"tool_name": "aggregate_dataset", "arguments": {"operation": "MAX", "column": "salary"}}'

        provider = OllamaProvider()
        tools = [
            ToolDefinition(
                name="aggregate_dataset",
                description="Aggregates dataset column",
                parameters={"type": "object", "properties": {"operation": {"type": "string"}}},
            )
        ]
        schema = {"columns": [{"name": "salary", "type": "number"}]}

        response = provider.select_tool(
            question="What is the maximum salary?",
            available_tools=tools,
            dataset_schema=schema,
        )

        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].tool_name, "aggregate_dataset")
        self.assertEqual(response.tool_calls[0].arguments.get("column"), "salary")


if __name__ == "__main__":
    unittest.main()

