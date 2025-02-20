import asyncio

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from prompttrail.agent.runners import APIRunner
from prompttrail.agent.templates import (
    AssistantTemplate,
    LinearTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.user_interface import UserInterface
from prompttrail.models.openai import OpenAIConfig, OpenAIModel


class MockUserInterface(UserInterface):
    def ask(self, session, instruction, default=None):
        return "mock input"


@pytest.fixture
def api_runner():
    model = OpenAIModel(
        OpenAIConfig(api_key="mock", model_name="mock"),
    )
    template = LinearTemplate(
        [
            SystemTemplate(content="System: Starting test"),
            AssistantTemplate(content="Assistant: Hello"),
            UserTemplate(description="Please input something"),
            AssistantTemplate(content="Final response"),
        ]
    )
    user_interface = MockUserInterface()
    return APIRunner(model, template, user_interface, host="127.0.0.1", port=8000)


@pytest.fixture
def test_client(api_runner):
    return TestClient(api_runner.app)


@pytest.mark.asyncio
async def test_create_session(test_client):
    response = test_client.post("/sessions", json={"metadata": {"test": "value"}})
    assert response.status_code == 200
    assert "session_id" in response.json()
    session_id = response.json()["session_id"]

    response = test_client.get(f"/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert "is_running" in data
    assert "session" in data
    assert data["is_running"] is False


@pytest.mark.asyncio
async def test_full_conversation_flow(api_runner):
    transport = ASGITransport(api_runner.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create session
        response = await client.post("/sessions", json={"metadata": {}})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        session_id = data["session_id"]

        # Start session
        response = await client.post(f"/sessions/{session_id}/start")
        assert response.status_code == 200

        # Wait for session to process
        await asyncio.sleep(0.1)

        # Check status
        response = await client.get(f"/sessions/{session_id}")
        data = response.json()
        assert data["is_running"]
        assert data["has_event"]

        # Send input
        response = await client.post(
            f"/sessions/{session_id}/input", json={"input": "test input"}
        )
        assert response.status_code == 200

        # Wait for session to process
        await asyncio.sleep(0.1)

        # Check status
        response = await client.get(f"/sessions/{session_id}")
        data = response.json()
        assert data["is_running"] is False
