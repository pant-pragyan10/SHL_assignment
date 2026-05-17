from shl_agent.api.app import app


def test_health_endpoint():
    assert app is not None
