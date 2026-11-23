from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_vite_development_origin_can_preflight_verification_post():
    response = client.options(
        "/verification/verify",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_untrusted_browser_origin_is_not_reflected():
    response = client.options(
        "/verification/verify",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers
