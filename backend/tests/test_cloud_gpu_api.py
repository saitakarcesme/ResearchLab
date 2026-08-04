from __future__ import annotations

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app


def test_cloud_account_secret_is_encrypted_and_not_returned(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/cloud/accounts",
            json={
                "provider": "shadeform",
                "name": "Personal Shadeform",
                "api_key": "secret-test-key",
                "budget_usd": 20,
                "ssh_key_id": "ssh-key-id",
                "workspace_path": "/workspace/researchlab",
            },
        )
        assert response.status_code == 201
        account = response.json()
        assert "api_key" not in account
        assert "secret_ref" not in account
        assert account["billing_url"].startswith("https://")

        listed = client.get("/api/cloud/accounts").json()
        assert listed == [account]
        assert list((settings.data_dir / "secrets").glob("*.bin"))

        deleted = client.delete(f"/api/cloud/accounts/{account['id']}")
        assert deleted.status_code == 204
        assert not list((settings.data_dir / "secrets").glob("*.bin"))
