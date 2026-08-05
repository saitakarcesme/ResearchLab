from __future__ import annotations

from typing import Any

from backend.cloud_providers import CloudProviderError, provider_client
from backend.db import Database, utc_now
from backend.secret_store import SecretStore


READY_STATUSES = {"active", "running", "ready"}
TERMINAL_STATUSES = {"terminated", "deleted", "exited"}


class CloudGpuManager:
    def __init__(self, database: Database, secrets: SecretStore):
        self.database = database
        self.secrets = secrets

    @staticmethod
    def public_account(account: dict[str, Any]) -> dict[str, Any]:
        value = dict(account)
        value.pop("secret_ref", None)
        value["billing_url"] = provider_client(value["provider"], "unused").billing_url
        return value

    def _client(self, account: dict[str, Any]):
        return provider_client(account["provider"], self.secrets.get(account["secret_ref"]))

    def offers(self, account_id: str) -> list[dict[str, Any]]:
        account = self.database.get_cloud_account(account_id)
        if not account:
            raise KeyError(account_id)
        return self._client(account).offers()

    def rent(self, account_id: str, offer_id: str, *, name: str, max_hours: float) -> dict[str, Any]:
        account = self.database.get_cloud_account(account_id)
        if not account:
            raise KeyError(account_id)
        client = self._client(account)
        offer = next((value for value in client.offers() if value["id"] == offer_id and value.get("available")), None)
        if not offer:
            raise ValueError("The selected GPU offer is no longer available")
        max_spend = max(0.01, round(float(offer["hourly_price_usd"]) * max_hours, 2))
        committed_spend = sum(
            float(instance["estimated_spend_usd"] if instance["status"] in TERMINAL_STATUSES else instance["max_spend_usd"])
            for instance in self.database.list_cloud_instances()
            if instance["provider_account_id"] == account_id
        )
        if committed_spend + max_spend > float(account["budget_usd"]) + 1e-9:
            raise ValueError("This rental would exceed the provider account budget")
        settings = account.get("settings") or {}
        if account["provider"] == "shadeform":
            ssh_key_id = str(settings.get("ssh_key_id") or "").strip()
            if not ssh_key_id:
                raise ValueError("Shadeform requires an SSH key ID in the provider settings")
            remote = client.create(offer["id"], name=name, ssh_key_id=ssh_key_id, max_spend_usd=max_spend)
        else:
            remote = client.create(offer["id"], name=name, image_name=str(settings.get("image_name") or "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"), max_spend_usd=max_spend)
        return self.database.create_cloud_instance({
            "provider_account_id": account_id,
            "external_id": remote["external_id"],
            "name": name,
            "offer_id": offer["id"],
            "status": remote["status"],
            "hourly_price_usd": offer["hourly_price_usd"],
            "max_spend_usd": max_spend,
            "host": remote.get("host"),
            "port": remote.get("port", 22),
            "username": remote.get("username"),
        })

    def sync(self, instance_id: str) -> dict[str, Any]:
        instance = self.database.get_cloud_instance(instance_id)
        if not instance:
            raise KeyError(instance_id)
        if instance["status"] in TERMINAL_STATUSES:
            return instance
        account = self.database.get_cloud_account(instance["provider_account_id"])
        if not account:
            raise KeyError(instance["provider_account_id"])
        if float(instance["estimated_spend_usd"]) >= float(instance["max_spend_usd"]):
            return self.terminate(instance_id)
        remote = self._client(account).get(instance["external_id"])
        updates = {key: remote.get(key) for key in ("status", "host", "port", "username") if remote.get(key) is not None}
        source_id = instance.get("gpu_source_id")
        if str(remote.get("status", "")).lower() in READY_STATUSES and remote.get("host"):
            settings = account.get("settings") or {}
            workspace_path = str(settings.get("workspace_path") or "/workspace/researchlab")
            if source_id:
                self.database.update_gpu_source(source_id, {"host": remote["host"], "port": remote.get("port", 22), "username": remote.get("username") or "root", "workspace_path": workspace_path})
            else:
                source = self.database.create_gpu_source({"name": instance["name"], "type": "remote", "host": remote["host"], "port": remote.get("port", 22), "username": remote.get("username") or "root", "auth_method": "agent", "workspace_path": workspace_path})
                updates["gpu_source_id"] = source["id"]
        return self.database.update_cloud_instance(instance_id, updates) or instance

    def sync_all(self) -> None:
        for instance in self.database.list_cloud_instances():
            if instance["status"] in TERMINAL_STATUSES:
                continue
            try:
                self.sync(instance["id"])
            except (CloudProviderError, OSError, ValueError, KeyError):
                continue

    def terminate(self, instance_id: str) -> dict[str, Any]:
        instance = self.database.get_cloud_instance(instance_id)
        if not instance:
            raise KeyError(instance_id)
        account = self.database.get_cloud_account(instance["provider_account_id"])
        if not account:
            raise KeyError(instance["provider_account_id"])
        if instance["status"] not in TERMINAL_STATUSES:
            self._client(account).terminate(instance["external_id"])
        return self.database.update_cloud_instance(instance_id, {"status": "terminated", "terminated_at": utc_now()}) or instance
