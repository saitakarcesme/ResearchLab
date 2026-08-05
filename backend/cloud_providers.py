from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


class CloudProviderError(RuntimeError):
    pass


def _request(url: str, api_key: str, *, method: str = "GET", body: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None) -> Any:
    request_headers = {"Accept": "application/json", **(headers or {})}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise CloudProviderError(f"Provider request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise CloudProviderError(f"Provider is unreachable: {exc.reason}") from exc


@dataclass(slots=True)
class CloudOffer:
    id: str
    name: str
    gpu: str
    vram_gb: float | None
    region: str
    hourly_price_usd: float
    available: bool
    raw: dict[str, Any]
    gpu_count: int = 1
    reliability: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "gpu": self.gpu, "vram_gb": self.vram_gb, "region": self.region, "hourly_price_usd": self.hourly_price_usd, "available": self.available, "gpu_count": self.gpu_count, "reliability": self.reliability}


class ShadeformProvider:
    id = "shadeform"
    base_url = "https://api.shadeform.ai/v1"
    billing_url = "https://platform.shadeform.ai/billing"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-KEY": self.api_key}

    def offers(self) -> list[dict[str, Any]]:
        payload = _request(f"{self.base_url}/instances/types?available=true", self.api_key, headers=self.headers)
        rows = payload.get("instance_types", payload.get("data", payload if isinstance(payload, list) else []))
        offers: list[CloudOffer] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            configuration = item.get("configuration") if isinstance(item.get("configuration"), Mapping) else {}
            price = item.get("hourly_price") or item.get("price_per_hour") or item.get("price")
            if price is None:
                continue
            cloud = str(item.get("cloud") or item.get("shade_cloud") or "")
            type_id = str(item.get("shade_instance_type") or item.get("instance_type") or item.get("id") or "")
            availability = item.get("availability") if isinstance(item.get("availability"), list) else [{"region": item.get("region"), "available": item.get("available", True)}]
            for location in availability:
                if not isinstance(location, Mapping) or not location.get("available", True):
                    continue
                region = str(location.get("region") or item.get("region") or "")
                offer_id = "|".join((cloud, region, type_id))
                offers.append(CloudOffer(offer_id, type_id, str(configuration.get("gpu_type") or type_id), _number(configuration.get("vram_per_gpu_in_gb")), region, float(price), True, item))
        return [offer.as_dict() for offer in sorted(offers, key=lambda value: value.hourly_price_usd)]

    def create(self, offer_id: str, *, name: str, ssh_key_id: str, max_spend_usd: float) -> dict[str, Any]:
        cloud, region, instance_type = offer_id.split("|", 2)
        body: dict[str, Any] = {"cloud": cloud, "region": region, "shade_instance_type": instance_type, "shade_cloud": True, "name": name, "ssh_key_id": ssh_key_id}
        if max_spend_usd > 0:
            body["auto_delete"] = {"spend_threshold": str(max_spend_usd)}
        payload = _request(f"{self.base_url}/instances/create", self.api_key, method="POST", body=body, headers=self.headers)
        return self._instance(payload.get("instance", payload))

    def get(self, instance_id: str) -> dict[str, Any]:
        payload = _request(f"{self.base_url}/instances/{urllib.parse.quote(instance_id)}/info", self.api_key, headers=self.headers)
        return self._instance(payload.get("instance", payload))

    def terminate(self, instance_id: str) -> None:
        _request(f"{self.base_url}/instances/{urllib.parse.quote(instance_id)}/delete", self.api_key, method="POST", headers=self.headers)

    @staticmethod
    def _instance(item: Mapping[str, Any]) -> dict[str, Any]:
        ssh = item.get("ssh") if isinstance(item.get("ssh"), Mapping) else {}
        return {"external_id": str(item.get("shade_instance_id") or item.get("id")), "status": str(item.get("status") or "creating").lower(), "host": item.get("ip") or item.get("public_ip") or ssh.get("host"), "port": int(ssh.get("port") or item.get("ssh_port") or 22), "username": ssh.get("username") or item.get("ssh_user") or "root"}


class RunPodProvider:
    id = "runpod"
    base_url = "https://rest.runpod.io/v1"
    billing_url = "https://www.runpod.io/console/user/billing"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def offers(self) -> list[dict[str, Any]]:
        query = "query { gpuTypes { id displayName memoryInGb secureCloud communityCloud lowestPrice(input: { gpuCount: 1 }) { stockStatus uninterruptablePrice availableGpuCounts } } }"
        payload = _request(f"https://api.runpod.io/graphql?api_key={urllib.parse.quote(self.api_key)}", self.api_key, method="POST", body={"query": query})
        data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
        rows = data.get("gpuTypes", [])
        offers = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            lowest = item.get("lowestPrice") if isinstance(item.get("lowestPrice"), Mapping) else {}
            price = lowest.get("uninterruptablePrice")
            gpu_id = str(item.get("id") or item.get("gpuTypeId") or "")
            if not gpu_id or price is None:
                continue
            stock = str(lowest.get("stockStatus") or "None")
            if stock.lower() == "none":
                continue
            offers.append({"id": gpu_id, "name": str(item.get("displayName") or gpu_id), "gpu": str(item.get("displayName") or gpu_id), "vram_gb": _number(item.get("memoryInGb")), "region": "Best available region", "hourly_price_usd": float(price), "available": True})
        return sorted(offers, key=lambda value: value["hourly_price_usd"])

    def create(self, offer_id: str, *, name: str, image_name: str, max_spend_usd: float) -> dict[str, Any]:
        payload = _request(f"{self.base_url}/pods", self.api_key, method="POST", headers=self.headers, body={"name": name, "computeType": "GPU", "gpuTypeIds": [offer_id], "gpuCount": 1, "imageName": image_name, "containerDiskInGb": 30, "volumeInGb": 20, "volumeMountPath": "/workspace", "ports": ["22/tcp"], "supportPublicIp": True})
        return self._instance(payload)

    def get(self, instance_id: str) -> dict[str, Any]:
        return self._instance(_request(f"{self.base_url}/pods/{urllib.parse.quote(instance_id)}", self.api_key, headers=self.headers))

    def terminate(self, instance_id: str) -> None:
        _request(f"{self.base_url}/pods/{urllib.parse.quote(instance_id)}", self.api_key, method="DELETE", headers=self.headers)

    @staticmethod
    def _instance(item: Mapping[str, Any]) -> dict[str, Any]:
        mappings = item.get("portMappings") if isinstance(item.get("portMappings"), Mapping) else {}
        ssh_port = mappings.get("22") or mappings.get(22) or 22
        return {"external_id": str(item.get("id")), "status": str(item.get("desiredStatus") or item.get("status") or "creating").lower(), "host": item.get("publicIp") or item.get("public_ip"), "port": int(ssh_port), "username": "root"}


class VastProvider:
    id = "vast"
    base_url = "https://console.vast.ai/api/v0"
    billing_url = "https://cloud.vast.ai/billing/"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def offers(self) -> list[dict[str, Any]]:
        query = {
            "verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False},
            "num_gpus": {"eq": 1}, "gpu_ram": {"gte": 20 * 1024}, "compute_cap": {"gte": 750},
            "limit": 60, "order": [["dph_total", "asc"]],
        }
        if self.api_key:
            payload = _request(f"{self.base_url}/bundles/", self.api_key, method="POST", headers=self.headers, body={"type": "ondemand", **query})
        else:
            encoded = urllib.parse.quote(json.dumps(query, separators=(",", ":")))
            payload = _request(f"{self.base_url}/bundles/?q={encoded}", "")
        rows = payload.get("offers", payload if isinstance(payload, list) else [])
        if isinstance(rows, Mapping):
            rows = [rows]
        offers: list[CloudOffer] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            offer_id = item.get("id") or item.get("ask_contract_id")
            price = item.get("dph_total") or item.get("dph_base")
            if offer_id is None or price is None:
                continue
            gpu = str(item.get("gpu_name") or "NVIDIA GPU").replace("_", " ")
            ram_mb = _number(item.get("gpu_ram"))
            location = str(item.get("geolocation") or item.get("country") or "Remote")
            reliability = _number(item.get("reliability2") or item.get("reliability"))
            offers.append(
                CloudOffer(
                    id=str(offer_id),
                    name=f"{gpu} · {location}",
                    gpu=gpu,
                    vram_gb=round(ram_mb / 1024, 1) if ram_mb and ram_mb > 256 else ram_mb,
                    region=location,
                    hourly_price_usd=float(price),
                    available=True,
                    raw=item,
                    gpu_count=int(item.get("num_gpus") or 1),
                    reliability=reliability,
                )
            )
        return [offer.as_dict() for offer in sorted(offers, key=lambda value: value.hourly_price_usd)]

    @classmethod
    def public_offers(cls) -> list[dict[str, Any]]:
        return cls("").offers()

    def credit_balance(self) -> float | None:
        payload = _request(f"{self.base_url}/users/current/", self.api_key, headers=self.headers)
        return _number(payload.get("credit", payload.get("balance")))

    def create(self, offer_id: str, *, name: str, image_name: str, max_spend_usd: float) -> dict[str, Any]:
        payload = _request(
            f"{self.base_url}/asks/{urllib.parse.quote(offer_id)}/",
            self.api_key,
            method="PUT",
            headers=self.headers,
            body={
                "image": image_name,
                "label": name,
                "disk": 50,
                "runtype": "ssh_direct",
                "target_state": "running",
                "cancel_unavail": True,
            },
        )
        external_id = payload.get("new_contract") or payload.get("id")
        if not external_id:
            raise CloudProviderError("Vast accepted no instance contract for this offer")
        return {"external_id": str(external_id), "status": "creating", "host": None, "port": 22, "username": "root"}

    def get(self, instance_id: str) -> dict[str, Any]:
        payload = _request(f"{self.base_url}/instances/{urllib.parse.quote(instance_id)}/", self.api_key, headers=self.headers)
        item = payload.get("instance", payload)
        return {
            "external_id": str(item.get("id") or instance_id),
            "status": str(item.get("actual_status") or item.get("status") or "creating").lower(),
            "host": item.get("ssh_host") or item.get("public_ipaddr") or item.get("public_ip"),
            "port": int(item.get("ssh_port") or 22),
            "username": "root",
        }

    def terminate(self, instance_id: str) -> None:
        _request(f"{self.base_url}/instances/{urllib.parse.quote(instance_id)}/", self.api_key, method="DELETE", headers=self.headers)


def provider_client(provider: str, api_key: str):
    if provider == "shadeform":
        return ShadeformProvider(api_key)
    if provider == "runpod":
        return RunPodProvider(api_key)
    if provider == "vast":
        return VastProvider(api_key)
    raise CloudProviderError("Unsupported cloud GPU provider")


def provider_billing_url(provider: str) -> str:
    return provider_client(provider, "unused").billing_url


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
