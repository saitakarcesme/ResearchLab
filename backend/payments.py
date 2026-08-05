from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from backend.cloud_gpu import CloudGpuManager
from backend.db import Database


class PaymentConfigurationError(RuntimeError):
    pass


class StripeCheckoutService:
    base_url = "https://api.stripe.com/v1"

    def __init__(self, database: Database, cloud: CloudGpuManager):
        self.database = database
        self.cloud = cloud
        self.secret_key = os.getenv("STRIPE_SECRET_KEY", "").strip()
        self.webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
        self.public_app_url = os.getenv("AUTORESEARCH_PUBLIC_APP_URL", "http://localhost:3000").rstrip("/")

    @property
    def enabled(self) -> bool:
        return self.secret_key.startswith("sk_")

    def public_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": "Stripe Checkout",
            "currency": "USD",
            "webhook_configured": self.webhook_secret.startswith("whsec_"),
        }

    def _stripe(self, path: str, *, method: str = "GET", form: list[tuple[str, str]] | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise PaymentConfigurationError("Add STRIPE_SECRET_KEY to enable Apple Pay checkout")
        data = urllib.parse.urlencode(form or []).encode("utf-8") if form is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method,
            headers={"Authorization": f"Bearer {self.secret_key}", "Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
            raise PaymentConfigurationError(f"Stripe checkout failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise PaymentConfigurationError(f"Stripe is unreachable: {exc.reason}") from exc

    def create(self, account_id: str, offer_id: str, hours: float) -> dict[str, Any]:
        account = self.database.get_cloud_account(account_id)
        if not account:
            raise KeyError(account_id)
        offer = next((item for item in self.cloud.offers(account_id) if item["id"] == offer_id), None)
        if not offer:
            raise ValueError("The selected GPU offer is no longer available")
        rental_cost = round(float(offer["hourly_price_usd"]) * float(hours), 2)
        if rental_cost < 0.50:
            raise ValueError("Apple Pay checkout must total at least $0.50; increase the rental duration")
        credit = self.cloud.account_credit(account_id)
        if credit is not None and credit + 1e-9 < rental_cost:
            raise ValueError(f"Vast account credit is ${credit:.2f}; at least ${rental_cost:.2f} is required before checkout")
        charge_total = rental_cost
        order = self.database.create_cloud_rental_order({
            "account_id": account_id,
            "offer_id": offer_id,
            "gpu_name": offer["gpu"],
            "hours": hours,
            "hourly_price_usd": offer["hourly_price_usd"],
            "total_usd": charge_total,
        })
        cents = int(round(charge_total * 100))
        session = self._stripe("/checkout/sessions", method="POST", form=[
            ("mode", "payment"),
            ("payment_method_types[0]", "card"),
            ("line_items[0][price_data][currency]", "usd"),
            ("line_items[0][price_data][unit_amount]", str(cents)),
            ("line_items[0][price_data][product_data][name]", f"{offer['gpu']} · {hours:g} hour GPU rental"),
            ("line_items[0][quantity]", "1"),
            ("client_reference_id", order["id"]),
            ("metadata[order_id]", order["id"]),
            ("success_url", f"{self.public_app_url}/gpus?checkout=success&order={order['id']}"),
            ("cancel_url", f"{self.public_app_url}/gpus?checkout=cancelled&order={order['id']}"),
            ("expires_at", str(int(time.time()) + 1800)),
        ])
        return self.database.update_cloud_rental_order(order["id"], {
            "stripe_session_id": session.get("id"),
            "checkout_url": session.get("url"),
        }) or order

    def sync(self, order_id: str) -> dict[str, Any]:
        order = self.database.get_cloud_rental_order(order_id)
        if not order:
            raise KeyError(order_id)
        if order["status"] != "checkout_open" or not order.get("stripe_session_id"):
            return order
        session = self._stripe(f"/checkout/sessions/{urllib.parse.quote(str(order['stripe_session_id']))}")
        if session.get("payment_status") == "paid":
            return self.fulfill(order_id, str(session.get("id") or order["stripe_session_id"]))
        if session.get("status") == "expired":
            return self.database.update_cloud_rental_order(order_id, {"status": "cancelled"}) or order
        return order

    def fulfill(self, order_id: str, stripe_session_id: str) -> dict[str, Any]:
        if not self.database.claim_cloud_rental_order(order_id, stripe_session_id):
            return self.database.get_cloud_rental_order(order_id) or {}
        order = self.database.get_cloud_rental_order(order_id) or {}
        try:
            instance = self.cloud.rent(
                order["account_id"], order["offer_id"],
                name=f"ResearchLab {order['gpu_name']}", max_hours=float(order["hours"]),
            )
            return self.database.update_cloud_rental_order(order_id, {
                "status": "active", "cloud_instance_id": instance["id"], "error": None,
            }) or order
        except Exception as exc:
            return self.database.update_cloud_rental_order(order_id, {
                "status": "failed", "error": str(exc)[:600],
            }) or order

    def handle_webhook(self, raw: bytes, signature: str) -> dict[str, Any] | None:
        if not self.webhook_secret:
            raise PaymentConfigurationError("STRIPE_WEBHOOK_SECRET is not configured")
        fields = dict(part.split("=", 1) for part in signature.split(",") if "=" in part)
        timestamp, received = fields.get("t", ""), fields.get("v1", "")
        if not timestamp or abs(time.time() - int(timestamp)) > 300:
            raise ValueError("Expired Stripe webhook signature")
        expected = hmac.new(
            self.webhook_secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + raw, hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, received):
            raise ValueError("Invalid Stripe webhook signature")
        event = json.loads(raw)
        if event.get("type") != "checkout.session.completed":
            return None
        session = event.get("data", {}).get("object", {})
        order_id = session.get("metadata", {}).get("order_id") or session.get("client_reference_id")
        if not order_id or session.get("payment_status") != "paid":
            return None
        return self.fulfill(str(order_id), str(session.get("id") or ""))
