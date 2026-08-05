"use client";

import { Check, ExternalLink, LoaderCircle, Minus, Plus, RefreshCw, Server, Trash2, X } from "lucide-react";
import { QRCodeSVG } from "qrcode.react";
import { FaApplePay } from "react-icons/fa";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createCloudAccount,
  createCloudCheckout,
  deleteCloudAccount,
  getCloudPaymentConfig,
  getCloudRentalOrder,
  listCloudAccounts,
  listCloudInstances,
  listCloudOffers,
  listPublicVastOffers,
  refreshCloudGpu,
  terminateCloudGpu,
} from "@/lib/api";
import type { CloudAccount, CloudGpuInstance, CloudOffer, CloudRentalOrder } from "@/lib/types";

const initialAccount = {
  provider: "vast" as const,
  name: "Vast.ai",
  api_key: "",
  budget_usd: 100,
  workspace_path: "/workspace/researchlab",
};

function money(value: number): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2 }).format(value);
}

export function CloudGpuRental() {
  const [accounts, setAccounts] = useState<CloudAccount[]>([]);
  const [instances, setInstances] = useState<CloudGpuInstance[]>([]);
  const [offers, setOffers] = useState<CloudOffer[]>([]);
  const [accountId, setAccountId] = useState("");
  const [selectedOffer, setSelectedOffer] = useState("");
  const [hours, setHours] = useState(1);
  const [form, setForm] = useState(initialAccount);
  const [showConnect, setShowConnect] = useState(false);
  const [paymentEnabled, setPaymentEnabled] = useState(false);
  const [order, setOrder] = useState<CloudRentalOrder | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [nextAccounts, nextInstances, payment, publicOffers] = await Promise.all([
      listCloudAccounts(), listCloudInstances(), getCloudPaymentConfig(), listPublicVastOffers(),
    ]);
    const vastAccounts = nextAccounts.filter((account) => account.provider === "vast");
    setAccounts(nextAccounts);
    setInstances(nextInstances);
    setPaymentEnabled(payment.enabled);
    setOffers((current) => current.length ? current : publicOffers);
    setSelectedOffer((current) => current || publicOffers[0]?.id || "");
    setAccountId((current) => current || vastAccounts[0]?.id || "");
  }, []);

  const loadOffers = useCallback(async (nextAccountId: string) => {
    setBusy("offers");
    try {
      const values = await listCloudOffers(nextAccountId);
      setOffers(values);
      setSelectedOffer((current) => values.some((offer) => offer.id === current) ? current : values[0]?.id ?? "");
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Live Vast.ai offers are unavailable.");
    } finally {
      setBusy(null);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void load().catch((reason) => setError(reason instanceof Error ? reason.message : "Cloud GPU data is unavailable."));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  useEffect(() => {
    if (!accountId) return;
    const timer = window.setTimeout(() => void loadOffers(accountId), 0);
    return () => window.clearTimeout(timer);
  }, [accountId, loadOffers]);

  useEffect(() => {
    if (!order || !["checkout_open", "provisioning"].includes(order.status)) return;
    const poll = async () => {
      try {
        const next = await getCloudRentalOrder(order.id);
        setOrder(next);
        if (next.status === "active") await load();
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Payment status is unavailable.");
      }
    };
    const timer = window.setInterval(() => void poll(), 2500);
    return () => window.clearInterval(timer);
  }, [load, order]);

  const chosen = useMemo(() => offers.find((offer) => offer.id === selectedOffer) ?? null, [offers, selectedOffer]);
  const estimated = chosen ? chosen.hourly_price_usd * hours : 0;
  const paymentMinimumMet = estimated + 1e-9 >= 0.5;

  async function connect(event: React.FormEvent) {
    event.preventDefault();
    setBusy("connect");
    setError(null);
    try {
      const account = await createCloudAccount(form);
      setForm(initialAccount);
      setShowConnect(false);
      await load();
      setAccountId(account.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Vast.ai could not be connected.");
    } finally {
      setBusy(null);
    }
  }

  async function checkout() {
    if (!chosen) return;
    if (!accountId) {
      setShowConnect(true);
      setError("Connect your Vast.ai API key once; the selected offer and duration will stay here.");
      return;
    }
    setBusy("checkout");
    setError(null);
    try {
      setOrder(await createCloudCheckout({ account_id: accountId, offer_id: chosen.id, hours }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Apple Pay checkout could not be created.");
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="cloud-rental">
      <div className="cloud-rental-heading">
        <div><strong>Vast.ai marketplace</strong><span>Verified, single-GPU machines · live hourly prices</span></div>
        <div className="cloud-rental-heading-actions">
          {accountId ? <button type="button" aria-label="Refresh GPU offers" onClick={() => void loadOffers(accountId)}><RefreshCw className={busy === "offers" ? "spin" : ""} size={14} /></button> : null}
          <button className="text-button" type="button" onClick={() => setShowConnect((value) => !value)}>{showConnect ? "Close" : accountId ? "Provider" : "Connect Vast.ai"}</button>
        </div>
      </div>

      {showConnect || !accountId ? (
        <form className="vast-connect" onSubmit={connect}>
          <div><strong>Connect your Vast.ai account</strong><span>The API key stays encrypted on this Windows computer.</span></div>
          <label><span>Vast.ai API key</span><input required type="password" autoComplete="off" value={form.api_key} onChange={(event) => setForm({ ...form, api_key: event.target.value })} placeholder="Paste a scoped API key" /></label>
          <label><span>Rental limit</span><input required type="number" min="1" step="1" value={form.budget_usd} onChange={(event) => setForm({ ...form, budget_usd: Number(event.target.value) })} /></label>
          <button className="primary-button compact" disabled={busy === "connect"}>{busy === "connect" ? <LoaderCircle className="spin" size={14} /> : null}Connect</button>
        </form>
      ) : null}

      {offers.length || accountId ? (
        <>
          <div className="gpu-offer-grid" aria-busy={busy === "offers"}>
            {offers.slice(0, 8).map((offer) => (
              <button
                className={`gpu-offer-card${selectedOffer === offer.id ? " gpu-offer-card-selected" : ""}`}
                key={offer.id}
                type="button"
                onClick={() => setSelectedOffer(offer.id)}
              >
                <div className="gpu-offer-visual" aria-hidden="true"><i /><i /><i /></div>
                <span className="gpu-offer-provider">Vast.ai verified</span>
                <strong>{offer.gpu}</strong>
                <span>{offer.vram_gb ? `${Number(offer.vram_gb.toFixed(1))} GB VRAM` : "NVIDIA CUDA"} · {offer.region}</span>
                <div><b>{money(offer.hourly_price_usd)}</b><small>/ hour</small>{selectedOffer === offer.id ? <Check size={14} /> : null}</div>
              </button>
            ))}
            {busy === "offers" && !offers.length ? <div className="gpu-offers-loading"><LoaderCircle className="spin" size={18} /> Reading live inventory</div> : null}
          </div>

          {chosen ? (
            <div className="gpu-checkout-bar">
              <div className="gpu-checkout-selection"><span>Selected</span><strong>{chosen.gpu}</strong><small>{money(chosen.hourly_price_usd)} / hour</small></div>
              <div className="hour-stepper" aria-label="Rental duration">
                <button type="button" aria-label="Remove one hour" disabled={hours <= 1} onClick={() => setHours((value) => Math.max(1, value - 1))}><Minus size={14} /></button>
                <div><strong>{hours}</strong><span>{hours === 1 ? "hour" : "hours"}</span></div>
                <button type="button" aria-label="Add one hour" onClick={() => setHours((value) => Math.min(720, value + 1))}><Plus size={14} /></button>
              </div>
              <div className="gpu-checkout-total"><span>Estimated</span><strong>{money(estimated)}</strong></div>
              <button className="apple-pay-button" type="button" aria-label="Pay with Apple Pay" disabled={!paymentEnabled || !paymentMinimumMet || busy != null} onClick={() => void checkout()}>{busy === "checkout" ? <LoaderCircle className="spin" size={17} /> : <FaApplePay size={43} aria-hidden="true" />}</button>
            </div>
          ) : null}
          {!paymentEnabled ? <p className="payment-setup-note">Live prices are active. Add the Stripe server key to activate Apple Pay checkout.</p> : null}
          {paymentEnabled && !paymentMinimumMet && chosen ? <p className="payment-setup-note">Increase the duration to {Math.ceil(0.5 / chosen.hourly_price_usd)} hours to reach the $0.50 checkout minimum.</p> : null}
        </>
      ) : null}

      {instances.length ? (
        <div className="cloud-instances">
          {instances.map((instance) => (
            <div className="cloud-instance" key={instance.id}>
              <Server size={15} />
              <div><strong>{instance.name}</strong><span>{instance.status} · {money(instance.estimated_spend_usd)} used / {money(instance.max_spend_usd)} limit</span></div>
              <code>{instance.host ?? "Provisioning…"}</code>
              <button type="button" aria-label="Refresh rental" onClick={() => void refreshCloudGpu(instance.id).then(load)}><RefreshCw size={13} /></button>
              {!['terminated', 'deleted'].includes(instance.status) ? <button type="button" aria-label="Terminate rental" onClick={() => void terminateCloudGpu(instance.id).then(load)}><Trash2 size={13} /></button> : null}
            </div>
          ))}
        </div>
      ) : null}

      <div className="cloud-account-links">
        {accounts.filter((account) => account.provider === "vast").map((account) => (
          <div key={account.id}><span>{account.name}</span><a href={account.billing_url} target="_blank" rel="noreferrer">Vast credit <ExternalLink size={12} /></a><button type="button" aria-label={`Disconnect ${account.name}`} onClick={() => void deleteCloudAccount(account.id).then(load).catch((reason) => setError(reason.message))}><Trash2 size={12} /></button></div>
        ))}
      </div>
      {error ? <p className="inline-error" role="alert">{error}</p> : null}

      {order ? (
        <div className="checkout-qr-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && setOrder(null)}>
          <section className="checkout-qr" role="dialog" aria-modal="true" aria-labelledby="checkout-title">
            <button className="checkout-close" type="button" aria-label="Close checkout" onClick={() => setOrder(null)}><X size={16} /></button>
            {order.status === "active" ? (
              <><div className="checkout-success"><Check size={24} /></div><h2 id="checkout-title">GPU is starting</h2><p>Payment confirmed. The new Vast.ai machine is being added to your GPU sources.</p></>
            ) : order.status === "failed" ? (
              <><h2 id="checkout-title">Provisioning needs attention</h2><p>{order.error}</p></>
            ) : (
              <>
                <span className="checkout-eyebrow">Apple Pay · phone checkout</span>
                <h2 id="checkout-title">Scan to rent {order.gpu_name}</h2>
                <p>{Number(order.hours.toFixed(2))} hour · {money(order.total_usd)}. The GPU starts only after Stripe confirms payment.</p>
                {order.checkout_url ? <div className="checkout-qr-code"><QRCodeSVG value={order.checkout_url} size={220} bgColor="#ffffff" fgColor="#050505" level="M" /></div> : <LoaderCircle className="spin" size={24} />}
                {order.checkout_url ? <a href={order.checkout_url} target="_blank" rel="noreferrer">Open checkout on this device <ExternalLink size={13} /></a> : null}
                <small>{order.status === "provisioning" ? "Payment received · provisioning" : "Waiting for payment"}</small>
              </>
            )}
          </section>
        </div>
      ) : null}
    </section>
  );
}
