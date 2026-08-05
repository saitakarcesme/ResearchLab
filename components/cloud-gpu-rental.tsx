"use client";

import { ExternalLink, LoaderCircle, RefreshCw, Server, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import {
  createCloudAccount,
  deleteCloudAccount,
  listCloudAccounts,
  listCloudInstances,
  listCloudOffers,
  refreshCloudGpu,
  rentCloudGpu,
  terminateCloudGpu,
} from "@/lib/api";
import type { CloudAccount, CloudGpuInstance, CloudOffer } from "@/lib/types";

const initialAccount = { provider: "shadeform" as "shadeform" | "runpod", name: "", api_key: "", budget_usd: 25, ssh_key_id: "", workspace_path: "/workspace/researchlab" };

export function CloudGpuRental() {
  const [accounts, setAccounts] = useState<CloudAccount[]>([]);
  const [instances, setInstances] = useState<CloudGpuInstance[]>([]);
  const [offers, setOffers] = useState<CloudOffer[]>([]);
  const [accountId, setAccountId] = useState("");
  const [selectedOffer, setSelectedOffer] = useState("");
  const [maxHours, setMaxHours] = useState(4);
  const [form, setForm] = useState(initialAccount);
  const [showConnect, setShowConnect] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [nextAccounts, nextInstances] = await Promise.all([listCloudAccounts(), listCloudInstances()]);
    setAccounts(nextAccounts);
    setInstances(nextInstances);
    setAccountId((current) => current || nextAccounts[0]?.id || "");
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void load().catch((reason) => setError(reason instanceof Error ? reason.message : "Cloud GPU data is unavailable.")), 0);
    return () => window.clearTimeout(timer);
  }, [load]);
  useEffect(() => {
    if (!accountId) return;
    const timer = window.setTimeout(() => {
      setBusy("offers");
      void listCloudOffers(accountId)
      .then((values) => { setOffers(values); setSelectedOffer(values[0]?.id ?? ""); setError(null); })
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Offers are unavailable."))
      .finally(() => setBusy(null));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [accountId]);

  async function connect(event: React.FormEvent) {
    event.preventDefault(); setBusy("connect"); setError(null);
    try {
      await createCloudAccount(form);
      setForm(initialAccount); setShowConnect(false); await load();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Provider could not be connected."); }
    finally { setBusy(null); }
  }

  async function rent() {
    const offer = offers.find((value) => value.id === selectedOffer);
    if (!offer) return;
    setBusy("rent"); setError(null);
    try {
      await rentCloudGpu({ account_id: accountId, offer_id: offer.id, name: `ResearchLab ${offer.gpu}`, max_hours: maxHours });
      await load();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "GPU could not be rented."); }
    finally { setBusy(null); }
  }

  return (
    <section className="cloud-rental">
      <div className="settings-section-heading"><div><strong>Rent a GPU</strong><span>Shadeform or RunPod · billed by the provider</span></div><button className="text-button" type="button" onClick={() => setShowConnect((value) => !value)}>{showConnect ? "Close" : "Connect provider"}</button></div>
      {showConnect ? (
        <form className="source-form cloud-account-form" onSubmit={connect}>
          <div className="field-grid two-columns">
            <label><span>Provider</span><select value={form.provider} onChange={(event) => setForm({ ...form, provider: event.target.value as "shadeform" | "runpod" })}><option value="shadeform">Shadeform</option><option value="runpod">RunPod</option></select></label>
            <label><span>Connection name</span><input required value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="Personal cloud" /></label>
            <label><span>API key</span><input required type="password" autoComplete="off" value={form.api_key} onChange={(event) => setForm({ ...form, api_key: event.target.value })} /></label>
            <label><span>App budget (USD)</span><input required type="number" min="1" step="1" value={form.budget_usd} onChange={(event) => setForm({ ...form, budget_usd: Number(event.target.value) })} /></label>
            {form.provider === "shadeform" ? <label><span>Shadeform SSH key ID</span><input required value={form.ssh_key_id} onChange={(event) => setForm({ ...form, ssh_key_id: event.target.value })} /></label> : null}
            <label><span>Remote workspace</span><input required value={form.workspace_path} onChange={(event) => setForm({ ...form, workspace_path: event.target.value })} /></label>
          </div>
          <p className="form-note">The key is encrypted with Windows DPAPI. Money stays with the provider; this budget is a hard ResearchLab reservation limit.</p>
          <button className="primary-button compact" disabled={busy === "connect"}>{busy === "connect" ? <LoaderCircle className="spin" size={14} /> : null}Connect</button>
        </form>
      ) : null}

      {accounts.length ? (
        <div className="cloud-rent-controls">
          <label><span>Provider account</span><select value={accountId} onChange={(event) => setAccountId(event.target.value)}>{accounts.map((account) => <option value={account.id} key={account.id}>{account.name} · ${account.budget_usd.toFixed(0)} budget</option>)}</select></label>
          <label><span>Available GPU</span><select value={selectedOffer} onChange={(event) => setSelectedOffer(event.target.value)} disabled={busy === "offers"}>{offers.map((offer) => <option value={offer.id} key={offer.id}>{offer.gpu} · {offer.region} · ${offer.hourly_price_usd.toFixed(2)}/h</option>)}</select></label>
          <label><span>Maximum hours</span><input type="number" min="0.25" max="720" step="0.25" value={maxHours} onChange={(event) => setMaxHours(Number(event.target.value))} /></label>
          <button className="primary-button compact" type="button" disabled={!selectedOffer || busy != null} onClick={() => void rent()}>{busy === "rent" ? <LoaderCircle className="spin" size={14} /> : <Server size={14} />}Rent</button>
        </div>
      ) : <p className="form-note">Connect a provider to see its live availability and prices.</p>}

      <div className="cloud-account-links">
        {accounts.map((account) => <div key={account.id}><span>{account.provider}</span><a href={account.billing_url} target="_blank" rel="noreferrer">Add provider credit <ExternalLink size={12} /></a><button type="button" aria-label={`Disconnect ${account.name}`} onClick={() => void deleteCloudAccount(account.id).then(load).catch((reason) => setError(reason.message))}><Trash2 size={12} /></button></div>)}
      </div>
      {instances.map((instance) => <div className="cloud-instance" key={instance.id}><div><strong>{instance.name}</strong><span>{instance.status} · ${instance.estimated_spend_usd.toFixed(2)} estimated / ${instance.max_spend_usd.toFixed(2)} limit</span></div><code>{instance.host ?? "Provisioning…"}</code><button type="button" aria-label="Refresh rental" onClick={() => void refreshCloudGpu(instance.id).then(load)}><RefreshCw size={13} /></button>{!["terminated", "deleted"].includes(instance.status) ? <button type="button" aria-label="Terminate rental" onClick={() => void terminateCloudGpu(instance.id).then(load)}><Trash2 size={13} /></button> : null}</div>)}
      {error ? <p className="inline-error" role="alert">{error}</p> : null}
    </section>
  );
}
