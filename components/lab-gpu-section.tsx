"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, ExternalLink, LoaderCircle, RefreshCw, Server, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  listCloudAccounts,
  listCloudInstances,
  listCloudOffers,
  refreshCloudGpu,
  rentCloudGpu,
  terminateCloudGpu,
} from "@/lib/api";
import type { CloudAccount, CloudGpuInstance, CloudOffer, GpuSource, GpuTelemetry } from "@/lib/types";

const TERMINAL_INSTANCE_STATES = new Set(["terminated", "deleted", "failed"]);

function photoClass(gpu: string): string {
  return /(?:RTX\s*)?30(?:80|90)/i.test(gpu) ? "gpu-photo-3090" : "gpu-photo-datacenter";
}

function offerLabel(offer: CloudOffer): string {
  const memory = offer.vram_gb ? ` · ${offer.vram_gb} GB` : "";
  return `${offer.gpu}${memory}`;
}

export function LabGpuSection({
  sources,
  telemetry,
}: {
  sources: GpuSource[];
  telemetry: GpuTelemetry | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const [accounts, setAccounts] = useState<CloudAccount[]>([]);
  const [instances, setInstances] = useState<CloudGpuInstance[]>([]);
  const [offers, setOffers] = useState<CloudOffer[]>([]);
  const [accountId, setAccountId] = useState("");
  const [selectedOffer, setSelectedOffer] = useState("");
  const [hours, setHours] = useState(1);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadInventory = useCallback(async () => {
    const [nextAccounts, nextInstances] = await Promise.all([listCloudAccounts(), listCloudInstances()]);
    setAccounts(nextAccounts);
    setInstances(nextInstances);
    setAccountId((current) => nextAccounts.some((account) => account.id === current) ? current : nextAccounts[0]?.id ?? "");
  }, []);

  useEffect(() => {
    let active = true;
    const refresh = () => {
      void loadInventory().catch(() => {
        if (!active) return;
        setAccounts([]);
        setInstances([]);
      });
    };
    refresh();
    const timer = window.setInterval(refresh, 8000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [loadInventory]);

  useEffect(() => {
    if (!expanded || !accountId) return;
    let active = true;
    const timer = window.setTimeout(() => {
      setBusy("offers");
      void listCloudOffers(accountId)
        .then((nextOffers) => {
          if (!active) return;
          const sorted = [...nextOffers].sort((left, right) => {
            const leftIs3090 = /3090/i.test(left.gpu) ? 0 : 1;
            const rightIs3090 = /3090/i.test(right.gpu) ? 0 : 1;
            return leftIs3090 - rightIs3090 || left.hourly_price_usd - right.hourly_price_usd;
          });
          setOffers(sorted);
          setSelectedOffer((current) => sorted.some((offer) => offer.id === current) ? current : sorted[0]?.id ?? "");
          setError(null);
        })
        .catch((reason) => {
          if (active) setError(reason instanceof Error ? reason.message : "Live rental prices are unavailable.");
        })
        .finally(() => {
          if (active) setBusy(null);
        });
    }, 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [accountId, expanded]);

  const activeInstances = useMemo(
    () => instances.filter((instance) => !TERMINAL_INSTANCE_STATES.has(instance.status.toLowerCase())),
    [instances],
  );
  const localSource = sources.find((source) => source.type === "local");
  const currentOffer = offers.find((offer) => offer.id === selectedOffer) ?? null;

  async function rent() {
    if (!currentOffer || !accountId) return;
    setBusy("rent");
    setError(null);
    try {
      await rentCloudGpu({
        account_id: accountId,
        offer_id: currentOffer.id,
        name: `ResearchLab ${currentOffer.gpu}`,
        max_hours: hours,
      });
      await loadInventory();
      setExpanded(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "GPU rental could not be started.");
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="lab-gpu-section" aria-labelledby="lab-gpu-title">
      <header className="lab-gpu-heading">
        <div>
          <span id="lab-gpu-title">GPU</span>
          <small>{activeInstances.length ? `${activeInstances.length} cloud rental${activeInstances.length === 1 ? "" : "s"}` : "Local and cloud compute"}</small>
        </div>
        <button className="lab-rent-trigger" type="button" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)}>
          Rent
          <ChevronDown size={13} className={expanded ? "rotate-180" : undefined} aria-hidden="true" />
        </button>
      </header>

      <div className="lab-gpu-inventory">
        <article className="lab-gpu-device">
          <div className="lab-gpu-photo gpu-photo-3090" role="img" aria-label="NVIDIA GeForce RTX 3090" />
          <div className="lab-gpu-device-copy">
            <strong>{telemetry?.name ?? localSource?.name ?? "Local NVIDIA GPU"}</strong>
            <span>{telemetry?.available ? `${Math.round(telemetry.utilization ?? 0)}% load · ${Math.round((telemetry.memory_used_mb ?? 0) / 1024)} / ${Math.round((telemetry.memory_total_mb ?? 0) / 1024)} GB` : "Waiting for local telemetry"}</span>
          </div>
          <span className="gpu-device-state">Local</span>
        </article>

        {activeInstances.map((instance) => (
          <article className="lab-gpu-device" key={instance.id}>
            <div className={`lab-gpu-photo ${photoClass(instance.name)}`} role="img" aria-label={instance.name} />
            <div className="lab-gpu-device-copy">
              <strong>{instance.name}</strong>
              <span>${instance.hourly_price_usd.toFixed(2)}/h · ${instance.estimated_spend_usd.toFixed(2)} used</span>
            </div>
            <span className="gpu-device-state">{instance.status}</span>
            <button className="gpu-device-icon" type="button" aria-label={`Refresh ${instance.name}`} onClick={() => void refreshCloudGpu(instance.id).then(loadInventory).catch((reason) => setError(reason instanceof Error ? reason.message : "Rental status could not be refreshed."))}>
              <RefreshCw size={12} />
            </button>
            <button className="gpu-device-icon" type="button" aria-label={`Stop ${instance.name}`} onClick={() => void terminateCloudGpu(instance.id).then(loadInventory).catch((reason) => setError(reason instanceof Error ? reason.message : "Rental could not be stopped."))}>
              <Square size={11} />
            </button>
          </article>
        ))}
      </div>

      <AnimatePresence initial={false}>
        {expanded ? (
          <motion.div
            className="lab-gpu-market"
            initial={{ opacity: 0, height: 0, y: -5 }}
            animate={{ opacity: 1, height: "auto", y: 0 }}
            exit={{ opacity: 0, height: 0, y: -5 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
          >
            {!accounts.length ? (
              <div className="lab-gpu-connect">
                <div className="lab-gpu-photo gpu-photo-datacenter" role="img" aria-label="NVIDIA data center GPU" />
                <div>
                  <strong>Connect RunPod after funding</strong>
                  <span>Live RTX 3090 inventory and prices appear here as soon as the provider API key is connected.</span>
                </div>
                <a href="https://console.runpod.io/user/settings" target="_blank" rel="noreferrer">RunPod <ExternalLink size={12} /></a>
              </div>
            ) : busy === "offers" ? (
              <div className="lab-gpu-market-loading"><LoaderCircle className="spin" size={15} /> Loading live inventory</div>
            ) : offers.length ? (
              <>
                <div className="lab-gpu-offers" role="list" aria-label="Live cloud GPU offers">
                  {offers.slice(0, 12).map((offer) => (
                    <button
                      className={`lab-gpu-offer ${selectedOffer === offer.id ? "lab-gpu-offer-selected" : ""}`}
                      type="button"
                      role="listitem"
                      key={offer.id}
                      onClick={() => setSelectedOffer(offer.id)}
                    >
                      <span className={`lab-gpu-offer-photo ${photoClass(offer.gpu)}`} aria-hidden="true" />
                      <span className="lab-gpu-offer-copy"><strong>{offerLabel(offer)}</strong><small>{offer.region}</small></span>
                      <span className="lab-gpu-offer-price"><strong>${offer.hourly_price_usd.toFixed(2)}</strong><small>/ hour</small></span>
                    </button>
                  ))}
                </div>
                <div className="lab-gpu-rent-bar">
                  <label><span>Duration</span><input type="number" min="0.25" max="720" step="0.25" value={hours} onChange={(event) => setHours(Math.max(0.25, Number(event.target.value) || 1))} /></label>
                  <span className="lab-gpu-estimate">{currentOffer ? `$${(currentOffer.hourly_price_usd * hours).toFixed(2)} maximum` : "Choose a GPU"}</span>
                  <button className="primary-button compact" type="button" disabled={!currentOffer || busy != null} onClick={() => void rent()}>
                    {busy === "rent" ? <LoaderCircle className="spin" size={13} /> : <Server size={13} />}
                    Rent {hours === 1 ? "for 1 hour" : "GPU"}
                  </button>
                </div>
              </>
            ) : (
              <div className="lab-gpu-market-loading">No GPU is available from this provider right now.</div>
            )}
          </motion.div>
        ) : null}
      </AnimatePresence>
      {error ? <p className="inline-error lab-gpu-error" role="alert">{error}</p> : null}
    </section>
  );
}
