import type { LocalModel } from "./types";

const MODEL_PROVIDER_PREFIX = /^(?:ollama|huggingface):/;

export function modelDisplayName(modelId: string | null | undefined): string | null {
  return modelId?.replace(MODEL_PROVIDER_PREFIX, "").replace(/:latest$/, "") ?? null;
}

export function modelProviderLabel(provider: LocalModel["provider"]): string {
  return provider === "huggingface" ? "Hugging Face" : "Ollama";
}
