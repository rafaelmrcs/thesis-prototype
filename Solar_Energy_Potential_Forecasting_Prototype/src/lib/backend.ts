function normalizeBackendUrl(rawUrl: string): string {
  const trimmed = rawUrl.trim().replace(/\/+$/, "");
  if (!trimmed) return "";

  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }

  if (trimmed.startsWith("//")) {
    return `https:${trimmed}`;
  }

  return `https://${trimmed}`;
}

export function getBackendCandidates(): string[] {
  const envUrl = import.meta.env.VITE_BACKEND_URL as string | undefined;
  const normalizedEnv = envUrl ? normalizeBackendUrl(envUrl) : "";

  if (normalizedEnv) {
    return [normalizedEnv];
  }

  const host = (window.location.hostname || "").toLowerCase();
  const isLocalHost = host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0";

  if (isLocalHost) {
    return [`http://${host}:8501`, "http://localhost:8501", "http://127.0.0.1:8501"];
  }

  return [];
}

export async function fetchBackendJson<T>(path: string, init?: RequestInit): Promise<T> {
  const backendCandidates = getBackendCandidates();

  if (backendCandidates.length === 0) {
    throw new Error("Backend URL is not configured. Set VITE_BACKEND_URL to your deployed backend URL.");
  }

  let lastError = "Backend request failed";
  for (const backendUrl of backendCandidates) {
    try {
      const response = await fetch(`${backendUrl}${path}`, init);
      if (response.ok) return (await response.json()) as T;
      lastError = `${path} failed with status ${response.status}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : "Network error";
    }
  }

  throw new Error(lastError);
}
