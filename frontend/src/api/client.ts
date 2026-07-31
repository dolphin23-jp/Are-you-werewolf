import type { CreateGameResponse, DebugView, GameView } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
export const WS_BASE = import.meta.env.VITE_WS_BASE_URL ?? "ws://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function createGame(humanName: string, seed?: number): Promise<CreateGameResponse> {
  return request("/api/games", {
    method: "POST",
    body: JSON.stringify({ human_name: humanName, seed }),
  });
}

export function startGame(sessionId: string): Promise<void> {
  return request(`/api/games/${sessionId}/start`, { method: "POST" });
}

export function getView(sessionId: string, playerId: string): Promise<GameView> {
  return request(`/api/games/${sessionId}/view?player_id=${encodeURIComponent(playerId)}`);
}

export function getDebugView(sessionId: string): Promise<DebugView> {
  return request(`/api/games/${sessionId}/debug`);
}

export function sendChat(
  sessionId: string,
  content: string,
  channel: "public" | "wolf" | "freemason" = "public",
): Promise<void> {
  return request(`/api/games/${sessionId}/chat`, {
    method: "POST",
    body: JSON.stringify({ content, channel }),
  });
}

export function castVote(sessionId: string, targetId: string): Promise<void> {
  return request(`/api/games/${sessionId}/vote`, {
    method: "POST",
    body: JSON.stringify({ target_id: targetId }),
  });
}

export function submitNightAction(
  sessionId: string,
  actionType: "divine" | "guard" | "attack",
  targetId: string,
): Promise<void> {
  return request(`/api/games/${sessionId}/night-action`, {
    method: "POST",
    body: JSON.stringify({ action_type: actionType, target_id: targetId }),
  });
}

export function declareCo(sessionId: string, claimedRole: string): Promise<void> {
  return request(`/api/games/${sessionId}/co`, {
    method: "POST",
    body: JSON.stringify({ claimed_role: claimedRole }),
  });
}

export function startDiscussion(sessionId: string): Promise<void> {
  return request(`/api/games/${sessionId}/start-discussion`, { method: "POST" });
}

export function endDiscussion(sessionId: string): Promise<void> {
  return request(`/api/games/${sessionId}/end-discussion`, { method: "POST" });
}

export function startNight(sessionId: string): Promise<void> {
  return request(`/api/games/${sessionId}/start-night`, { method: "POST" });
}

export function resolveNight(sessionId: string): Promise<void> {
  return request(`/api/games/${sessionId}/resolve-night`, { method: "POST" });
}

export function resolveVotes(sessionId: string): Promise<void> {
  return request(`/api/games/${sessionId}/resolve-votes`, { method: "POST" });
}
