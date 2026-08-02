import type { GameView } from "../../api/types";

export function makeView(overrides: Partial<GameView> = {}): GameView {
  return {
    session_id: "s1",
    phase: "discussion",
    day: 1,
    vote_round: 1,
    runoff_candidates: [],
    your_player_id: "p0",
    your_role: "villager",
    allies: [],
    players: [
      { player_id: "p0", name: "Taro", alive: true, death_cause: null, death_day: null },
      { player_id: "p1", name: "Hanako", alive: true, death_cause: null, death_day: null },
      { player_id: "p2", name: "Jiro", alive: true, death_cause: null, death_day: null },
      { player_id: "p3", name: "Saburo", alive: false, death_cause: "executed", death_day: 1 },
    ],
    public_chat: [
      {
        message_id: "m1",
        author_id: "p1",
        content: "こんにちは",
        channel: "public",
        day: 1,
        reply_to: null,
        quote: null,
      },
    ],
    private_chat: [],
    pending_questions: [],
    awaiting_your_speech: false,
    speech_wait_remaining_seconds: 0,
    speech_wait_token: null,
    discussion_progress: { spoken: 0, total: 0 },
    discussion_paused: false,
    your_divine_results: [],
    your_medium_results: [],
    co_declarations: [],
    freemason_partner_claims: [],
    vote_history: [],
    winner: null,
    victory_reason: "",
    is_draw: false,
    ...overrides,
  };
}
