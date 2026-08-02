export type Phase =
  | "waiting"
  | "night"
  | "dawn"
  | "discussion"
  | "voting"
  | "runoff"
  | "vote_result"
  | "game_over";

export type RoleName =
  | "villager"
  | "werewolf"
  | "madman"
  | "seer"
  | "medium"
  | "hunter"
  | "fox"
  | "freemason";

export type Team = "village" | "werewolf" | "fox";

export type DeathCause = "executed" | "attacked" | "cursed" | "night_death" | "first_victim";

export type ChatChannel = "public" | "wolf" | "freemason";

export interface PublicPlayer {
  player_id: string;
  name: string;
  alive: boolean;
  death_cause: DeathCause | null;
  death_day: number | null;
}

export interface ChatMessage {
  message_id: string;
  author_id: string;
  content: string;
  channel: ChatChannel;
  day: number;
  reply_to: string | null;
  quote: string | null;
}

export interface PendingQuestion {
  asker: string;
  target: string;
  question: string;
  source_message_id: string;
  day: number;
}

export interface DivineResult {
  seer_id: string;
  target_id: string;
  day: number;
  is_werewolf: boolean;
}

export interface MediumResult {
  medium_id: string;
  target_id: string;
  day: number;
  is_werewolf: boolean;
}

export interface CoDeclarationRecord {
  player_id: string;
  claimed_role: RoleName;
  day: number;
}

export interface PublicResultClaimRecord {
  claimant_id: string;
  result_type: "seer" | "medium";
  target_id: string;
  is_werewolf: boolean;
  day: number;
}

export interface VoteRecordEntry {
  voter_id: string;
  target_id: string;
  day: number;
  round: number;
}

export interface GameView {
  session_id: string;
  phase: Phase;
  day: number;
  vote_round: number;
  /** In a runoff, the only players who may be voted for. Empty in an
   * ordinary round, meaning anyone alive is eligible. */
  runoff_candidates: string[];
  your_player_id: string;
  your_role: RoleName | null;
  allies: string[];
  players: PublicPlayer[];
  public_chat: ChatMessage[];
  private_chat: ChatMessage[];
  pending_questions: PendingQuestion[];
  awaiting_your_speech: boolean;
  speech_wait_remaining_seconds: number;
  speech_wait_token: string | null;
  discussion_progress: { spoken: number; total: number };
  your_divine_results: DivineResult[];
  your_medium_results: MediumResult[];
  co_declarations: CoDeclarationRecord[];
  public_result_claims?: PublicResultClaimRecord[];
  vote_history: VoteRecordEntry[];
  first_victim_id?: string | null;
  has_voted_current_round?: boolean;
  typing_player_ids?: string[];
  winner: Team | null;
  victory_reason: string;
  is_draw: boolean;
  /** Only present when your_role is "werewolf": only the alpha may submit
   * the night attack. */
  is_alpha_wolf?: boolean;
}

export interface DebugPlayer extends PublicPlayer {
  role: RoleName;
  team: Team;
  is_human: boolean;
}

export interface DebugView {
  session_id: string;
  phase: Phase;
  day: number;
  vote_round: number;
  runoff_candidates: string[];
  players: DebugPlayer[];
  chat_log: ChatMessage[];
  divine_records: DivineResult[];
  medium_records: MediumResult[];
  guard_records: { hunter_id: string; target_id: string; day: number }[];
  attack_records: {
    wolf_id: string;
    target_id: string;
    day: number;
    succeeded: boolean;
  }[];
  vote_records: VoteRecordEntry[];
  death_records: { player_id: string; cause: DeathCause; day: number }[];
  co_declarations: CoDeclarationRecord[];
  winner: Team | null;
  victory_reason: string;
  is_draw: boolean;
}

export interface CreateGameResponse {
  session_id: string;
  human_player_id: string;
  player_names: Record<string, string>;
}

export const ROLE_LABELS: Record<RoleName, string> = {
  villager: "村人",
  werewolf: "人狼",
  madman: "狂人",
  seer: "占い師",
  medium: "霊媒師",
  hunter: "狩人",
  fox: "妖狐",
  freemason: "共有者",
};

export const ROLE_DEFINITIONS_JA: Record<RoleName, string> = {
  villager: "議論と投票で人狼を追放する",
  werewolf: "夜に仲間と相談し襲撃先を決定する",
  madman: "人狼陣営。特殊能力はないが人狼の勝利を助ける",
  seer: "毎夜1人を占い人狼か判定する",
  medium: "処刑者の正体を自動で知る",
  hunter: "2日目夜から毎夜1人を護衛する",
  fox: "占われると呪殺される。生き残れば勝利する",
  freemason: "相方と確定白として協調する",
};

export const PHASE_LABELS: Record<Phase, string> = {
  waiting: "待機中",
  night: "夜",
  dawn: "夜明け",
  discussion: "議論中",
  voting: "投票中",
  runoff: "決選投票",
  vote_result: "処刑結果",
  game_over: "終了",
};
