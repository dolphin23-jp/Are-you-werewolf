interface AiBudgetNoticeProps {
  exhausted: boolean;
  costUsd: number | null;
}

export const AI_BUDGET_EXHAUSTED_MESSAGE =
  "AIの費用上限(WEREWOLF_SESSION_MAX_COST_USD)に達したため、AIの発言・投票を止めました。新しいゲームを作るか上限を見直してください。";

/** Shown once the server stopped the AI seats because this game reached its
 * spend cap. Without it the game just stops moving -- no AI speaks, votes or
 * acts at night -- and the player has no way to tell a stopped game from a
 * slow one. */
export function AiBudgetNotice({ exhausted, costUsd }: AiBudgetNoticeProps) {
  if (!exhausted) return null;
  return (
    <div className="notice notice--warning" role="alert">
      <p className="notice__message">{AI_BUDGET_EXHAUSTED_MESSAGE}</p>
      {costUsd !== null && (
        <p className="notice__detail">このゲームの推定費用: ${costUsd.toFixed(2)}</p>
      )}
    </div>
  );
}
