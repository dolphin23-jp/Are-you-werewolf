import { ROLE_LABELS } from "../../api/types";
import { useGameStore } from "../../state/gameStore";

export function COPanel() {
  const view = useGameStore((s) => s.view);
  const playerNames = useGameStore((s) => s.playerNames);

  if (!view) return null;

  return (
    <div className="panel co-panel">
      <h3>公開情報まとめ</h3>
      <p className="co-panel__hint">議論中に本人が明かしたCOを自動で記録します</p>
      {view.co_declarations.length > 0 ? (
        <ul>
          {view.co_declarations.map((c, i) => (
            <li key={i}>
              {playerNames[c.player_id] ?? c.player_id}: {ROLE_LABELS[c.claimed_role]}CO
            </li>
          ))}
        </ul>
      ) : (
        <p className="co-panel__empty">まだ誰もCOしていません</p>
      )}
      {(view.public_result_claims ?? []).length > 0 && (
        <section className="public-results">
          <h4>公開された判定</h4>
          <ul>
            {(view.public_result_claims ?? []).map((claim, index) => (
              <li key={`${claim.claimant_id}-${claim.target_id}-${claim.day}-${index}`}>
                {claim.day}日目 {playerNames[claim.claimant_id] ?? claim.claimant_id}（{claim.result_type === "seer" ? "占い" : "霊媒"}）→ {playerNames[claim.target_id] ?? claim.target_id}：{claim.is_werewolf ? "黒" : "白"}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
