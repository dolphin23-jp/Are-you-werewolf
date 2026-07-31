import { useEffect } from "react";
import { ROLE_LABELS } from "../../api/types";
import { useGameStore } from "../../state/gameStore";

export function DebugPanel() {
  const debugMode = useGameStore((s) => s.debugMode);
  const debug = useGameStore((s) => s.debug);
  const toggleDebug = useGameStore((s) => s.toggleDebug);
  const refreshDebug = useGameStore((s) => s.refreshDebug);

  useEffect(() => {
    if (debugMode) void refreshDebug();
  }, [debugMode, refreshDebug]);

  return (
    <div className="panel debug-panel">
      <div className="debug-panel__header">
        <h3>デバッグ / 観戦モード</h3>
        <button className="btn btn--small" onClick={toggleDebug}>
          {debugMode ? "隠す" : "表示する"}
        </button>
      </div>
      {debugMode && debug && (
        <div className="debug-panel__body">
          <section>
            <h4>全員の役職</h4>
            <ul>
              {debug.players.map((p) => (
                <li key={p.player_id}>
                  {p.name}: {ROLE_LABELS[p.role]} ({p.team}) {!p.alive && "☠"}
                </li>
              ))}
            </ul>
          </section>
          <section>
            <h4>人狼/共有チャット</h4>
            <ul>
              {debug.chat_log
                .filter((m) => m.channel !== "public")
                .map((m, i) => (
                  <li key={i}>
                    [{m.channel}] {m.author_id}: {m.content}
                  </li>
                ))}
            </ul>
          </section>
          <section>
            <h4>占い/霊媒結果</h4>
            <ul>
              {debug.divine_records.map((r, i) => (
                <li key={`d${i}`}>
                  Day{r.day} 占い: {r.seer_id} → {r.target_id} = {r.is_werewolf ? "人狼" : "人狼でない"}
                </li>
              ))}
              {debug.medium_records.map((r, i) => (
                <li key={`m${i}`}>
                  Day{r.day} 霊媒: {r.medium_id} → {r.target_id} = {r.is_werewolf ? "人狼だった" : "人狼でなかった"}
                </li>
              ))}
            </ul>
          </section>
          <section>
            <h4>投票履歴</h4>
            <ul>
              {debug.vote_records.map((v, i) => (
                <li key={i}>
                  Day{v.day} R{v.round}: {v.voter_id} → {v.target_id}
                </li>
              ))}
            </ul>
          </section>
        </div>
      )}
    </div>
  );
}
