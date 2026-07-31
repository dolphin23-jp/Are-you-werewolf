import { useEffect, useState } from "react";
import { castVote } from "../../api/client";
import { useGameStore } from "../../state/gameStore";

export function VotePanel() {
  const view = useGameStore((s) => s.view);
  const sessionId = useGameStore((s) => s.sessionId);
  const refreshView = useGameStore((s) => s.refreshView);
  const setError = useGameStore((s) => s.setError);
  const [target, setTarget] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [votedRound, setVotedRound] = useState<number | null>(null);

  useEffect(() => {
    if (view && votedRound !== null && view.vote_round !== votedRound) {
      setVotedRound(null);
    }
  }, [view, votedRound]);

  if (!view || !sessionId || (view.phase !== "voting" && view.phase !== "runoff")) return null;

  const isAlive = view.players.find((p) => p.player_id === view.your_player_id)?.alive ?? false;
  if (!isAlive) {
    return (
      <div className="panel vote-panel">
        <p>あなたは既に脱落しています。AIの投票が進行中です...</p>
      </div>
    );
  }

  if (votedRound === view.vote_round) {
    return (
      <div className="panel vote-panel">
        <p>投票しました。他のプレイヤーの投票を待っています...</p>
      </div>
    );
  }

  const candidates = view.players.filter((p) => p.alive && p.player_id !== view.your_player_id);

  const handleVote = async () => {
    if (!target) return;
    setSubmitting(true);
    try {
      await castVote(sessionId, target);
      setVotedRound(view.vote_round);
      setTarget("");
      await refreshView();
    } catch (e) {
      setError(e instanceof Error ? e.message : "投票に失敗しました");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="panel vote-panel">
      <h3>{view.phase === "runoff" ? `決選投票 (${view.vote_round}回目)` : "投票"}</h3>
      <select value={target} onChange={(e) => setTarget(e.target.value)} disabled={submitting}>
        <option value="">投票先を選択...</option>
        {candidates.map((p) => (
          <option key={p.player_id} value={p.player_id}>
            {p.name}
          </option>
        ))}
      </select>
      <button className="btn btn--primary" disabled={!target || submitting} onClick={() => void handleVote()}>
        投票する
      </button>
    </div>
  );
}
