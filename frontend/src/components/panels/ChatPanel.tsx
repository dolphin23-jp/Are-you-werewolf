import { Fragment, useEffect, useRef, useState } from "react";
import { controlDiscussion, passDiscussionTurn, sendChat } from "../../api/client";
import type { ChatChannel, ChatMessage } from "../../api/types";
import { useGameStore } from "../../state/gameStore";
import { TypingIndicator } from "../common/TypingIndicator";

const MESSAGE_REFERENCE_RE = /(【m\d+(?:への回答)?】|\[m\d+\]|m\d+)/g;

function MessageBody({ content, onReference }: { content: string; onReference: (id: string) => void }) {
  return content.split(MESSAGE_REFERENCE_RE).map((part, index) => {
    const id = part.match(/m\d+/)?.[0];
    if (!id) return <Fragment key={index}>{part}</Fragment>;
    return (
      <button key={index} type="button" className="chat-message__reference" onClick={() => onReference(id)}>
        {part}
      </button>
    );
  });
}

export function ChatPanel() {
  const view = useGameStore((s) => s.view);
  const sessionId = useGameStore((s) => s.sessionId);
  const playerNames = useGameStore((s) => s.playerNames);
  const refreshView = useGameStore((s) => s.refreshView);
  const setError = useGameStore((s) => s.setError);
  const selectedSpeakerId = useGameStore((s) => s.selectedSpeakerId);
  const setSelectedSpeakerId = useGameStore((s) => s.setSelectedSpeakerId);

  const [tab, setTab] = useState<ChatChannel>("public");
  const [selectedDay, setSelectedDay] = useState<number | "all">("all");
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [passing, setPassing] = useState(false);
  const [controlling, setControlling] = useState(false);
  const [replyingTo, setReplyingTo] = useState<ChatMessage[]>([]);
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const [newMessageCount, setNewMessageCount] = useState(0);
  const [waitRemaining, setWaitRemaining] = useState(0);
  const chatLogRef = useRef<HTMLDivElement>(null);
  const previousMessageCount = useRef(0);
  const autoPassedToken = useRef<string | null>(null);

  useEffect(() => {
    const log = chatLogRef.current;
    if (!log || !view) return;
    const count = view.public_chat.length + view.private_chat.length;
    const added = Math.max(0, count - previousMessageCount.current);
    const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
    if (nearBottom || previousMessageCount.current === 0) {
      log.scrollTo?.({ top: log.scrollHeight, behavior: "smooth" });
      setNewMessageCount(0);
    } else if (added > 0) {
      setNewMessageCount((current) => current + added);
    }
    previousMessageCount.current = count;
  }, [view]);

  useEffect(() => {
    setWaitRemaining(view?.speech_wait_remaining_seconds ?? 0);
    if (!view?.awaiting_your_speech) return;
    const timer = window.setInterval(
      () => setWaitRemaining((remaining) => Math.max(0, remaining - 1)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [view?.awaiting_your_speech, view?.speech_wait_remaining_seconds]);

  // Auto-pass at most once per wait. Keying off the server's token rather than a
  // boolean matters because the 2.5s poll re-runs this effect: a wait that stays at
  // zero would otherwise fire a pass request on every single poll.
  const waitToken = view?.speech_wait_token ?? null;
  useEffect(() => {
    if (!view?.awaiting_your_speech || view.discussion_paused || waitRemaining > 0 || !sessionId) return;
    if (waitToken === null || autoPassedToken.current === waitToken) return;
    autoPassedToken.current = waitToken;
    void passDiscussionTurn(sessionId)
      .then(refreshView)
      .catch((error: unknown) => {
        setError(error instanceof Error ? error.message : "パスに失敗しました");
      });
  }, [refreshView, sessionId, setError, view?.awaiting_your_speech, view?.discussion_paused, waitRemaining, waitToken]);

  if (!view || !sessionId) return null;

  const isAlive = view.players.find((p) => p.player_id === view.your_player_id)?.alive ?? false;
  const canUseWolfChat = view.your_role === "werewolf";
  const canUseFreemasonChat = view.your_role === "freemason";
  const canSendPublic = isAlive && view.phase === "discussion";
  const canSendPrivate =
    isAlive && (tab === "wolf" ? canUseWolfChat : tab === "freemason" ? canUseFreemasonChat : false);
  const canSend = tab === "public" ? canSendPublic : canSendPrivate;

  const channelMessages =
    tab === "public" ? view.public_chat : view.private_chat.filter((m) => m.channel === tab);
  const availableDays = Array.from(
    new Set([
      ...channelMessages.map((message) => message.day),
      ...view.vote_history.map((vote) => vote.day),
      ...(tab === "public" && view.day > 0 ? [view.day] : []),
    ]),
  ).sort((a, b) => a - b);
  const messages = channelMessages.filter(
    (message) =>
      (selectedDay === "all" || message.day === selectedDay) &&
      (tab !== "public" || selectedSpeakerId === null || message.author_id === selectedSpeakerId),
  );
  const visibleVotes = view.vote_history.filter(
    (vote) => selectedDay === "all" || vote.day === selectedDay,
  );
  const voteGroups = Array.from(new Set(visibleVotes.map((vote) => `${vote.day}:${vote.round}`))).map(
    (key) => {
      const [day, round] = key.split(":").map(Number);
      const votes = visibleVotes.filter((vote) => vote.day === day && vote.round === round);
      const counts = new Map<string, string[]>();
      votes.forEach((vote) => counts.set(vote.target_id, [...(counts.get(vote.target_id) ?? []), vote.voter_id]));
      return { day, round, counts: [...counts.entries()].sort((a, b) => b[1].length - a[1].length) };
    },
  );
  const nameClass = (playerId: string) => {
    const role = view.co_declarations.find((claim) => claim.player_id === playerId)?.claimed_role;
    const confirmedWhite = (view.public_result_claims ?? []).some(
      (claim) => claim.target_id === playerId && !claim.is_werewolf,
    );
    return `chat-message__author player-name--${role ?? (confirmedWhite ? "white" : "gray")}`;
  };
  const publicDeathDay = (player: (typeof view.players)[number]) =>
    player.death_day === null
      ? null
      : player.death_cause === "night_death" || player.death_cause === "first_victim"
        ? player.death_day + 1
        : player.death_day;

  const handleSend = async () => {
    const content = draft.trim();
    if (!content || !canSend || sending) return;
    setSending(true);
    try {
      await sendChat(
        sessionId,
        content,
        tab,
        replyingTo[0]?.message_id,
        replyingTo[0]?.content.slice(0, 160),
        replyingTo.slice(1).map((message) => message.message_id),
      );
      setDraft("");
      setReplyingTo([]);
      await refreshView();
    } catch (e) {
      setError(e instanceof Error ? e.message : "発言に失敗しました");
    } finally {
      setSending(false);
    }
  };

  const jumpToMessage = (messageId: string) => {
    const element = document.getElementById(`chat-${messageId}`);
    element?.scrollIntoView?.({ behavior: "smooth", block: "center" });
    setHighlightedId(messageId);
    window.setTimeout(() => setHighlightedId(null), 1600);
  };

  const selectReply = (message: ChatMessage) => {
    setReplyingTo([message]);
    jumpToMessage(message.message_id);
  };

  const toggleQuestionReply = (message: ChatMessage) => {
    setReplyingTo((current) =>
      current.some((item) => item.message_id === message.message_id)
        ? current.filter((item) => item.message_id !== message.message_id)
        : [...current, message],
    );
  };

  // Passing has its own in-flight flag: sharing `sending` made a pass disable the
  // send button and vice versa, even though they are independent actions.
  const handlePass = async () => {
    if (passing) return;
    setPassing(true);
    try {
      await passDiscussionTurn(sessionId);
      await refreshView();
    } catch (error) {
      setError(error instanceof Error ? error.message : "パスに失敗しました");
    } finally {
      setPassing(false);
    }
  };

  const handleDiscussionControl = async (action: "pause" | "resume" | "step") => {
    if (controlling) return;
    setControlling(true);
    try {
      await controlDiscussion(sessionId, action);
      await refreshView();
    } catch (error) {
      setError(error instanceof Error ? error.message : "AI議論の操作に失敗しました");
    } finally {
      setControlling(false);
    }
  };

  return (
    <div className="panel chat-panel">
      <div className="chat-panel__tabs">
        <button
          className={tab === "public" ? "tab tab--active" : "tab"}
          onClick={() => setTab("public")}
        >
          全体チャット
        </button>
        {canUseWolfChat && (
          <button className={tab === "wolf" ? "tab tab--active" : "tab"} onClick={() => setTab("wolf")}>
            人狼チャット
          </button>
        )}
        {canUseFreemasonChat && (
          <button
            className={tab === "freemason" ? "tab tab--active" : "tab"}
            onClick={() => setTab("freemason")}
          >
            共有者チャット
          </button>
        )}
      </div>

      <div className="chat-panel__days" aria-label="日付別ログ">
        <button className={selectedDay === "all" ? "tab tab--active" : "tab"} onClick={() => setSelectedDay("all")}>全日程</button>
        {availableDays.map((day) => (
          <button key={day} className={selectedDay === day ? "tab tab--active" : "tab"} onClick={() => setSelectedDay(day)}>
            {day}日目
          </button>
        ))}
      </div>

      {tab === "public" && selectedSpeakerId && (
        <button className="speaker-filter" onClick={() => setSelectedSpeakerId(null)}>
          {playerNames[selectedSpeakerId] ?? selectedSpeakerId}の発言のみ ×
        </button>
      )}

      <div
        className="chat-panel__log"
        ref={chatLogRef}
        onScroll={(event) => {
          const element = event.currentTarget;
          if (element.scrollHeight - element.scrollTop - element.clientHeight < 80) {
            setNewMessageCount(0);
          }
        }}
      >
        {tab === "public" && selectedDay !== "all" && (
          <>
            {view.players.filter((player) => publicDeathDay(player) === selectedDay && player.death_cause === "first_victim").map((player) => (
              <p className="system-message" key={`first-${player.player_id}`}>初日犠牲者（{player.name}）が死亡しました</p>
            ))}
            {view.players.filter((player) => publicDeathDay(player) === selectedDay && player.death_cause === "night_death").length > 0 && (
              <p className="system-message">{view.players.filter((player) => publicDeathDay(player) === selectedDay && player.death_cause === "night_death").map((player) => player.name).join("、")}が死体となって発見されました</p>
            )}
            {selectedDay > 1 && selectedDay <= view.day && view.players.every(
              (player) => publicDeathDay(player) !== selectedDay || player.death_cause !== "night_death",
            ) && (
              <p className="system-message">昨夜は誰も死体となって発見されませんでした</p>
            )}
            {view.players.filter((player) => player.death_day === selectedDay && player.death_cause === "executed").map((player) => (
              <p className="system-message" key={`executed-${player.player_id}`}>投票の結果、{player.name}が処刑されました</p>
            ))}
          </>
        )}
        {messages.length === 0 && <p className="chat-panel__empty">まだ発言はありません</p>}
        {messages.map((m, index) => {
          const source = m.reply_to
            ? channelMessages.find((candidate) => candidate.message_id === m.reply_to)
            : undefined;
          // A quote visually starts a new conversational unit. Keeping the author on
          // that line prevents a consecutive post from looking like somebody else's.
          const grouped = index > 0 && messages[index - 1].author_id === m.author_id && !m.reply_to;
          const showDay = selectedDay === "all" && (index === 0 || messages[index - 1].day !== m.day);
          return (
            <Fragment key={m.message_id}>
              {showDay && <div className="chat-day-separator">{m.day}日目</div>}
              <div
                id={`chat-${m.message_id}`}
                className={[
                  "chat-message",
                  m.author_id === view.your_player_id ? "chat-message--you" : "",
                  grouped ? "chat-message--grouped" : "",
                  highlightedId === m.message_id ? "chat-message--highlighted" : "",
                ].filter(Boolean).join(" ")}
              >
                <div className="chat-message__author-column">
                  {!grouped && (
                    <button
                      className={nameClass(m.author_id)}
                      onClick={() => setSelectedSpeakerId(m.author_id)}
                    >
                      {playerNames[m.author_id] ?? m.author_id}
                    </button>
                  )}
                </div>
                <div className="chat-message__content">
                  {m.reply_to && (
                    <button
                      type="button"
                      className="chat-message__quote"
                      onClick={() => jumpToMessage(m.reply_to!)}
                    >
                      <strong>{source ? playerNames[source.author_id] ?? source.author_id : m.reply_to}</strong>
                      <span>{m.quote || source?.content.slice(0, 160) || "元発言を表示"}</span>
                    </button>
                  )}
                  {(m.references ?? []).length > 0 && (
                    <div className="chat-message__references">
                      同時返信:
                      {(m.references ?? []).map((messageId) => (
                        <button key={messageId} type="button" onClick={() => jumpToMessage(messageId)}>
                          [{messageId}]
                        </button>
                      ))}
                    </div>
                  )}
                  <span className="chat-message__body">
                    <MessageBody content={m.content} onReference={jumpToMessage} />
                  </span>
                  <button
                    className="chat-message__reply-button"
                    type="button"
                    onClick={() => selectReply(m)}
                    aria-label={`${m.message_id}に返信`}
                  >
                    返信
                  </button>
                </div>
              </div>
            </Fragment>
          );
        })}
        {tab === "public" && visibleVotes.length > 0 && (
          <section className="vote-history">
            <h4>投票記録</h4>
            {voteGroups.map((group) => (
              <div key={`${group.day}-${group.round}`} className="vote-history__round">
                <strong>{group.day}日目 R{group.round}</strong>
                {group.counts.map(([targetId, voters], rank) => (
                  <p key={targetId}>{rank + 1}. {playerNames[targetId] ?? targetId} — {voters.length}票<br /><small>投票者: {voters.map((id) => playerNames[id] ?? id).join("、")}</small></p>
                ))}
              </div>
            ))}
          </section>
        )}
      </div>

      {newMessageCount > 0 && (
        <button
          className="new-message-badge"
          type="button"
          onClick={() => {
            const log = chatLogRef.current;
            log?.scrollTo({ top: log.scrollHeight, behavior: "smooth" });
            setNewMessageCount(0);
          }}
        >
          新着 {newMessageCount} 件
        </button>
      )}

      <div className="chat-panel__input">
        {view.phase === "discussion" && isAlive && (
          <div className="discussion-controls">
            {view.discussion_paused ? (
              <>
                <strong>AI議論は一時停止中です</strong>
                <button className="btn" type="button" disabled={controlling} onClick={() => void handleDiscussionControl("step")}>
                  次の1発言
                </button>
                <button className="btn" type="button" disabled={controlling} onClick={() => void handleDiscussionControl("resume")}>
                  自動進行を再開
                </button>
              </>
            ) : (
              <button className="btn" type="button" disabled={controlling} onClick={() => void handleDiscussionControl("pause")}>
                AI議論を一時停止
              </button>
            )}
          </div>
        )}
        {/* Shown to living players too: while a round is segmented and paced, the
            log can sit still for a while, and "how far along is this day" is the
            question that answers. It was previously visible only after you died. */}
        {view.discussion_progress.total > 0 && (
          <small className="discussion-progress">
            AI議論進行: {view.discussion_progress.spoken}/{view.discussion_progress.total}
          </small>
        )}
        {view.awaiting_your_speech && !view.discussion_paused && (
          <div className="speech-waiting">
            <strong>あなたの発言を待っています（残り {waitRemaining} 秒）</strong>
            <button className="btn" type="button" disabled={passing} onClick={() => void handlePass()}>
              パス
            </button>
          </div>
        )}
        {view.pending_questions.length > 0 && (
          <div className="pending-questions">
            <strong>あなたへの未回答質問</strong>
            {view.pending_questions.map((question) => {
              const source = view.public_chat.find(
                (message) => message.message_id === question.source_message_id,
              );
              return (
                <label
                  key={question.source_message_id}
                  className="pending-question"
                >
                  <input
                    type="checkbox"
                    checked={Boolean(source && replyingTo.some((item) => item.message_id === source.message_id))}
                    onChange={() => source && toggleQuestionReply(source)}
                    disabled={!source}
                  />
                  [{question.source_message_id}] {playerNames[question.asker] ?? question.asker}: 「{question.question}」
                </label>
              );
            })}
          </div>
        )}
        {(view.typing_player_ids ?? []).length > 0 && (
          <TypingIndicator label={`${(view.typing_player_ids ?? []).map((id) => playerNames[id] ?? id).join("、")}が書き込み中…`} />
        )}
        <div className="chat-composer">
        {replyingTo.length > 0 && (
          <div className="reply-preview">
            <span>返信先 {replyingTo.map((message) => `[${message.message_id}] ${playerNames[message.author_id] ?? message.author_id}`).join("、")}</span>
            <button type="button" onClick={() => setReplyingTo([])} aria-label="返信をキャンセル">×</button>
          </div>
        )}
        <div className="chat-composer__row"><textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && e.keyCode !== 229) {
              e.preventDefault();
              void handleSend();
            }
          }}
          disabled={!canSend || sending}
          maxLength={1500}
          placeholder={canSend ? "発言を入力..." : "現在は発言できません"}
        />
        <button className="btn" disabled={!canSend || sending || !draft.trim()} onClick={() => void handleSend()}>
          送信
        </button>
        </div>
        </div>
      </div>
    </div>
  );
}
