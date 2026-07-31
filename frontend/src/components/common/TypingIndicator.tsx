export function TypingIndicator({ label }: { label: string }) {
  return (
    <div className="typing-indicator">
      <span className="typing-indicator__dot" />
      <span className="typing-indicator__dot" />
      <span className="typing-indicator__dot" />
      {label}
    </div>
  );
}
