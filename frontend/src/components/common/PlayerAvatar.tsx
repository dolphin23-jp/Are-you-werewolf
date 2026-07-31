interface PlayerAvatarProps {
  name: string;
  alive: boolean;
  isYou?: boolean;
}

function colorForName(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 55%, 55%)`;
}

export function PlayerAvatar({ name, alive, isYou }: PlayerAvatarProps) {
  return (
    <div
      className={`player-avatar${alive ? "" : " player-avatar--dead"}${isYou ? " player-avatar--you" : ""}`}
      style={{ backgroundColor: colorForName(name) }}
      title={name}
    >
      {name.slice(0, 1)}
    </div>
  );
}
