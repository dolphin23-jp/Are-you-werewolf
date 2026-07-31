import { PHASE_LABELS } from "../../api/types";
import type { Phase } from "../../api/types";

interface PhaseBannerProps {
  phase: Phase;
  day: number;
  connected: boolean;
}

export function PhaseBanner({ phase, day, connected }: PhaseBannerProps) {
  return (
    <div className={`phase-banner phase-banner--${phase}`}>
      <span className="phase-banner__label">
        {day}日目 ・ {PHASE_LABELS[phase]}
      </span>
      <span className={`connection-dot${connected ? " connection-dot--on" : ""}`} title={connected ? "接続中" : "未接続"} />
    </div>
  );
}
