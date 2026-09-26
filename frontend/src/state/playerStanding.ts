import type { GameView } from "../api/types";

/** The colour a player's name is shown in: their standing role claim, else
 * "white" when every published verdict about them is white, else "gray".
 *
 * Any single white claim used to be enough -- a fake seer's white, even
 * alongside a black from the other seer -- which presented a contested claim
 * as settled. */
export function nameStanding(view: GameView, playerId: string): string {
  const role = view.co_declarations.find((claim) => claim.player_id === playerId)?.claimed_role;
  if (role) return role;
  const verdicts = (view.public_result_claims ?? []).filter(
    (claim) => claim.target_id === playerId,
  );
  const allWhite = verdicts.length > 0 && verdicts.every((claim) => !claim.is_werewolf);
  return allWhite ? "white" : "gray";
}
