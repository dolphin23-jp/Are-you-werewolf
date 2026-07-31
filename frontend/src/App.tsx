import { useGameSocket } from "./api/useGameSocket";
import { useGamePolling } from "./api/useGamePolling";
import { WelcomeScreen } from "./components/screens/WelcomeScreen";
import { RoleRevealScreen } from "./components/screens/RoleRevealScreen";
import { MainGameScreen } from "./components/screens/MainGameScreen";
import { useGameStore } from "./state/gameStore";

export default function App() {
  const screen = useGameStore((s) => s.screen);
  useGameSocket();
  useGamePolling();

  switch (screen) {
    case "welcome":
      return <WelcomeScreen />;
    case "role-reveal":
      return <RoleRevealScreen />;
    case "main":
      return <MainGameScreen />;
    default:
      return null;
  }
}
