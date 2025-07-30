from __future__ import annotations
import argparse
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np
from tensorflow import keras


def load_lander_model(model_path: str) -> keras.Model:
    fichier = Path(model_path)
    if not fichier.exists():
        raise FileNotFoundError(f"Le modèle '{model_path}' est introuvable.")

    return keras.models.load_model(fichier, compile=False)

def predict_action(model: keras.Model, state: np.ndarray) -> int:
    q_values = model.predict(state[None, :], verbose=0)
    return int(np.argmax(q_values[0]))

def play_lunar_lander(model_path: str, episodes: int = 3, render: bool = True) -> None:
    model = load_lander_model(model_path)
    print(f"Modèle '{model_path}' chargé avec succès. \u2705")

    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            terminated = truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                # Le rendu est automatique quand render_mode="human".
                action = predict_action(model, np.asarray(state, dtype=np.float32))
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state

            print(f"Épisode {episode:02d} terminé : récompense totale = {total_reward:.2f}")
    finally:
        env.close()

def parse_arguments(argv: list[str]):
    parser = argparse.ArgumentParser(description="Joue à LunarLander-v3 avec un modèle Keras.")
    parser.add_argument(
        "--model",
        default="./models/LunarLander_model.keras",
        help="Chemin vers le modèle Keras (par défaut : LunarLander_model.keras)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Nombre d'épisodes à jouer (par défaut : 3)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Désactive l'affichage visuel pour accélérer l'exécution.",
    )
    return parser.parse_args(argv)


def main() -> None:  # pragma: no cover
    args = parse_arguments(sys.argv[1:])
    play_lunar_lander(args.model, episodes=args.episodes, render=not args.no_render)


if __name__ == "__main__":
    main()
