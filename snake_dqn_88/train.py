import argparse
import json
import os
from collections import deque
from typing import List

import torch

from snake_dqn_88.agent import Agent
from snake_dqn_88.game import SnakeGame


def train(
    episodes: int = 100,
    render_every: int = 0,
    fps: int = 8,
    grid_size: int = 8,
    gamma: float = 0.99,
    lr: float = 1e-4,
    batch_size: int = 256,
    epsilon_start: float = 0.01,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 1.0,
    save_every: int = 0,
    save_dir: str = "checkpoints_88",
    load_from: str | None = None,
    session_id: int = 1,
) -> None:
    agent = Agent(
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    os.makedirs(save_dir, exist_ok=True)
    session_ckpt = os.path.join(save_dir, f"session_{session_id}.pth")
    session_meta = os.path.join(save_dir, f"session_{session_id}.json")
    episodes_completed = 0

    if load_from:
        try:
            meta = agent.load_checkpoint(load_from)
            episodes_completed = int(meta.get("episodes_trained", 0))
            print(f"Loaded checkpoint from: {load_from} (episodes: {episodes_completed})")
        except Exception as exc:
            print(f"Failed to load weights ({exc}). Starting from scratch.")
    elif os.path.exists(session_ckpt):
        try:
            meta = agent.load_checkpoint(session_ckpt)
            print(f"Loaded session {session_id} from: {session_ckpt}")
            episodes_completed = int(meta.get("episodes_trained", 0))
            print(f"  -> episodes completed so far: {episodes_completed}")
            print(f"  -> epsilon on resume: {agent.epsilon:.3f}")
        except Exception as exc:
            print(f"Failed to load existing session ({exc}). Starting from scratch.")
            episodes_completed = 0

    render_requested = render_every > 0
    game = SnakeGame(render=render_requested, fps=fps, grid_size=grid_size)
    game.reset()

    scores: List[int] = []
    mean_window_20 = deque(maxlen=20)
    mean_window_100 = deque(maxlen=100)

    try:
        for episode in range(1, episodes + 1):
            # render only every N episodes (or never)
            if render_requested:
                game.render_enabled = episode % render_every == 0
            else:
                game.render_enabled = False

            state = game.reset()
            done = False
            loss = None

            while not done:
                action = agent.act(state)
                next_state, reward, done, score = game.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay()
                state = next_state

            agent.update_epsilon()
            scores.append(score)
            mean_window_20.append(score)
            mean_window_100.append(score)

            mean_20 = sum(mean_window_20) / len(mean_window_20)
            mean_100 = sum(mean_window_100) / len(mean_window_100)
            loss_display = f", loss={loss:.4f}" if loss is not None else ""
            total_ep = episodes_completed + episode
            death = game.last_death_reason or "-"
            print(
                f"[Ep {total_ep:04d}] score={score} mean(20)={mean_20:.2f} mean(100)={mean_100:.2f} "
                f"epsilon={agent.epsilon:.3f}{loss_display} death={death}"
            )

            if save_every > 0 and episode % save_every == 0:
                total_ep = episodes_completed + episode
                meta = {
                    "session_id": session_id,
                    "episodes_trained": total_ep,
                    "epsilon": agent.epsilon,
                }
                agent.save_checkpoint(session_ckpt, meta=meta)
                with open(session_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                print(f"  -> saved model: {session_ckpt} (episodes in session: {total_ep})")
    finally:
        game.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN CNN training (default grid 8x8)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render-every", type=int, default=0, help="0=off, 1=every episode, N=every Nth episode")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epsilon-start", type=float, default=0.01)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=0, help="save weights every N episodes (0=off)")
    parser.add_argument("--save-dir", type=str, default="checkpoints_88", help="checkpoint output directory")
    parser.add_argument("--load-from", type=str, default=None, help="path to .pth file to load weights from")
    parser.add_argument("--session-id", type=int, default=1, help="session number (saved as session_<id>.pth)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        episodes=args.episodes,
        render_every=args.render_every,
        fps=args.fps,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        save_every=args.save_every,
        save_dir=args.save_dir,
        load_from=args.load_from,
        session_id=args.session_id,
        grid_size=args.grid_size,
    )
