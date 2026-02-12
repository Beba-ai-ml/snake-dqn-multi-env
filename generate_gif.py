#!/usr/bin/env python3
"""Generate a gameplay GIF of the trained DQN agent playing Snake."""

import sys
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from snake_dqn_8_multi.game import SnakeGame
from snake_dqn_8_multi.model import DQN

# --- Config ---
GRID_SIZE = 8
MODEL_PATH = "checkpoints_8_multi/session_4.pth"
OUTPUT_PATH = "gameplay.gif"
MAX_EPISODES = 50  # try up to this many episodes to find a good one
CELL_SIZE = 30  # pixels per cell
FPS = 64
FRAME_DURATION = 1000 // FPS  # ms per frame (~16ms)

# Colors
BG_COLOR = (30, 30, 30)
GRID_COLOR = (50, 50, 50)
HEAD_COLOR = (50, 200, 50)
BODY_COLOR = (80, 160, 80)
FOOD_COLOR = (220, 50, 50)


def render_frame(game: SnakeGame) -> Image.Image:
    """Render the game state as a PIL image."""
    size = GRID_SIZE * CELL_SIZE
    img = Image.new("RGB", (size, size), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        pos = i * CELL_SIZE
        draw.line([(pos, 0), (pos, size - 1)], fill=GRID_COLOR, width=1)
        draw.line([(0, pos), (size - 1, pos)], fill=GRID_COLOR, width=1)

    # Draw snake body (skip head)
    for idx, (x, y) in enumerate(game.snake):
        if idx == 0:
            continue
        x0, y0 = x * CELL_SIZE + 1, y * CELL_SIZE + 1
        x1, y1 = x0 + CELL_SIZE - 2, y0 + CELL_SIZE - 2
        draw.rectangle([x0, y0, x1, y1], fill=BODY_COLOR)

    # Draw head
    hx, hy = game.snake[0]
    x0, y0 = hx * CELL_SIZE + 1, hy * CELL_SIZE + 1
    x1, y1 = x0 + CELL_SIZE - 2, y0 + CELL_SIZE - 2
    draw.rectangle([x0, y0, x1, y1], fill=HEAD_COLOR)

    # Draw food
    if game.food:
        fx, fy = game.food
        x0, y0 = fx * CELL_SIZE + 1, fy * CELL_SIZE + 1
        x1, y1 = x0 + CELL_SIZE - 2, y0 + CELL_SIZE - 2
        draw.rectangle([x0, y0, x1, y1], fill=FOOD_COLOR)

    return img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = DQN(output_size=4, grid_size=GRID_SIZE).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    # Checkpoint may be a raw state_dict or a dict with 'model' key
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    game = SnakeGame(grid_size=GRID_SIZE, render=False)
    max_score = GRID_SIZE * GRID_SIZE - 3  # win condition

    best_score = 0
    best_frames = []

    for ep in range(1, MAX_EPISODES + 1):
        state = game.reset()
        frames = [render_frame(game)]
        done = False
        score = 0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_t)
            action = int(torch.argmax(q_values).item())

            state, reward, done, score = game.step(action)
            frames.append(render_frame(game))

        win = score >= max_score
        print(f"Episode {ep}: score={score} win={win} death={game.last_death_reason} frames={len(frames)}")

        if score > best_score:
            best_score = score
            best_frames = frames
            print(f"  -> New best! score={best_score}")

        if win:
            print(f"WIN found at episode {ep}! Score: {score}")
            best_frames = frames
            best_score = score
            break

    if not best_frames:
        print("No episodes completed. Something is wrong.")
        return

    # No subsampling — keep all frames at target FPS
    frame_duration = FRAME_DURATION

    print(f"Saving GIF with {len(best_frames)} frames, score={best_score}...")

    # Save as GIF
    best_frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=best_frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
    )

    import os
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"GIF saved to {OUTPUT_PATH} ({size_mb:.2f} MB, {len(best_frames)} frames)")

    if size_mb > 5:
        print("WARNING: GIF is over 5MB. Reducing quality...")
        # Reduce by taking every 3rd frame and making cells smaller
        reduced = best_frames[::3]
        reduced[0].save(
            OUTPUT_PATH,
            save_all=True,
            append_images=reduced[1:],
            duration=frame_duration * 3,
            loop=0,
            optimize=True,
        )
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"Reduced GIF: {size_mb:.2f} MB, {len(reduced)} frames")


if __name__ == "__main__":
    main()
