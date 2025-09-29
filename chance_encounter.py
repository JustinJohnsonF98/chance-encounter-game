"""
Chance Encounter — a grid game inspired by Optiver's puzzle

Two agents move on a grid. Meet (same cell) or cross paths (swap positions)
ends the round. Includes two modes:
  1) Player vs Random (you move Blue; Red moves randomly)
  2) Random vs Random (both agents random; use for simulations/visualization)

Controls
--------
M : toggle mode
R : reset round
O : toggle obstacles
SPACE : step once (in Random vs Random mode)
ENTER : auto-run / pause (in Random vs Random mode)
ARROWS / WASD : move player (in Player vs Random mode)
P : run Monte Carlo (1000 trials) and show stats overlay
ESC or Q : quit

Requirements
-----------
Python 3.9+
pip install pygame

Notes
-----
- "Crossing" means agents swapped squares in one turn.
- Obstacles are randomly generated; agents and exit remain reachable.
- Monte Carlo runs logical (headless) simulations for speed; obstacles are off for MC.
"""

import math
import random
import sys
from dataclasses import dataclass
from typing import List, Set, Tuple

import pygame

# --------------------------- Config ---------------------------
GRID_W, GRID_H = 12, 12         # grid size in cells
CELL = 48                       # pixel size per cell
MARGIN = 1                      # grid line width
PADDING = 220                   # right-side panel width
FPS = 60

BG = (15, 16, 20)
GRID_COLOR = (40, 44, 52)
TEXT = (230, 230, 235)
BLUE = (76, 161, 255)
RED = (255, 92, 92)
WALL = (88, 94, 110)
MEET = (90, 214, 125)

WINDOW_W, WINDOW_H = GRID_W * (CELL + MARGIN) + MARGIN + PADDING, GRID_H * (CELL + MARGIN) + MARGIN

# Monte Carlo settings
MC_TRIALS = 1000
MC_MAX_STEPS = 2000

# ------------------------ Data structures ---------------------
Vec = Tuple[int, int]

@dataclass
class Agent:
    pos: Vec
    prev: Vec
    color: Tuple[int, int, int]

    def move(self, d: Vec):
        self.prev = self.pos
        self.pos = (self.pos[0] + d[0], self.pos[1] + d[1])

# ------------------------ Helper functions --------------------

def in_bounds(p: Vec) -> bool:
    return 0 <= p[0] < GRID_W and 0 <= p[1] < GRID_H


def neighbors(p: Vec) -> List[Vec]:
    return [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]


def random_free_cell(walls: Set[Vec], exclude: Set[Vec]) -> Vec:
    while True:
        p = (random.randrange(GRID_W), random.randrange(GRID_H))
        if p not in walls and p not in exclude:
            return p


def draw_grid(surface: pygame.Surface, walls: Set[Vec]):
    for y in range(GRID_H):
        for x in range(GRID_W):
            rect = pygame.Rect(x * (CELL + MARGIN) + MARGIN,
                               y * (CELL + MARGIN) + MARGIN,
                               CELL, CELL)
            pygame.draw.rect(surface, GRID_COLOR, rect)
            if (x, y) in walls:
                pygame.draw.rect(surface, WALL, rect)


def draw_agent(surface: pygame.Surface, a: Agent):
    x, y = a.pos
    rect = pygame.Rect(x * (CELL + MARGIN) + MARGIN,
                       y * (CELL + MARGIN) + MARGIN,
                       CELL, CELL)
    pygame.draw.rect(surface, a.color, rect, border_radius=10)


def encounter(a: Agent, b: Agent) -> bool:
    return a.pos == b.pos or (a.pos == b.prev and b.pos == a.prev)


def valid_moves(p: Vec, walls: Set[Vec]) -> List[Vec]:
    opts = [d for d in neighbors(p) if in_bounds(d) and d not in walls]
    if not opts:
        return [p]
    return opts


def random_step(p: Vec, walls: Set[Vec]) -> Vec:
    return random.choice(valid_moves(p, walls))


def gen_walls(density: float = 0.12) -> Set[Vec]:
    walls: Set[Vec] = set()
    for y in range(GRID_H):
        for x in range(GRID_W):
            if random.random() < density:
                walls.add((x, y))
    return walls


def ensure_reachable(walls: Set[Vec], starts: List[Vec]):
    # Remove walls that block starting cells
    for s in starts:
        walls.discard(s)

# --------------------------- Game -----------------------------
class ChanceEncounter:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Chance Encounter — Pygame Prototype")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.big = pygame.font.SysFont("consolas", 28, bold=True)

        self.mode = "PLAYER_VS_RANDOM"  # or "RANDOM_VS_RANDOM"
        self.auto_run = False
        self.turns = 0
        self.met = False
        self.show_mc = None  # (avg_steps, meet_rate)
        self.obstacles_on = False

        self.reset_round()

    def reset_round(self):
        self.turns = 0
        self.met = False
        self.auto_run = False
        self.show_mc = None
        self.walls: Set[Vec] = set()
        if self.obstacles_on:
            self.walls = gen_walls()
        # Start agents far apart
        a_start = (0, 0)
        b_start = (GRID_W - 1, GRID_H - 1)
        ensure_reachable(self.walls, [a_start, b_start])
        self.blue = Agent(a_start, a_start, BLUE)
        self.red = Agent(b_start, b_start, RED)

    # -------------------- Simulation logic --------------------
    def step_random_vs_random(self):
        if self.met:
            return
        self.blue.prev, self.red.prev = self.blue.pos, self.red.pos
        self.blue.pos = random_step(self.blue.pos, self.walls)
        self.red.pos = random_step(self.red.pos, self.walls)
        self.turns += 1
        self.met = encounter(self.blue, self.red)

    def step_player_vs_random(self, player_dir: Vec):
        if self.met:
            return
        # Apply player move if legal
        target = (self.blue.pos[0] + player_dir[0], self.blue.pos[1] + player_dir[1])
        if in_bounds(target) and target not in self.walls:
            self.blue.prev = self.blue.pos
            self.blue.pos = target
        else:
            # illegal move: stay
            self.blue.prev = self.blue.pos
        # Red moves randomly
        self.red.prev = self.red.pos
        self.red.pos = random_step(self.red.pos, self.walls)
        self.turns += 1
        self.met = encounter(self.blue, self.red)

    # -------------------- Monte Carlo -------------------------
    def monte_carlo(self, trials: int = MC_TRIALS, max_steps: int = MC_MAX_STEPS) -> Tuple[float, float]:
        meets = 0
        total_steps = 0
        for _ in range(trials):
            a = (0, 0)
            b = (GRID_W - 1, GRID_H - 1)
            a_prev, b_prev = a, b
            steps = 0
            for _ in range(max_steps):
                a_next = random_step(a, set())
                b_next = random_step(b, set())
                steps += 1
                if a_next == b_next or (a_next == b_prev and b_next == a_prev):
                    meets += 1
                    total_steps += steps
                    break
                a_prev, b_prev = a, b
                a, b = a_next, b_next
        meet_rate = meets / trials if trials else 0.0
        avg_steps = (total_steps / meets) if meets else float('inf')
        return avg_steps, meet_rate

    # -------------------- Rendering ---------------------------
    def draw_panel(self):
        # Right panel background
        panel = pygame.Rect(GRID_W * (CELL + MARGIN) + MARGIN, 0, PADDING, WINDOW_H)
        pygame.draw.rect(self.screen, (24, 26, 32), panel)

        def blit_line(text: str, y: int, big=False, color=TEXT):
            surf = (self.big if big else self.font).render(text, True, color)
            self.screen.blit(surf, (GRID_W * (CELL + MARGIN) + 16, y))

        blit_line("Chance Encounter", 16, big=True)
        blit_line(f"Mode: {'Player vs Random' if self.mode=='PLAYER_VS_RANDOM' else 'Random vs Random'}", 60)
        blit_line(f"Turns: {self.turns}", 88)
        if self.met:
            blit_line("Encounter!", 116, big=True, color=MEET)
        if self.auto_run and self.mode == 'RANDOM_VS_RANDOM':
            blit_line("Auto-Run: ON", 148)
        blit_line(f"Obstacles: {'ON' if self.obstacles_on else 'OFF'}", 176)

        y = 220
        blit_line("Controls:", y, big=True)
        y += 34
        lines = [
            "M — toggle mode",
            "R — reset round",
            "O — toggle obstacles",
            "P — Monte Carlo stats",
            "Arrows/WASD — move",
            "Space — single step",
            "Enter — auto-run",
            "Esc/Q — quit",
        ]
        for ln in lines:
            blit_line(ln, y)
            y += 26

        if self.show_mc:
            y += 6
            blit_line("MC (1000 trials):", y, big=True)
            y += 30
            avg_steps, meet_rate = self.show_mc
            blit_line(f"Avg steps to meet: {avg_steps:.1f}", y)
            y += 26
            blit_line(f"Meet rate ≤{MC_MAX_STEPS} steps: {meet_rate*100:.1f}%", y)

    # -------------------- Main loop ---------------------------
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_m:
                        self.mode = "RANDOM_VS_RANDOM" if self.mode == "PLAYER_VS_RANDOM" else "PLAYER_VS_RANDOM"
                        self.reset_round()
                    if event.key == pygame.K_r:
                        self.reset_round()
                    if event.key == pygame.K_o:
                        self.obstacles_on = not self.obstacles_on
                        self.reset_round()
                    if event.key == pygame.K_p:
                        avg, rate = self.monte_carlo()
                        self.show_mc = (avg, rate)
                    if self.mode == "RANDOM_VS_RANDOM":
                        if event.key == pygame.K_SPACE:
                            self.step_random_vs_random()
                        if event.key == pygame.K_RETURN:
                            self.auto_run = not self.auto_run
                    elif self.mode == "PLAYER_VS_RANDOM":
                        dir_map = {
                            pygame.K_UP: (0, -1), pygame.K_w: (0, -1),
                            pygame.K_DOWN: (0, 1), pygame.K_s: (0, 1),
                            pygame.K_LEFT: (-1, 0), pygame.K_a: (-1, 0),
                            pygame.K_RIGHT: (1, 0), pygame.K_d: (1, 0),
                        }
                        if event.key in dir_map:
                            self.step_player_vs_random(dir_map[event.key])

            # Auto run logic
            if self.mode == "RANDOM_VS_RANDOM" and self.auto_run and not self.met:
                # Step at ~15 Hz so it's visible
                if pygame.time.get_ticks() % 3 == 0:
                    self.step_random_vs_random()

            # Draw
            self.screen.fill(BG)
            draw_grid(self.screen, self.walls)
            draw_agent(self.screen, self.blue)
            draw_agent(self.screen, self.red)
            if self.met:
                # highlight the meeting cell
                x, y = self.blue.pos
                rect = pygame.Rect(x * (CELL + MARGIN) + MARGIN,
                                   y * (CELL + MARGIN) + MARGIN,
                                   CELL, CELL)
                pygame.draw.rect(self.screen, MEET, rect, width=4, border_radius=8)

            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    ChanceEncounter().run()

