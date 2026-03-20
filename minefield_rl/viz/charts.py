from __future__ import annotations

from typing import Sequence

import pygame


def draw_line_chart(
    surface: pygame.Surface,
    rect: pygame.Rect,
    values: Sequence[float],
    color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    label: str,
    font: pygame.font.Font,
) -> None:
    pygame.draw.rect(surface, bg_color, rect, border_radius=8)
    pygame.draw.rect(surface, (80, 90, 105), rect, width=1, border_radius=8)
    surface.blit(font.render(label, True, (230, 235, 240)), (rect.x + 8, rect.y + 6))
    if len(values) < 2:
        return

    min_value = min(values)
    max_value = max(values)
    value_range = max(max_value - min_value, 1e-6)
    plot_rect = pygame.Rect(rect.x + 8, rect.y + 28, rect.width - 16, rect.height - 36)
    points = []
    for index, value in enumerate(values):
        x = plot_rect.x + int(index * (plot_rect.width - 1) / max(len(values) - 1, 1))
        y = plot_rect.bottom - int((value - min_value) / value_range * plot_rect.height)
        points.append((x, y))
    pygame.draw.lines(surface, color, False, points, 2)


def draw_bar_chart(
    surface: pygame.Surface,
    rect: pygame.Rect,
    labels: Sequence[str],
    values: Sequence[float],
    highlight_idx: int | None,
    font: pygame.font.Font,
    title: str,
) -> None:
    pygame.draw.rect(surface, (30, 38, 52), rect, border_radius=8)
    pygame.draw.rect(surface, (80, 90, 105), rect, width=1, border_radius=8)
    surface.blit(font.render(title, True, (230, 235, 240)), (rect.x + 8, rect.y + 6))
    if not values:
        return

    plot_rect = pygame.Rect(rect.x + 8, rect.y + 28, rect.width - 16, rect.height - 42)
    max_abs = max(max(abs(v) for v in values), 1e-6)
    baseline = plot_rect.y + plot_rect.height // 2
    bar_width = max(8, plot_rect.width // max(len(values) * 2, 1))

    pygame.draw.line(surface, (100, 110, 125), (plot_rect.x, baseline), (plot_rect.right, baseline), 1)
    for index, (label, value) in enumerate(zip(labels, values)):
        x = plot_rect.x + index * (plot_rect.width // max(len(values), 1)) + bar_width // 2
        height = int((abs(value) / max_abs) * (plot_rect.height // 2 - 8))
        color = (103, 182, 255) if index != highlight_idx else (255, 198, 109)
        bar_rect = pygame.Rect(x, baseline - height if value >= 0 else baseline, bar_width, max(height, 2))
        pygame.draw.rect(surface, color, bar_rect, border_radius=4)
        label_surface = font.render(label, True, (220, 225, 230))
        surface.blit(label_surface, (x - label_surface.get_width() // 2, rect.bottom - 18))
