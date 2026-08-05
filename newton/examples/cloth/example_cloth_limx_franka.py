# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Place a fixed square LIMX cloth on a table for a Franka grasp sequence."""

from __future__ import annotations

import numpy as np


def _create_square_cloth_grid(
    grid_cells: int,
    width: float,
    center: tuple[float, float],
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid_side = grid_cells + 1
    positions = np.empty((grid_side * grid_side, 3), dtype=np.float32)
    triangles: list[tuple[int, int, int]] = []

    for y in range(grid_side):
        for x in range(grid_side):
            index = y * grid_side + x
            positions[index] = (
                center[0] - 0.5 * width + width * x / grid_cells,
                center[1] - 0.5 * width + width * y / grid_cells,
                height,
            )

    for y in range(grid_cells):
        for x in range(grid_cells):
            lower_left = y * grid_side + x
            lower_right = lower_left + 1
            upper_left = lower_left + grid_side
            upper_right = upper_left + 1
            if (x + y) % 2 == 0:
                triangles.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
            else:
                triangles.extend(((lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left)))

    return positions, np.asarray(triangles, dtype=np.int32)
