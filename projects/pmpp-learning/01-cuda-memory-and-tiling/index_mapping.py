#!/usr/bin/env python3
"""Show the relation between CUDA (x, y) and matrix [row][col]."""

TILE_WIDTH = 2


def explain(block_x: int, block_y: int, thread_x: int, thread_y: int) -> None:
    x = block_x * TILE_WIDTH + thread_x
    y = block_y * TILE_WIDTH + thread_y
    row, col = y, x

    print(
        f"block=({block_x},{block_y}), thread=({thread_x},{thread_y}) "
        f"-> (x,y)=({x},{y}) "
        f"-> book Pd[{x},{y}] "
        f"-> usual P[row={row}][col={col}]"
    )


def main() -> None:
    print("x is the horizontal/column coordinate; y is the vertical/row coordinate.\n")
    for block_y in range(2):
        for block_x in range(2):
            for thread_y in range(TILE_WIDTH):
                for thread_x in range(TILE_WIDTH):
                    explain(block_x, block_y, thread_x, thread_y)

    print("\nThe example quoted from PMPP:")
    explain(block_x=0, block_y=1, thread_x=1, thread_y=0)


if __name__ == "__main__":
    main()
