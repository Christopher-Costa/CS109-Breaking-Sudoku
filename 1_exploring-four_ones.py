#! /usr/bin/python3

from sudoku.sudoku import Sudoku

starting_squares = [
    (0, 0, 1),
    (1, 3, 1),
    (3, 1, 1),
    (4, 4, 1)
]

S = Sudoku(
    square_size = 3, 
    method = "BruteForce",
    sample_size = 20000,
    squares = starting_squares,
    randomize = True
)

S.print()
S.print_p(1)
