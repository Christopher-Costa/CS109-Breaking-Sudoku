#! /usr/bin/python3

from sudoku.sudoku import Sudoku

starting_squares = [
    (0, 3, 4),
    (0, 5, 7),
    (1, 0, 7)
]

S = Sudoku(
    square_size = 3, 
    training_data = "training_data/9x9_solutions", 
    method = "Conditional",
    squares = starting_squares
)

S.solve()
S.print()
