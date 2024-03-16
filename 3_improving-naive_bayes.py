#! /usr/bin/python3

from sudoku.sudoku import Sudoku

starting_squares = [
    (0, 0, 1),
    (0, 1, 2),
    (0, 2, 3),
    (3, 3, 4),
    (3, 4, 5),
    (3, 5, 6),
]

S = Sudoku(
    square_size = 3, 
    training_data = "training_data/9x9_solutions", 
    method = "NaiveBayes",
    squares = starting_squares
)

S.print()
S.solve()
S.print()
