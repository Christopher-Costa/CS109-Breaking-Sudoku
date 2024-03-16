#! /usr/bin/python3

from sudoku.sudoku import Sudoku

starting_squares = [
    (0, 1, 1),
    (1, 2, 2)
]

S = Sudoku(
    square_size = 2, 
    training_data = "training_data/4x4_solutions", 
    method = "NaiveBayes",
    squares = starting_squares
)

S.solve()
S.print()
