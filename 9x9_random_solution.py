#! /usr/bin/python3

from sudoku.sudoku import Sudoku

S = Sudoku(
    square_size = 3, 
    method = "BruteForce",
    sample_size = 1,
    randomize = True
)

S.print()
