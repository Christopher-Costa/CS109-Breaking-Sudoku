#! /usr/bin/python3

from sudoku.sudoku import Sudoku

S = Sudoku(square_size = 3, training_data = "training_data/9x9_solutions", squares = [])

while True:

    S.print()

    row = input("Row: ")
    col = input("Col: ")
    value = input("Value: ")

    S.update_board([[int(row), int(col), int(value)]])

