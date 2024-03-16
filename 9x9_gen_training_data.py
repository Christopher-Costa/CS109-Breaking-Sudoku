#! /usr/bin/python3

from sudoku.sudoku import Sudoku

S = Sudoku(method='Generate', square_size = 3, training_data='training_data/9x9_solutions')
S.gen_training_data()
