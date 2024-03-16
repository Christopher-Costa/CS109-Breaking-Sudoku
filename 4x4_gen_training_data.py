#! /usr/bin/python3

from sudoku.sudoku import Sudoku

S = Sudoku(method='Generate', square_size = 2, training_data='training_data/4x4_solutions')
S.gen_training_data()
