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
    method = "Conditional",
    squares = starting_squares
)

S.print()
S.print_p(1)
S.print_p(2)
S.print_p(3)
S.print_p(4)
S.print_p(5)
S.print_p(6)
S.print_p(7)
S.print_p(8)
S.print_p(9)

S.solve()
S.print()
