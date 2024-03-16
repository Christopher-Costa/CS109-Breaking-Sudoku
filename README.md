# CS109 Challenge Project

Welcome to the source code repository for Breaking Sudoku.

## Module

The file `sudoku/sudoku.py` contains all the code used for this project, specifically a custom python `Suduko` class.  The class has methods that do a variety of things that support all of the functionality leveraged during the course of completing this project.  This includes:

* Maintain an object representing the values on a Sudoku board.
* Allow for validating moves and updating values on the board.
* Dynamically compute and update probabilities on the board based on a variety of different methods.
* Display the board and values probabilities textually on a terminal, using a "heat map" approach where the brightness of a value represents higher probabilities.
* Solve a puzzle using a variety of different methods.
* Generate training data for Machine Learning and other probability computations.

## Probability Methods

### Brute-Force

Using a conventional, recursive backtracking methodology, sample possible solutions and derive the probability from the set of results.

### Conditional Probability

Using a simplified representation of all board possibilities as input, derive conditional probabilities of values based on the current board configuration.

### Machine Learning

Using a variation of Naive Bayes, train a model using the same input data as above, and estimate/approximate probabilities assuming conditional independence of cells.

## Provided Scripts

These scripts were used to produce the results included the in the projects written report.

*  `1_exploring-empty_board.py`
*  `1_exploring-four_ones.py`
*  `2_implementing-conditional_solve.py`
*  `3_improving-naive_bayes.py`

These are general purpose scripts that provide examples for most of the Sudoku modules functionality
  
* `4x4_conditional_solve.py`
* `4x4_gen_training_data.py`
* `4x4_naive_bayes_solve.py`
* `9x9_brute_force_solve.py`
* `9x9_conditional_solve.py`
* `9x9_gen_training_data.py`
* `9x9_naive_bayes_solve.py`
* `9x9_random_solution.py`

One last very basic script if you want an interactive session to play a game of Sudoku and avail yourself of all the benefits of understanding the probability of possible next moves.

* `interactive.py`
