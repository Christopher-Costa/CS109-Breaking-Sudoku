import random
import math
import numpy as np
import pandas as pd

from collections import deque

class Sudoku():
    """
    Play a game of Sudoku, and use random sampling to estimate the
    probability of remaining valid moves!

    Arguments:
      square_size: The length & height of the sudoku square.  A standard game is 3x3 squares and a 9x9 board
      sample_size: When doing sampling, how large should the samples be.  Use sane numbers, or be very patient!
      squares:     A list of lists representing starting board values in the format (row, column, value)
      method:      The method to be used to derive probabilities.  valid options are:
                        "NaiveBayes", "BruteForce", "Conditional".  Defaults to "Conditional"
      laplace:     Boolean to indicate whether LaPlace smoothing should be employed, where appropriate
      randomize:   Boolean to indicate whether to use randomized functions during sample generation

    Attributes:
      board:       A list representing each cell on the board, and it's current value if any.
      p:           A dictionary of estimated probabilies for each possible cell/value combination.
    """ 

    def __init__(self, square_size = 3, sample_size = 10000,  squares = [], 
                 training_data=0,       method='Conditional', laplace=False,
                 randomize = False):

        self.square_size = square_size
        self.sample_size = sample_size
        self.training_data = training_data
        self.method = method
        self.laplace = laplace
        self.randomize = randomize

        # Initialize the board with 'None' values to indicate no values have been placed on the board yet.
        self.board = [ None for n in range(self.num_cells()) ]

        # Initialize the dictionary of starting probabilities to None
        self.initialize_probabilities()

        if self.method == 'NaiveBayes':
            self.train_nb_model()
        elif self.method == 'Conditional':
            self.create_conditional_probability_table()
        elif self.method == 'Generate':
            self.gen_training_data()
            return

        # Initialize the starting squares
        self.update_board(squares, True)
            
    def initialize_probabilities(self):
        self.p = dict()
        for cell in range(0, self.num_cells()):
            self.p[cell] = dict()
            for value in range(1, self.num_values() + 1):
                self.p[cell][value] = None

    def create_conditional_probability_table(self):
        if self.training_data:
            self.p_table = pd.read_csv(self.training_data,header=None).to_numpy()

    ###################
    # Helper Functions
    ###################
    
    def num_cells(self):
        """
        The total number of cells on the board
        """
        return self.square_size ** 4

    def num_values(self):
        """
        The total number of values to be placed in cells.  Also, the maximum possible value.
        """
        return self.square_size ** 2

    def cell_from_rc(self, row, column):
        """
        The index of a cell, converted from a row and column number
        """
        return row * self.num_values() + column

    def available_values(self, board, cell):
        """
        Following the rules of Sudoku, where a value may only appear
        one time in each row, column, and local square, determine which 
        values are available to be assigned to the passed cell index.
        """

        n = self.num_values()
        # Start by assuming all values are available...
        values = [x for x in range(1, n + 1)]
        
        # ... remove any values we find in the cell's row
        row = int(cell / n)
        for r in range(row * n, row * n + n):
            if board[r] not in (None, 0):
                values.remove(board[r])
    
        # ... and any values we find in the cell's column
        col = cell % n
        for c in range(col, self.num_cells(), n):
            if board[c] not in (None, 0) and board[c] in values:
                values.remove(board[c])
    
        # ... and finally any value we find in the cell's local square
        starting_row = int(cell / n)
        starting_col = int(cell % n)
        starting_cell = (starting_row - (starting_row % self.square_size)) * n + starting_col - int(starting_col % self.square_size)
        sc = starting_cell
         
        for cr in range(sc, sc + self.square_size):
            for cc in range(0, self.square_size):
                c = cr + cc * n
                if board[c] is not None and board[c] in values:
                    values.remove(board[c])

        # Every remaining value is available to this cell
        return(values)

    def square_values(self, square):
        """
        Given the index of a square, return a list of cell indexes comprising the square.  
        The square indexes are identified like this for a default 3x3 board

            +-+-+-+
            |0|1|2|
            +-+-+-+
            |3|4|5|
            +-+-+-+
            |6|7|8|
            +-+-+-+

        contained within that square.
        """

        row = int(square / self.square_size) * self.square_size
        col = square % self.square_size * self.square_size

        cells = []
        for r in range(row, row + self.square_size):
            for c in range(col, col + self.square_size):
                cells += [self.cell_from_rc(r, c)]
        
        return(cells)

    #########################
    # Board Update Functions
    #########################

    def _update_rc(self, row, column, value):
        """
        Add a value to the board, based on a row and column
        """
        self._update_cell(self.cell_from_rc(row, column), value)

    def _update_cell(self, cell, value):
        """
        Add a value to the board, based on the index of a cell
        """
        if cell >= 0 and cell < self.num_cells():
            if value >= 0 and value <= 9:
                if value in self.available_values(self.board, cell):
                    self.board[cell] = value

    def update_board(self, updates, forceUpdate = False):
        """
        Updates are either in the form (cell, value), or (row, column, value), depending
        on how the user decides to encode the change.
        """

        updated = False
        if isinstance(updates, list):
            for update in updates:
                if isinstance(update, list) or isinstance(update, tuple):
                    if len(update) == 3:
                        (row, column, value) = update
                        self._update_rc(row, column, value)
                    elif len(update) == 2:
                        (cell, value) = update
                        self._update_cell(cell, value)
                    updated = True

        if updated or forceUpdate:
            if self.method == 'NaiveBayes':
                self.update_ml_probabilities()
            elif self.method == 'BruteForce':
                self.update_bf_probabilities()
            else:
                self.update_cond_probabilities()

    ####################
    # Solving Functions
    ####################

    def solve(self, display_progress=True):
        """
        Iterate through the board, empty cell by empty cell using the most likely
        possible entry (based on the selected probability method). In the event
        it becomes impossible to move forward, short circuit procesing and print
        and error to the screen.
        """

        if display_progress:
            count = 0
            total = sum(1 for x in self.board if x is None)
            print("Solving Board      [" + ' ' * 100 + "]", end='\r', flush=True)

        failed = False
        while None in self.board:
            highest_value = 0
            highest_cell = None
            highest_prob = 0

            for cell in range(self.num_cells()):
                if self.board[cell] is None:

                    for value in range(1, self.num_values() + 1):
                        prob = self.p[cell][value]
                        if prob > highest_prob:
                            highest_value = value
                            highest_prob = prob
                            highest_cell = cell

            if display_progress:
                count += 1
                percent = int(count / total * 100) 
                print("Solving Board      [" + '#' * percent + ' ' * (100 - percent) + "]", end='\r', flush=True)

            if highest_cell is not None:
                if self.board[highest_cell] is None:
                    self.update_board([(highest_cell, highest_value)])
                    continue

            failed = True 
            break

        if display_progress:
            print()

        if failed:
            print("Unsolvable")

    #############################
    # Machine Learning Functions
    #############################

    def train_nb_model(self, display_progress=True):
        """
        Train a Naive Bayes machine learning model from training data supplied
        during class initialization.
        """

        self.label_counts = {}
        self.feature_counts = {}
        grid = pd.read_csv(self.training_data, header=None).to_numpy()

        if display_progress:
            total_steps = self.num_cells() * len(grid)
            steps = 0
            print("Training ML Model  [" + ' ' * 100 + "]", end='\r', flush=True)

        for cell in range(self.num_cells()):
    
            self.label_counts[cell] = [0, 0]
            self.feature_counts[cell] = {}
            self.feature_counts[cell][0] = [0 for x in range(self.num_cells())]
            self.feature_counts[cell][1] = [0 for x in range(self.num_cells())]

            train_features = grid
            train_labels = grid[:, cell].T 

            # Iterate through all the provided samples
            for sample_num, sample in enumerate(train_features):
                label = train_labels[sample_num]
                
                # For whatever label this is, increment the counter and set
                # up an array to track the feature counts if needed
                self.label_counts[cell][label] += 1

                #if label not in self.feature_counts[cell]:
                #    self.feature_counts[cell][label] = [0 for x in range(len(sample))]

                # Iterate through all the features, and update the counts
                for feature_num, feature in enumerate(sample):
                    self.feature_counts[cell][label][feature_num] += feature

            if display_progress:
                steps += len(train_features)
                percent = int(steps / total_steps * 100) 
                print("Training ML Model  [" + '#' * percent + ' ' * (100 - percent) + "]", end='\r', flush=True)

        if display_progress:
            print()
                    

    def predict_nb_cell(self, cell):
        """
        For a given board cell, use the Naive Bayes approach to estimate the relative
        probabilities for the cell taking each value.  Apply LaPlace smoothing, optionally.
        """

        probabilities = [0 for x in range(self.num_values())]
        board = self.board

        for value in range(1, self.num_values() + 1):
            probabilities[value-1] = self.label_counts[value][1] / sum(self.label_counts[value])

            for feature in range(self.num_cells()):
                if board[feature] == None:
                    continue

                if board[feature] == value:
                    label_count = self.label_counts[cell][1]
                    count = self.feature_counts[cell][1][feature]
                else:
                    label_count = self.label_counts[cell][0]
                    count = self.feature_counts[cell][0][feature]

                if self.laplace:
                    count += 1
                    label_count += 2

                probabilities[value-1] *= count / label_count

        return probabilities

    #####################################
    # Training Data Generation Functions
    #####################################

    def check_row(self, board, row, file):
        if row >= self.num_values():
            b = board.copy()
            file.write(','.join(str(cell) for cell in board) + "\n")
            return

        n = self.num_values()
        for cell in range(row * n, row * n + n):
            if 1 in self.available_values(board, cell):
                board[cell] = 1
                self.check_row(board, row+1, file)
                board[cell] = 0

    def gen_training_data(self):
        """
        Recursively detemine every possible pattern for a single value
        in a valid Soduku solution, and write it to the supplied file
        """
        file = open(self.training_data, 'w+')

        board = [ 0 for n in range(self.num_cells()) ]
        self.check_row(board, 0, file)
        file.close()


    ########################
    # Probability Functions
    ########################
    
    def update_ml_probabilities(self):
        
        for cell in range(self.num_cells()):
    
            if self.board[cell] is not None:
                for value in range(1, self.num_values() + 1):
                    if self.board[cell] == value:
                        self.p[cell][value] = 1
                    else:
                        self.p[cell][value] = 0

            else:
                probabilities = self.predict_nb_cell(cell)
                for value, probability in enumerate(probabilities):
                    self.p[cell][value+1] = probability

    def update_bf_probabilities(self):
        """
        Called upon initialization, and after any board updates, using rejection sampling
        of valid completed boards to compute the probability of every possible value
        existing in every possible cell on the board.
        """

        samples_observation = self.sample_solutions()
        for cell in self.p:
            for value in self.p[cell]:
                samples_event = self.reject_inconsistent(samples_observation, cell, value)
                self.p[cell][value] = len(samples_event) / len(samples_observation)


    def update_cond_probabilities(self):
        """
        Called upon initialization, and after any board updates, from the supplied training
        data use the precise conditional probabilities to determine the probability of each
        possible value being contained in each cell on the board.
        """
        for update_cell in self.p:
            for update_value in self.p[update_cell]:
                Y = [update_cell, 1]
                X = []
                for i, current_value in enumerate(self.board):
                    if self.board[i] != None and self.board[i] != 0:
                        if current_value == update_value:
                            X.append([i, 1])
                        else:
                            X.append([i, 0])
                self.p[update_cell][update_value] = self.P_Y_given_X(Y, X)
    
    def shuffled_indices(self):
        """
        For efficiency of generating samples, it works well to try to solve a square, 
        then solve the remaining squares in the same horizontal or vertical stripe, and then
        solve the remaining stripes. 

        This function randomizes the stripe order, the square order within the stripe, and
        the cell order within the square, and returns this randomized/optimized ordered list
        of indices.  This process maximizes the randomness of the sample by ensuring different 
        parts of the board are solved in a different order for each sample, while also minimizing 
        the time it takes for samples to be generated.
        """

        stripes = [x for x in range(self.square_size)]
        random.shuffle(stripes)

        indices = []
        for stripe in stripes:
            start = stripe * self.square_size
    
            squares = [x for x in range(start, start + self.square_size)]
            random.shuffle(squares)
            for square in squares:
                cells = self.square_values(square)
                random.shuffle(cells)
                indices += cells

        return indices

    def P_Y_given_X(self, Y, X):
        """
        For a list of passed (cell, value) combinations, determine the conditional
        probability of the the (cell, value) of Y.  i.e. Y[0] containing Y[1].
        """

        t = self.p_table.copy()

        for (cell, value) in X:
            t = t[t[:,cell] == value]
        
        total = len(t)

        t = t[t[:,Y[0]] == Y[1]]
        samples = len(t)
         
        return samples / total


    ########################
    # Brute Force Functions
    ########################

    def find_solution(self, board, indices, index):
        """
        Employ a "backtracking" function to recursively place random available values into cells.
        
        If an invalid state is reached where no further values can be assigned, backtrack and try
        different values until the board can be completed in a valid configuration.
        """

        # We've reached the end of the board, which means we have a valid
        # configuration.  Return True
        if index == self.num_cells():
            return True

        cell = indices[index]
    
        # If the current cell contains a value already, there are no choices to make, so move
        # towards the next cell.
        if board[cell] != None:
            return self.find_solution(board, indices, index + 1)
    
        # Start with all possible values, and randomize their order
        values = [v for v in range(1, self.num_values() + 1)]
        if self.randomize:
            random.shuffle(values)
    
        # Iterate through each of the values.  Check if there is a valid solution beginning
        # with that value.  Return if one is found, iterate if not.
        for value in values:
            if value in self.available_values(board, cell):
                board[cell] = value
    
                if self.find_solution(board, indices, index + 1):
                    return True

            board[cell] = None

        # We didn't find a solution.
        return False

    def sample_solutions(self, display_progress=True):
        """
        Sample and store sample_size number of samples of valid solved
        Sudoku board configurations.
        """

        solutions = []

        if display_progress:
            print("Sampling Solutions [" + ' ' * 100 + "]", end='\r', flush=True)

        # Iterate until we have enough samples (solutions)
        while len(solutions) < self.sample_size:
            # Create a working copy of the current state of the board to use 
            # for recursively seeking a random, valid solution
            board = self.board.copy()

            # Create a list for the indices and then rotate it randomly, so the
            # solution begins from a random point on the board for each sample.
            if self.randomize:
                indices = self.shuffled_indices()
            else:
                indices = list(range(0, self.num_cells()))

            if self.find_solution(board, indices, 0):
                solutions.append(board)
                if display_progress:
                    percent = int(len(solutions) / self.sample_size * 100)
                    print("Sampling Solutions [" + '#' * percent + ' ' * (100 - percent) + "]", end='\r', flush=True)

            else:
                # It's possible that there aren't any valid solutions depending
                # on the configuration.
                exit("Unsolvable")

        if display_progress:
            print()
        return solutions

    def reject_inconsistent(self, samples, cell, value):
        """
        From a valid set of board samples (solutions), assumed to be consistent with the 
        current board configuration, reject any samples inconsistent with the passed
        cell containing the passed value.        
        """

        consistent_samples = []
        for sample in samples:
            if sample[cell] == value:
              consistent_samples.append(sample)
        return consistent_samples

    ####################
    # Display Functions
    ####################

    def print_line(self, left_end, bar, vert, thick_vert, right_end, cell_width):
        if not cell_width:
            cell_width = self.square_size

        print(left_end, end='')
        for x in range(1, self.num_values()+1):
            for c in range(cell_width * 2 + 1):
                print(bar, end='')

            if x == self.num_values():
                print(right_end)
            elif (x % self.square_size) == 0:
                print(thick_vert, end='')
            else:
                print(vert, end='')

    def print_header(self, cell_width=0):
        self.print_line('╔', '═', '╤', '╦', '╗', cell_width)

    def print_footer(self, cell_width=0):
        self.print_line('╚', '═', '╧', '╩', '╝', cell_width)

    def print_thin_divider(self, cell_width=0):
        self.print_line('╟', '┈', '┼', '╂', '║', cell_width)

    def print_thick_divider(self, cell_width=0):
        self.print_line('╟', '═', '┼', '╂', '║', cell_width)

    def text_color(self, probability):
        # Attempt to provide as useful a color gradient
        # given the limitations of ANSI colors

        colors =  [c for c in range(232, 254)]
        colors += [254 for c in range(12)]
        colors += [255 for c in range(12)]

        # Select a color based on the probability.  The brighter the color, 
        # the higher the probability.  For a probablity of 1 use a special
        # color.

        if probability == 1:
            return 10
        else:  
            return colors[int(len(colors) * probability)] 
       
    def format_text(self, value, probability):
        # Colorize and embolden the text being displayed
        color_code = self.text_color(probability)
        return "\033[1m\033[38;5;{}m{}\033[0m".format(color_code, value)

    def value_text(self, value):
        # Keep the value as a single character so that everything prints well
        # in the case of large boards like 16x16 or 25x25.  

        # This won't work for boards larger than that, but who's trying to 
        #solve a 36x36 game of Sudoku anyway?
        values = '?1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if value > len(values):
            return value
        
        return values[value]
 
    def print(self):
        """
        Print the board as a traditional Sudoku grid.  Within each square
        display each value, in a "heatmap" style where the boldness of a 
        color represents higher probability
        """
        self.print_header()
        
        for row in range(0, self.num_values()):
            for cell_row in range(self.square_size):
                print('║ ', end='')
                for col in range(self.num_values()):
                    for cell_col in range(self.square_size):
                        value = cell_row * self.square_size + cell_col + 1
                        cell = self.cell_from_rc(row, col)
                        prob = self.p[cell][value]

                        if prob == 0:
                            text = ' '
                        elif math.isnan(prob):
                            text = self.value_text(0)
                        else:
                            text = self.format_text(self.value_text(value), prob)

                        print("{} ".format(text), end='')
                    if col == self.num_values() - 1:
                        print('║', end='')
                    elif (col + 1) % self.square_size == 0:
                        print('┃ ', end='')
                    else:
                        print('┊ ', end='')
                print()
                        
            if row == self.num_values() - 1:
                self.print_footer()
            elif (row + 1) % self.square_size == 0:
                self.print_thick_divider()
            else:
                self.print_thin_divider()

        print()

    def print_p(self, value):
        """
        Print the board as a traditional Sudoku grid.  Within each square, for the value
        that was passed, display the probability of each cell taking on that value.  Use
        a "heatmap" style where the boldness of a color represents higher probability.
        """
        self.print_header(3)

        for row in range(self.num_values()):
            for cell_row in range(3):
            #for cell_row in range(self.square_size):
                print('║', end='')
                for col in range(self.num_values()):
                    if cell_row == 0:
                        print('   {}   '.format(self.value_text(value)), end='')
                    elif cell_row == 1:
                        print('       ', end='')
                    else:
                        cell = self.cell_from_rc(row, col)
                        prob = self.p[cell][value]
                        text = self.format_text("%0.3f" % prob, prob)
                        print(' {} '.format(text), end='')

                    if col == self.num_values() - 1:
                        print('║', end='')
                    elif (col + 1) % self.square_size == 0:
                        print('┃', end='')
                    else:
                        print('┊', end='')
                print()

            if row == self.num_values() - 1:
                self.print_footer(3)
            elif (row + 1) % self.square_size == 0:
                self.print_thick_divider(3)
            else:
                self.print_thin_divider(3)
