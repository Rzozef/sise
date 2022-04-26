import time
import argparse
from os.path import exists as file_exists
from queue import LifoQueue, PriorityQueue

MAX_DEPTH = 25


class Arguments:
    @staticmethod
    def __is_permutation(str1, str2):
        return set(str1) == set(str2)

    def __validate__(self):
        if self.strategy not in ("bfs", "dfs", "astr"):
            raise ValueError("Jako argument podano nieistniejaca strategie")
        if self.additional_param not in ("hamm", "manh") and not (
                Arguments.__is_permutation(self.additional_param, "LRUD")):
            raise ValueError("Podany akronim jest błędny")
        if not file_exists(self.source_file):
            raise ValueError("Podany plik zrodlowy nie istnieje")
        if not file_exists(self.save_file):
            raise ValueError("Plik do ktorego maja zostac zapisane wyniki nie istnieje")
        if not file_exists(self.additional_info_file):
            raise ValueError("Plik do którego mają zostać zapisanme dodatkowe informacje nie istnieje")

    def __init__(self, strategy, acronym, source_file, save_file, additional_info_file):
        self.strategy = strategy
        self.additional_param = acronym
        self.source_file = source_file
        self.save_file = save_file
        self.additional_info_file = additional_info_file
        self.__validate__()


class State:
    target_boards = {}  # dimension: Board

    def validate(self):
        correct_values = set(range(0, len(self.elements) * len(self.elements[0])))  # TODO do poprawy!!!
        for y in range(0, len(self.elements)):  # Może da się to prościej napisać?
            for x in range(0, len(self.elements[y])):
                if self.elements[y][x] not in correct_values:
                    raise ValueError("Plansza zawiera bledne wartości!")

    def __init__(self, elements):
        self.elements = elements

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

    def __eq__(self, other):
        for y in range(0, len(self.elements)): # TODO popraw
            for x in range(0, len(self.elements[0])):
                if self.elements[y][x] != other.elements[y][x]:
                    return False
        return True

    def __hash__(self): #TODO przepisz to
        return 0

    def get_dimension(self):
        return len(self.elements)

    @staticmethod
    def get_target_state(dimension):
        if dimension <= 1:
            raise Exception(f"Nie mozna stworzyc rozwiazanej planszy dla wymiarow {dimension}x{dimension}")
        if dimension in State.target_boards:
            return State.target_boards[dimension]
        elements = []
        for y in range(0, dimension):
            row = []
            for x in range(0, dimension):
                row.append(y * dimension + x + 1)
            elements.append(row)
        elements[dimension - 1][dimension - 1] = 0
        State.target_boards[dimension] = State(elements)
        return State.target_boards[dimension]


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("strategy", nargs=1)
    arg_parser.add_argument("acronym", nargs=1)
    arg_parser.add_argument("source_file", nargs=1)
    arg_parser.add_argument("save_file", nargs=1)
    arg_parser.add_argument("additional_info_file", nargs=1)
    arguments = arg_parser.parse_args()
    return Arguments(
        arguments.strategy[0],
        arguments.acronym[0],
        arguments.source_file[0],
        arguments.save_file[0],
        arguments.additional_info_file[0]
    )


class Node:
    def __init__(self, current_state, previous_node, step, *, sequence=None, depth=0):
        # aktualny stan planszy
        self.state = current_state

        # stany do ktorych mozna dojsc po wykonaniu kroku
        # operator wykonany na rodzicu
        self.last_move = step
        # kolejnosc kroków przy szukaniu rozwiązania
        if sequence is not None:
            self.sequence = sequence.copy()  # SEQUENCE.copy()
        else:
            self.sequence = ['L', 'R', 'U', 'D'] # Przykładowa sekwencja

        self.step = step
        self.zero = self.find_zero()
        self.parent = previous_node
        self.depth = depth

    def __eq__(self, other):
        return self.state == other.state and self.last_move == other.last_move and self.depth == other.depth

    def __hash__(self):
        return hash(self.state) + hash(self.last_move) + hash(self.depth)

    def get_neighbours(self):
        """Zwraca sąsiadów w kolejności sequence"""
        x = self.zero["x"]
        y = self.zero["y"]
        neighbours = []
        original_sequence = self.sequence.copy()
        self.block_prohibited_moves()
        for i in range(len(self.sequence)):
            neighbours.append([row[:] for row in self.state])

        if self.sequence is not None:
            index = 0
            for c in self.sequence:
                neighbour = neighbours[index]
                if c == "L":
                    neighbour[y][x - 1], neighbour[y][x] = neighbour[y][x], neighbour[y][x - 1]
                elif c == "R":
                    neighbour[y][x + 1], neighbour[y][x] = neighbour[y][x], neighbour[y][x + 1]
                elif c == "U":
                    neighbour[y - 1][x], neighbour[y][x] = neighbour[y][x], neighbour[y - 1][x]
                elif c == "D":
                    neighbour[y + 1][x], neighbour[y][x] = neighbour[y][x], neighbour[y + 1][x]
                # def __init__(self, current_puzzle, previous_puzzle, solution, step, *, sequence=None):
                # self.zero = self.find_zero()
                neighbours[index] = Node(State(neighbour), self, c, sequence=original_sequence, depth=self.depth + 1)
                index += 1

        self.zero = self.find_zero()  # TODO można poprawić żeby liczył dla każdego ruchu osobno
        return neighbours

    def block_prohibited_moves(self):
        # sprawdzamy krańce macierzy stanu i blokujemy wyjście poza granice
        if self.zero["y"] == 0:
            self.sequence.remove('U')
        elif self.zero["y"] == len(self.state) - 1:
            self.sequence.remove('D')
        if self.zero["x"] == 0:
            self.sequence.remove('L')
        elif self.zero["x"] == len(self.state) - 1:
            self.sequence.remove('R')

    def find_zero(self):
        for y in range(len(self.state)):
            for x in range(len(self.state[y])):
                if self.state[y][x] == 0:
                    return {"x": x, "y": y}
        raise Exception("Nie znaleziono zera w ukladance!")

    def get_solution(self):
        solution = []
        if self.step is None:
            return solution
        solution.append(self.step)
        parent = self.parent
        while parent.step is not None:
            solution.append(parent.step)
            parent = parent.parent
        solution.reverse()
        return solution

    def __lt__(self, other): # TODO chyba może być tutaj cokolwiek?
        return True
    def __le__(self, other):
        return True


# sprawdza czy osiagnelismy stan docelowy
def is_goal(board):  # TODO przepisz!!!!
    if board == State.get_target_state(
            board.get_dimension()):  # TODO nie powinno to być generowane przy kazdym sprawdzeniu...
        return True
    return False


def bfs(start_time, board, additional_param):
    visited = 1
    processed = 0
    current_node = Node(board, None, None, sequence=additional_param)
    open_states = []
    closed_states = set()
    max_depth = 0
    open_states.append(current_node)
    # pętla zatrzyma sie gdy wszystkie stany otwarte zostaną przetworzone
    while open_states:
        v = open_states.pop(0)
        processed += 1
        max_depth = max(max_depth, v.depth)
        if is_goal(v.state):
            return v.get_solution(), len(v.get_solution()), \
                   processed, visited, round((time.process_time() - start_time) * 1000, 3), max_depth
        if v not in closed_states and v.depth < MAX_DEPTH:  # TODO <= MAX_DEPTH na pewno w tym miejscu?
            closed_states.add(v)
            neighbours = v.get_neighbours()
            for n in neighbours:
                if n not in closed_states:
                    open_states.append(n)
                    visited += 1
    return False


# todo przemyslec gdzie wstawic processed i visited
def dfs(start_time, board, additional_param):
    visited = 1
    processed = 0
    current_node = Node(board, None, None, sequence=additional_param)
    open_states = LifoQueue()
    closed_states = set()
    open_states.put(current_node)
    max_depth = 0
    while not open_states.empty():
        v = open_states.get()
        processed += 1
        max_depth = max(max_depth, v.depth)
        if is_goal(v.state):
            return v.get_solution(), len(v.get_solution()), \
                   processed, visited, round((time.process_time() - start_time) * 1000, 3), max_depth
        if v not in closed_states and v.depth < MAX_DEPTH:
            closed_states.add(v)
            for n in list(reversed(v.get_neighbours())):
                open_states.put(n)
                visited += 1
    return False


class Hamming:
    def __call__(self, neighbour):
        diff = 0
        dimension = neighbour.state.get_dimension()
        target_state = State.get_target_state(dimension)
        for y in range(0, dimension):  # TODO da się to uprościć???
            for x in range(0, dimension):
                if neighbour.state[y][x] != target_state[y][x] and neighbour.state[y][x] != 0:
                    diff += 1
        return diff


class Manhattan:
    def __search_for_position__(self, value, target_state): # value nie może być 0 i musi znajdować się w stanie
        dimension = target_state.get_dimension()
        return (value % dimension) - 1, (value - 1) // dimension
    def __call__(self, neighbour):
        diff = 0
        dimension = neighbour.state.get_dimension()
        target_state = State.get_target_state(dimension)
        for y in range(0, dimension):  # TODO da się to uprościć???
            for x in range(0, dimension):
                if neighbour.state[y][x] != target_state[y][x] and neighbour.state[y][x] != 0:
                    pos = self.__search_for_position__(neighbour.state[y][x], target_state)
                    diff += abs(x - pos[0]) + abs(y - pos[1])
        return diff


class Record:
    def __init__(self, priority, id, node):
        self.priority = priority
        self.id = id
        self.node = node

    def __lt__(self, other):
        return (self.priority, self.id) < (other.priority, other.id)

    def __le__(self, other):
        return (self.priority, self.id) <= (other.priority, other.id)


def astr(start_time, board, additional_param):
    heuristics = None
    if additional_param == 'manh':
        heuristics = Manhattan()
    elif additional_param == 'hamm':
        heuristics = Hamming()
    else:
        raise Exception(f"Nieznany akronim heurystyki: {additional_param}")

    visited = 1
    processed = 0
    current_node = Node(board, None, None)
    open_states = PriorityQueue()
    max_depth = 0
    closed_states = set()
    open_states.put(Record(heuristics(current_node), processed, current_node))
    while open_states:
        v = open_states.get().node
        processed += 1
        max_depth = max(max_depth, v.depth)
        if is_goal(v.state):
            return v.get_solution(), len(v.get_solution()), \
                   processed, visited, round((time.process_time() - start_time) * 1000, 3), max_depth
        closed_states.add(v)
        for n in v.get_neighbours():
            if n not in closed_states:
                f = n.depth + heuristics(n)
                open_states.put(Record(f, processed, n))
                visited += 1
    return False


def main():
    args = parse_arguments()
    board = None
    with open(args.source_file, "r") as file:
        if not file.readable():
            raise Exception("Nie można otworzyć pliku" + file.name)

        input_nums = [int(i) for i in file.read().split() if i.isdigit()]
        if len(input_nums) < 3:
            raise Exception("Plik wejściowy zawiera zbyt mało elementów: " + str(len(input_nums)))
        dimensions = (input_nums[0], input_nums[1])

        del input_nums[:2]

        # Na podstawie wymiarów tworzymy z input macierz o wymiarach podanych w pliku
        matrix = []
        for y in range(dimensions[1]):
            row = []
            for x in range(dimensions[0]):
                row.append(input_nums[0])
                input_nums.pop(0)
            matrix.append(row)

        board = State(matrix)
        board.validate()

    output = None
    start_time = None
    if args.strategy == "bfs":
        start_time = time.process_time()
        output = bfs(start_time, board, list(args.additional_param))
    elif args.strategy == "dfs":
        start_time = time.process_time()
        output = dfs(start_time, board, list(args.additional_param))
    elif args.strategy == "astr":
        start_time = time.process_time()
        output = astr(start_time, board, ''.join(list(args.additional_param)))

    # Pozostało zapisać wyniki
    with open(args.save_file, "w") as file:
        if not output:
            file.write("-1")
        else:
            file.write(str(output[1]) + '\n')
            file.write(''.join(output[0]))

    # I dodatkowe informacje
    with open(args.additional_info_file, "w") as file:
        if output is False:
            file.write("-1")
        else:
            file.write(str(output[1]) + '\n')
            file.write(str(output[3]) + '\n')
            file.write(str(output[2]) + '\n')
            file.write(str(output[5]) + '\n')
            file.write(str(output[4]) + '\n')

    # TODO stworzyć osobną klasę z output, w tej formie jak teraz nie wiadomo ktory argument do czego służy


if __name__ == "__main__":
    main()
