import time
import argparse
from os.path import exists as file_exists
from queue import LifoQueue, PriorityQueue

# todo program ma byc stosowany do roznych wymiarow tablic, do zmiany pozniej
SOLVED_PUZZLE = [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 0]]
# todo start_puzzle ma byc argumentem wywolania, narazie na sztywno do testow
DEPTH = 25

class Arguments:
    @staticmethod
    def __is_permutation(str1, str2):
        return set(str1) == set(str2)

    def __validate__(self):
        if self.strategy not in ("bfs", "dfs", "astr"):
            raise ValueError("Jako argument podano nieistniejaca strategie")
        if self.additional_param not in ("hamm", "manh") and not (Arguments.__is_permutation(self.additional_param, "LRUD")):
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


class Board:
    def __validate_board(self):
        correct_values = set(range(0, len(self.elements) * len(self.elements[0]))) # TODO do poprawy!!!
        for y in range(0, len(self.elements)): # Może da się to prościej napisać?
            for x in range(0, len(self.elements[y])):
                if self.elements[y][x] not in correct_values:
                    raise ValueError("Plansza zawiera bledne wartości!")

    def __init__(self, elements):
        self.elements = elements
        self.__validate_board()


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
    def __init__(self, current_puzzle, previous_puzzle, solution, step, *, sequence=None):
        # aktualny stan planszy
        self.puzzle = current_puzzle.copy()

        # stany do ktorych mozna dojsc po wykonaniu kroku
        # operator wykonany na rodzicu
        self.last_move = step
        # aktualny ciąg operatorów (rozwiązanie)
        self.solution = solution.copy()
        if self.last_move is not None:
            self.solution.append(step)
        # kolejnosc kroków przy szukaniu rozwiązania
        if sequence is not None:
            self.sequence = sequence.copy()  # SEQUENCE.copy()
        # todo usunac komentarz na koniec, sequence ma byc argumentem wywolania
        self.zero = self.find_zero()
        if previous_puzzle != 'parentless':
            self.parent = previous_puzzle

    def get_neighbours(self): # TODO dodaj obsługę astar (heurystyk)
        """Zwraca sąsiadów w kolejności sequence"""
        x = self.zero["x"]
        y = self.zero["y"]
        neighbours = []
        original_sequence = self.sequence.copy()
        self.block_prohibited_moves()
        for i in range(len(self.sequence)):
            neighbours.append([row[:] for row in self.puzzle])

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
                #self.zero = self.find_zero()
                neighbours[index] = Node(neighbour, self.puzzle, self.solution, c, sequence=original_sequence)
                index += 1

        self.zero = self.find_zero() # TODO można poprawić żeby liczył dla każdego ruchu osobno
        return neighbours

    def block_prohibited_moves(self):
        # Likwidujemy możliwość cofnięcia ruchu, która sztucznie by wydłużała szukanie rozwiązania
        # W pierwszej iteracji nie ma czego cofać
        if self.last_move is not None:
            if self.last_move == 'L':
                self.sequence.remove('R')
            elif self.last_move == 'R':
                self.sequence.remove('L')
            elif self.last_move == 'U':
                self.sequence.remove('D')
            elif self.last_move == 'D':
                self.sequence.remove('U')
        # sprawdzamy krańce macierzy stanu i blokujemy wyjście poza granice
        if self.zero["y"] == 0:
            self.sequence.remove('U')
        elif self.zero["y"] == len(self.puzzle) - 1:
            self.sequence.remove('D')
        if self.zero["x"] == 0:
            self.sequence.remove('L')
        elif self.zero["x"] == len(self.puzzle) - 1:
            self.sequence.remove('R')

    def find_zero(self):
        for y in range(len(self.puzzle)):
            for x in range(len(self.puzzle[y])):
                if self.puzzle[y][x] == 0:
                    return {"x": x, "y": y}
        raise Exception("Nie znaleziono zera w ukladance!")


# sprawdza czy osiagnelismy stan docelowy
def is_goal(puzzle): # TODO przepisz!!!!
    if puzzle == SOLVED_PUZZLE:
        return True
    return False

# todo czym jest maksymalna osiagnieta glebokosc rekursji? dodac do kodu
def bfs(start_time, board, additional_param):
    visited = 1
    processed = 1
    current_node = Node(board.elements, 'parentless', [], None, sequence=additional_param)
    if is_goal(current_node.puzzle):
        return current_node.solution, len(current_node.solution),\
               processed, visited, round((time.process_time() - start_time) * 1000, 3)
    open_states = []
    closed_states = set()
    open_states.append(current_node)
    # pętla zatrzyma sie gdy wszystkie stany otwarte zostaną przetworzone
    while open_states:
        v = open_states.pop(0)
        if is_goal(v.puzzle):
            return v.solution, len(v.solution), \
                processed, visited, round((time.process_time() - start_time) * 1000, 3)
        if v not in closed_states:
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
    processed = 1
    current_node = Node(board, 'parentless', [], None, sequence=additional_param)
    if is_goal(current_node.puzzle):
        return current_node.solution, len(current_node.solution),\
               processed, visited, round((time.process_time() - start_time) * 1000, 3)
    open_states = LifoQueue()
    closed_states = set()
    open_states.put(current_node)
    while not open_states.empty():
        v = open_states.get()
        if v not in closed_states:
            processed += 1
            closed_states.add(v)
            for n in list(reversed(v.sequence)):
                v.move(n, sequence=additional_param)
            for n in v.neighbours.values():
                if is_goal(n.puzzle):
                    return n.solution, len(n.solution),\
                       processed, visited, round((time.process_time() - start_time) * 1000, 3)
                if len(n.solution) < DEPTH:
                    open_states.put(n)
        visited += 1
    return False


def astar(start_time, board, heuristic, additional_param):
    processed = 1
    visited = 1
    current_node = Node(board, 'parentless', [], None)
    if is_goal(current_node.puzzle):
        return current_node.solution, len(current_node.solution), \
               processed, visited, round((time.process_time() - start_time) * 1000, 3)
    p = PriorityQueue()
    t = set()
    p.put((0, current_node)) # nie jestem pewny czy tak powinno to wygladac
    while not p.empty():
        v = p.get()
        if is_goal(current_node.puzzle):
            return v.solution, len(current_node.solution), \
                   processed, visited, round((time.process_time() - start_time) * 1000, 3)
        t.add(v)




def main():
    args = parse_arguments()
    board = None
    with open(args.source_file, "r") as file:
        if not file.readable():
            raise Exception("Nie można otworzyć pliku" + file.name)

        input = [int(i) for i in file.read().split() if i.isdigit()]
        if len(input) < 3:
            raise Exception("Plik wejściowy zawiera zbyt mało elementów: " + str(len(input)))
        dimensions = (input[0], input[1])

        del input[:2]

        # Na podstawie wymiarów tworzymy z input macierz o wymiarach podanych w pliku
        matrix = []
        for y in range(dimensions[1]):
            row = []
            for x in range(dimensions[0]):
                row.append(input[0])
                input.pop(0)
            matrix.append(row)

        board = Board(matrix)

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
        #output = astar(start_time, board)

    # Pozostało zapisać wyniki
    with open(args.save_file, "w") as file:
        if output == False:
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
            #file.write(str(output[5]) + '\n') # TODO maksymalna głębokość rekursji w kodzie!
            file.write(str(output[4]) + '\n')

    # TODO stworzyć osobną klasę z output, w tej formie jak teraz nie wiadomo ktory argument do czego służy

if __name__ == "__main__":
    main()
