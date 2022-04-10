import time
from queue import LifoQueue, PriorityQueue

# todo program ma byc stosowany do roznych wymiarow tablic, do zmiany pozniej
SOLVED_PUZZLE = [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 0]]
SEQUENCE = []
START_PUZZLE = [[1, 3, 4, 8],
                [5, 2, 7, 0],
                [9, 6, 11, 12],
                [13, 10, 14, 15]]
# todo start_puzzle ma byc argumentem wywolania, narazie na sztywno do testow
DEPTH = 25


class Node:
    def __init__(self, current_puzzle, previous_puzzle, solution, step):
        # aktualny stan planszy
        self.puzzle = current_puzzle
        # stany do ktorych mozna dojsc po wykonaniu kroku
        self.neighbours = {}
        # operator wykonany na rodzicu
        self.last_move = step
        # aktualny ciąg operatorów (rozwiązanie)
        self.solution = solution.copy()
        if self.last_move is not None:
            self.solution.append(step)
        # kolejnosc kroków przy szukaniu rozwiązania
        self.sequence = ['L', 'R', 'U', 'D']  # SEQUENCE.copy()
        # todo usunac komentarz na koniec, sequence ma byc argumentem wywolania
        self.zero = self.find_zero()
        if previous_puzzle != 'parentless':
            self.parent = previous_puzzle

    # metoda tworzy stan-dziecko i zapisuje je w dictionary neighbours
    def create_neighbour(self, new_puzzle, step):
        neighbour = Node(new_puzzle, self.puzzle, self.solution, step)
        self.neighbours[step] = neighbour

    def move(self, step):
        x = self.zero["x"]
        y = self.zero["y"]
        if step == 'L':
            helper = []
            for row in self.puzzle:
                helper.append(row.copy())
            helper[y][x - 1], helper[y][x] = helper[y][x], helper[y][x - 1]
            self.create_neighbour(helper, 'L')
        elif step == 'R':
            helper = []
            for row in self.puzzle:
                helper.append(row.copy())
            helper[y][x + 1], helper[y][x] = helper[y][x], helper[y][x + 1]
            self.create_neighbour(helper, 'R')
        elif step == 'U':
            helper = []
            for row in self.puzzle:
                helper.append(row.copy())
            helper[y - 1][x], helper[y][x] = helper[y][x], helper[y - 1][x]
            self.create_neighbour(helper, 'U')
        elif step == 'D':
            helper = []
            for row in self.puzzle:
                helper.append(row.copy())
            helper[y + 1][x], helper[y][x] = helper[y][x], helper[y + 1][x]
            self.create_neighbour(helper, 'D')
        self.zero = self.find_zero() # TODO można poprawić żeby liczył dla każdego ruchu osobno

    def find_zero(self):
        for y in range(len(self.puzzle)):
            for x in range(len(self.puzzle[y])):
                if self.puzzle[y][x] == 0:
                    return {"x": x, "y": y}
        raise Exception("Nie znaleziono zera w ukladance!")


# sprawdza czy osiagnelismy stan docelowy
def is_goal(puzzle):
    if puzzle == SOLVED_PUZZLE:
        return True
    return False


def block_prohibited_moves(current_node):
    # Likwidujemy możliwość cofnięcia ruchu, która sztucznie by wydłużała szukanie rozwiązania
    # W pierwszej iteracji nie ma czego cofać
    if current_node.last_move is not None:
        if current_node.last_move == 'L':
            current_node.sequence.remove('R')
        elif current_node.last_move == 'R':
            current_node.sequence.remove('L')
        elif current_node.last_move == 'U':
            current_node.sequence.remove('D')
        elif current_node.last_move == 'D':
            current_node.sequence.remove('U')
    # sprawdzamy krańce macierzy stanu i blokujemy wyjście poza granice
    if current_node.zero["y"] == 0:
        current_node.sequence.remove('U')
    elif current_node.zero["y"] == len(current_node.puzzle) - 1:
        current_node.sequence.remove('D')
    if current_node.zero["x"] == 0:
        current_node.sequence.remove('L')
    elif current_node.zero["x"] == len(current_node.puzzle) - 1:
        current_node.sequence.remove('R')


# todo czym jest maksymalna osiagnieta glebokosc rekursji? dodac do kodu
def bfs(start_time):
    visited = 1
    processed = 1
    current_node = Node(START_PUZZLE, 'parentless', [], None)
    if is_goal(current_node.puzzle):
        return current_node.solution, len(current_node.solution),\
               processed, visited, round((time.time() - start_time) * 1000, 3)
    open_states = []
    closed_states = set()
    open_states.append(current_node)
    closed_states.add(current_node)
    # pętla zatrzyma sie gdy wszystkie stany otwarte zostaną przetworzone
    while open_states:
        v = open_states.pop(0)
        block_prohibited_moves(v)
        for n in v.sequence:
            v.move(n)
        for n in v.neighbours.values():
            processed += 1
            if is_goal(n.puzzle):
                return n.solution, len(n.solution),\
                       processed, visited, round((time.time() - start_time) * 1000, 3)
            if n not in closed_states:
                open_states.append(n)
                closed_states.add(n)
        visited += 1
    return False


# todo przemyslec gdzie wstawic processed i visited
def dfs(start_time):
    visited = 1
    processed = 1
    current_node = Node(START_PUZZLE, 'parentless', [], None)
    if is_goal(current_node.puzzle):
        return current_node.solution, len(current_node.solution),\
               processed, visited, round((time.time() - start_time) * 1000, 3)
    open_states = LifoQueue()
    closed_states = set()
    open_states.put(current_node)
    while not open_states.empty():
        v = open_states.get()
        block_prohibited_moves(v)
        if v not in closed_states:
            processed += 1
            closed_states.add(v)
            for n in list(reversed(v.sequence)):
                v.move(n)
            for n in v.neighbours.values():
                if is_goal(n.puzzle):
                    return n.solution, len(n.solution),\
                       processed, visited, round((time.time() - start_time) * 1000, 3)
                if len(n.solution) < DEPTH:
                    open_states.put(n)
        visited += 1
    return False


def astar(start_time, heuristic):
    processed = 1
    visited = 1
    current_node = Node(START_PUZZLE, 'parentless', [], None)
    if is_goal(current_node.puzzle):
        return current_node.solution, len(current_node.solution), \
               processed, visited, round((time.time() - start_time) * 1000, 3)
    p = PriorityQueue()
    t = set()
    p.put((0, current_node)) # nie jestem pewny czy tak powinno to wygladac
    while not p.empty():
        v = p.get()
        if is_goal(current_node.puzzle):
            return v.solution, len(current_node.solution), \
                   processed, visited, round((time.time() - start_time) * 1000, 3)
        t.add(v)




def main():
    start_time = time.time()
    print(bfs(start_time))


if __name__ == "__main__":
    main()
