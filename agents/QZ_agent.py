# Student agent: Add your own agent here
import copy
import math
import random
import time

from agents.agent import Agent
from store import register_agent


@register_agent("QZ_agent")
class QZ_agent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(QZ_agent, self).__init__()
        self.name = "QZ_agent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True
        self.current_turn = 0
        self.board = None
        self.MCS_Tree = None
        self.last_step = None

        # init simulation times
        self.sim_dir   = {5 : 100,
                          6 : 30,
                          7 : 8,
                          8 : 3,
                          9 : 1}
        # init expansion times
        self.exp_dir   = {5 : 10,
                          6 : 8,
                          7 : 3,
                          8 : 1,
                          9 : 1}
        # extra simulation times
        self.exsim_dir = {5 : 40,
                          6 : 10,
                          7 : 5,
                          8 : 2,
                          9 : 1}

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        st = time.time()
        board_size = len(chess_board)
        if self.current_turn == 0:
            self.board = Board(chess_board, my_pos, adv_pos, max_step)
            self.MCS_Tree = MCSTree(self.board)
            self.MCS_Tree.max_sim = self.exp_dir[board_size]
            # print("\nBoard size: ", board_size)

            for _ in range(self.sim_dir[board_size]):
                self.MCS_Tree.expend()

            next_step = self.MCS_Tree.best_step()
            new_root = self.MCS_Tree.root.children[next_step]
            self.last_step = next_step

            self.MCS_Tree.update_root(new_root)

            if len(self.MCS_Tree.root.children) == 0:
                self.MCS_Tree.max_sim = 30
                for _ in range(1):
                    self.MCS_Tree.expend()

            self.current_turn += 1

            et = time.time()
            init_time = round(et - st, 3)
            # print("init time =", init_time, "sec")
            return next_step

        cur_board = Board(chess_board, my_pos, adv_pos, max_step)
        adv_step = self.board.get_step(cur_board, self.last_step)

        if adv_step not in self.MCS_Tree.root.children:
            self.MCS_Tree = MCSTree(cur_board)
        else:
            next_root = self.MCS_Tree.root.children[adv_step]
            self.MCS_Tree.update_root(next_root)

        mt = time.time()
        # no choices for current state
        if len(self.MCS_Tree.root.children) == 0:
            self.MCS_Tree.max_sim = self.exsim_dir[board_size]
            for _ in range(1):
                self.MCS_Tree.expend()

        next_step = self.MCS_Tree.best_step()

        # TO DO: (0, 0), 0
        next_root = self.MCS_Tree.root.children[next_step]
        self.MCS_Tree.update_root(next_root)

        # no choices for next state
        if len(self.MCS_Tree.root.children) == 0:
            self.MCS_Tree.max_sim = 0
            for _ in range(1):
                self.MCS_Tree.expend()

        self.last_step = next_step
        self.current_turn += 1

        et = time.time()
        init_time = round(et - mt, 3)
        # print("step time =", init_time, "sec")

        return next_step


class Board:
    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step

    def deep_copy(self):
        dup = copy.deepcopy(self)
        return dup

    def get_step(self, board, last_move):
        self.move_to(last_move, 0)
        cur_pos = board.adv_pos
        wall = 0
        for i in range(4):
            if self.chess_board[cur_pos[0], cur_pos[1], i] != board.chess_board[cur_pos[0], cur_pos[1], i]:
                wall = i
        return (cur_pos[0], cur_pos[1]), wall

    def check_endgame(self):
        board_size = int(math.sqrt(self.chess_board.size / 4))

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dire, move in enumerate(
                        moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dire + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))

        p0_r = find(tuple(self.my_pos))
        p1_r = find(tuple(self.adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score

        return True, p0_score, p1_score

    def move_to(self, move, player):
        if move is None:
            return
        self.chess_board[move[0][0], move[0][1], move[1]] = True
        if move[1] == 0:
            self.chess_board[move[0][0] - 1, move[0][1], 2] = True
        elif move[1] == 1:
            self.chess_board[move[0][0], move[0][1] + 1, 3] = True
        elif move[1] == 2:
            self.chess_board[move[0][0] + 1, move[0][1], 0] = True
        else:
            self.chess_board[move[0][0], move[0][1] - 1, 1] = True
        if player == 0:
            self.my_pos = move[0]
        else:
            self.adv_pos = move[0]


class Node:
    def __init__(self, father, player, board, max_sim, move):
        self.father = father
        self.player = player
        self.wins = 0
        self.simulations = max_sim
        self.max_sim = max_sim
        self.chess_board = board
        self.move = move

        self.simulation_board = self.chess_board.deep_copy()
        self.children = dict()

    def reset_board(self):
        self.simulation_board = self.chess_board.deep_copy()

    def run_simulation(self, max_sim):
        for _ in range(max_sim):
            # print(self.chess_board.my_pos)
            self.reset_board()
            turn = self.player
            is_end, p0_score, p1_score = self.simulation_board.check_endgame()
            while not is_end:
                player = turn % 2
                player_move = self.random_player_step(player)

                self.simulation_board.move_to(player_move, player)
                turn += 1
                is_end, p0_score, p1_score = self.simulation_board.check_endgame()

            if p0_score > p1_score:
                self.wins += 1

        self.update_simulations()

    def update_simulations(self):
        father = self.father
        while father is not None:
            father.wins += self.wins
            father.simulations += self.simulations
            father = father.father

    def q_star(self):
        if self.father is None:
            return 0
        return self.wins / self.simulations * math.sqrt(2 * math.log10(self.father.simulations))

    def expend(self, max_sim):
        chess_board = self.chess_board.chess_board
        my_pos = self.chess_board.my_pos
        adv_pos = self.chess_board.adv_pos
        max_step = self.chess_board.max_step

        # find all moves
        if self.player == 0:
            move_area = self.get_move_area(chess_board, my_pos, adv_pos, max_step)
        else:
            move_area = self.get_move_area(chess_board, adv_pos, my_pos, max_step)

        all_moves = []
        for move in move_area:
            r, c = move
            if list(self.chess_board.chess_board[r][c]).count(True) > 2:
                continue

            i = 0
            for wall in chess_board[move[0]][move[1]]:
                if not wall:
                    all_moves.append((move, i))
                i += 1

        # create new node
        for move in all_moves:
            new_chess_board = self.chess_board.deep_copy()
            new_chess_board.move_to(move, self.player)
            new_child = Node(self, (self.player + 1) % 2, new_chess_board, self.max_sim, move)
            self.children[move] = new_child
            new_child.run_simulation(max_sim)

    def random_player_step(self, player_num):
        """
        this method simulate a random player's move
        """
        chess_board_r = self.simulation_board.chess_board
        max_step_r = self.simulation_board.max_step
        if player_num == 0:
            my_pos_r = self.simulation_board.my_pos
            adv_pos_r = self.simulation_board.adv_pos
        else:
            my_pos_r = self.simulation_board.adv_pos
            adv_pos_r = self.simulation_board.my_pos

        valid_moves = self.get_move_area(chess_board_r, my_pos_r, adv_pos_r, max_step_r)
        if len(valid_moves) == 0:
            next_pos = my_pos_r
        else:
            i = random.randint(0, len(valid_moves) - 1)
            next_pos = valid_moves[i]

        next_dir = random.randint(0, 3)
        while chess_board_r[next_pos[0]][next_pos[1]][next_dir]:
            next_dir = random.randint(0, 3)
        return next_pos, next_dir

    @staticmethod
    def get_move_area(chess_board, my_pos, adv_pos, max_step):
        """
        improved (bool): if improved, it will not return the position with more than 2 barriers
        This method is to find all the available moves in current position by BFS
        return: list[(x,y)]
        """
        max_x, max_y = len(chess_board), len(chess_board[0])
        result = []
        moves = [my_pos]
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        def valid_move(x, y, x_max, y_max):
            return 0 <= x < x_max and 0 <= y < y_max

        for _ in range(max_step):
            next_move = []
            for pos in moves:
                r, c = pos

                # check four direction
                for key in dir_map:
                    direction = dir_map[key]

                    # block by wall
                    if chess_board[r][c][direction]:
                        continue

                    new_x, new_y = (r + directions[direction][0], c + directions[direction][1])

                    if valid_move(new_x, new_y, max_x, max_y) and \
                            (new_x, new_y) not in result and (new_x, new_y) != adv_pos:
                        next_move.append((new_x, new_y))
                        result.append((new_x, new_y))

            moves = next_move[:]
        return result


class MCSTree:
    def __init__(self, board):
        self.max_sim = 10
        self.root = Node(None, 0, board, self.max_sim, None)

    def expend(self):
        node_ptr = self.root
        while len(node_ptr.children) != 0:
            max_q = 0
            next_child = None
            for key in node_ptr.children:
                child = node_ptr.children[key]
                q = child.q_star()
                if q > max_q:
                    max_q = q
                    next_child = child
            node_ptr = next_child
        node_ptr.expend(self.max_sim)

    def update_root(self, node):
        self.root = node
        self.root.father = None

    def best_step(self):
        max_q = 0
        best_child = None
        for key in self.root.children:
            child = self.root.children[key]
            q = child.q_star()

            is_end, p0_score, p1_score = child.chess_board.check_endgame()
            if is_end and p0_score > p1_score:
                return child.move

            if q > max_q:
                max_q = q
                best_child = child
        if best_child is not None:
            return best_child.move
        return (0, 0), 0
