# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import logging
from copy import deepcopy
import math
import random

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

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
        
        temp_board=self.Board(chess_board,my_pos,adv_pos,max_step)
        temp_Node=self.MCST_Node(temp_board,None,0)
        temp_Node.expend()
        print("found")
        return temp_Node.best_children()

    class Board:

        def __init__(self, chess_board, my_pos, adv_pos, max_step):
            self.chess_board=chess_board
            self.my_pos=my_pos
            self.adv_pos=adv_pos
            self.max_step=max_step
            self.initchessboard=deepcopy(chess_board)
            self.init_my_pos=my_pos
            self.init_adv_pos=adv_pos
            self.board_size=len(chess_board)

        def record(self):
            self.initchessboard=deepcopy(self.chess_board)
            self.init_my_pos=self.my_pos
            self.init_adv_pos=self.adv_pos

        def step(self, move, player):
            """
            tupple, int -> None
            move: ((x, y), dir)
            player: 0 or 1
            """

            if player==0:
                self.my_pos=move[0]
            else:
                self.adv_pos=move[0]
            self.chess_board[move[0][0]][move[0][1]][move[1]]=True
            if move[1]==0:
                self.chess_board[move[0][0]-1][move[0][1]][2]=True
            if move[1]==1:
                self.chess_board[move[0][0]][move[0][1]+1][3]=True
            if move[1]==2:
                self.chess_board[move[0][0]+1][move[0][1]][0]=True
            if move[1]==3:
                self.chess_board[move[0][0]][move[0][1]-1][1]=True
        
        def deep_copy(self):
            return StudentAgent.Board(deepcopy(self.chess_board),
                                        deepcopy(self.my_pos),
                                        deepcopy(self.adv_pos),
                                        self.max_step)

        def check_endgame(self):
            """
            Check if the game ends and compute the current score of the agents.

            Returns
            -------
            is_endgame : bool
                Whether the game ends.
            player_1_score : int
                The score of player 1.
            player_2_score : int
                The score of player 2.
            """
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
            # Union-Find
            father = dict()
            for r in range(self.board_size):
                for c in range(self.board_size):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(self.board_size):
                for c in range(self.board_size):
                    for dir, move in enumerate(
                        moves[1:3]
                    ):  # Only check down and right
                        if self.chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(self.board_size):
                for c in range(self.board_size):
                    find((r, c))
            p0_r = find(tuple(self.my_pos))
            p1_r = find(tuple(self.adv_pos))
            p0_score = list(father.values()).count(p0_r)
            p1_score = list(father.values()).count(p1_r)
            if p0_r == p1_r:
                return False, p0_score, p1_score
            return True, p0_score, p1_score

        def reset(self):
            self.chess_board=deepcopy(self.initchessboard)
            self.my_pos=self.init_my_pos
            self.adv_pos=self.init_adv_pos
        
        def check_boundary(self, pos):
            r, c = pos
            return 0 <= r < self.board_size and 0 <= c < self.board_size

    class MCST_Node:

        def __init__(self, board, move, player):
            """
            Board, tupple, int
            """
            self.board=board
            self.move=move
            self.player=player
            self.wins = 0
            self.simulation_times = 0
            self.children = []
            self.father=None

        def simulate(self, num):
            """
            int -> None
            """
            
            self.simulation_times+=num
            for i in range(num):
                is_end, p0_score, p1_score = self.board.check_endgame()
                player = (self.player + 1) % 2
                
                while not is_end:
                               
                    try:
                        move=self.random_step(self.board, player)
                        self.board.step(move,player)
                        is_end, p0_score, p1_score = self.board.check_endgame()
                    except ValueError:
                        is_end, p0_score, p1_score = self.board.check_endgame()
                        is_end = True
                    
                    player = (player + 1) % 2
                
                if p0_score>p1_score:
                    self.wins += 1
                
                self.board.reset()
            
            self.update_win_rates()
            

        def check_valid_step(self, start_pos, end_pos, barrier_dir):
            """
            Check if the step the agent takes is valid (reachable and within max steps).

            Parameters
            ----------
            start_pos : tuple
                The start position of the agent.
            end_pos : np.ndarray
                The end position of the agent.
            barrier_dir : int
                The direction of the barrier.
            """
            moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
           
            r, c = end_pos
            if 0 > r or r >= self.board.board_size or 0 > c or c >= self.board.board_size:
                return False
            
            if self.board.chess_board[r][c][barrier_dir]:
                return False

            if np.array_equal(start_pos, end_pos):
                return True
          

            # BFS
            visited = []
            cur_pos = [start_pos]
            for i in range(self.board.max_step):
                next_to = []
                for pos in cur_pos:
                    for dir in range(4):
                        if self.board.chess_board[pos[0]][pos[1]][dir]:
                            continue
                        new_pos = (pos[0] + moves[dir][0], pos[1] + moves[dir][1])
                        if new_pos in visited or new_pos == self.board.adv_pos:
                            continue
                        if new_pos == end_pos:
                            return True
                        visited.append(new_pos)
                        next_to.append(new_pos)
                cur_pos = next_to[:]
            return False


       
        def possibleMoves(self):
            
            my_pos=self.board.my_pos
            max_step=self.board.max_step
            result=[]
        
            while max_step >= 0:
                temp=0   
                while temp<=max_step:
                    end_pos=(my_pos[0]+max_step-temp,my_pos[1]+temp)
                    for num in range(4):
                        if self.check_valid_step(my_pos, end_pos, num):
                            result.append(((end_pos[0],end_pos[1]), num))
                    end_pos=(my_pos[0]+max_step-temp,my_pos[1]-temp)
                    for num in range(4):
                        if self.check_valid_step(my_pos, end_pos, num):
                            result.append(((end_pos[0],end_pos[1]), num))
                    end_pos=(my_pos[0]-max_step+temp,my_pos[1]+temp)
                    for num in range(4):
                        if self.check_valid_step(my_pos, end_pos, num):
                            result.append(((end_pos[0],end_pos[1]), num))
                    end_pos=(my_pos[0]-max_step+temp,my_pos[1]-temp)
                    for num in range(4):
                        if self.check_valid_step(my_pos, end_pos, num):
                            result.append(((end_pos[0],end_pos[1]), num))
                    temp+=1    
                max_step-=1
            
            return result    

        def random_step(self, board, player):
            all_moves = self.possibleMoves()
            
            i = len(all_moves) - 1
            r = random.randint(0, i)
            return all_moves[r]
            

        def update_win_rates(self):
            """
            None -> None
            """
            if self.father is None:
                return 
            ptr=self.father
            while ptr is not None:
                ptr.wins+=self.wins
                ptr.simulation_times+=self.simulation_times
                ptr=ptr.father

        def expend(self):
            temp=self.possibleMoves()
            board_size=self.board.board_size
            i=0
            if board_size=5:
                i=15
            elif board_size=6:
                i=8
            elif board_size=7:
                i=5
            elif board_size=8:
                i=2
            elif board_size=9:
                i=1
            for move in temp:
             
                new_board=self.board.deep_copy() 
                new_board.step(move,self.player)
                new_board.record()          
                child=StudentAgent.MCST_Node(new_board,move,self.player)             
                child.father=self
                child.simulate(i)
                self.children.append(child)
            

        def best_children(self):
            """
            None -> tupple ((x, y), dir)
            """
            num=0
            temp=None
            for child in self.children:
                if not self.check_valid_step(self.board.my_pos,child.move[0],child.move[1]):
                    continue

                is_end, p0_scord, p1_scord = child.board.check_endgame()
                
                if is_end:
                    if p0_scord > p1_scord:
                        return child.move
                    else:
                        continue
                Q=child.wins/child.simulation_times+math.sqrt(math.log10(self.simulation_times)/child.simulation_times)
                #print(child.wins)
                if Q>num:
                    num=Q
                    temp=child
            
            
            if temp is not None:
                return temp.move
                
            return self.board.my_pos, 0
