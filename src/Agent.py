import torch
import random
import numpy as np
from collections import deque
from Snake import SnakeGame, Direction, Point, BLOCK_SIZE
from Model import Linear_QNet, QTrainer
from Plotter import plot
from HyperParameters import MAX_MEMORY, BATCH_SIZE, LR, EPSILON, GAMMA, LAYERS_SIZES

class Agent:

    def __init__(self):
        self.games_number = 0
        self.epslon = EPSILON
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(LAYERS_SIZES)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (direction_right and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_left)) or
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)),

            # Danger right
            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_right and game.is_collision(point_down)),

            # Danger left
            (direction_down and game.is_collision(point_right)) or
            (direction_up and game.is_collision(point_left)) or
            (direction_right and game.is_collision(point_up)) or
            (direction_left and game.is_collision(point_down)),

            # Move direction
            direction_down,
            direction_left,
            direction_right,
            direction_up,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, over):
        self.memory.append((state, action, reward, next_state, over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, overs)

    def train_short_memory(self, state, action, reward, next_state, over):
        self.trainer.train_step(state, action, reward, next_state, over)

    def get_action(self, state):
        self.epslon = 80 - self.games_number
        move = [0, 0, 0]
        if  (random.randint(0, 200) < self.epslon):
            i = np.random.randint(0, 3)
            move[i] = 1
        else:
            prediction = self.model(torch.tensor(state, dtype=torch.float))
            move[torch.argmax(prediction).item()] = 1

        return move

def train():
    scores = []
    average_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        state = agent.get_state(game)
        move = agent.get_action(state)
        reward, game_over, score = game.play_step(move)
        new_state = agent.get_state(game)

        agent.train_short_memory(state, move, reward, new_state, game_over)
        agent.remember(state, move, reward, new_state, game_over)

        if(game_over):
            game.reset()
            agent.games_number += 1
            agent.train_long_memory()

            if score >= best_score:
                best_score = score
                agent.model.save()
            
            print(f"Game: {agent.games_number} | Score: {score} | Best Score: {best_score}")

            scores.append(score)
            total_score += score
            average_scores.append(total_score / agent.games_number)
            plot(scores, average_scores)

if __name__ == '__main__':
    train()