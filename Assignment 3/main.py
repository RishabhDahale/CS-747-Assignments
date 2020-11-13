import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1, help='random seed of the experiments')
parser.add_argument("--moves", default=4, help="Maximum number of moves possible from a position. For normal windy gridworld this should be 4, for kings will be 8 or 9")
parser.add_argument("--windDev", default=0, help="Deviation of the wind. Should be integer")
parser.add_argument("--maxTimeSteps", default=8000, help="Maximum Time Steps to allow the run")
parser.add_argument("--epsilon", default=0.1, help="Probability with which exploration should happen")
parser.add_argument("--alpha", default=0.5, help="Learning rate")
parser.add_argument("--algo", default="sarsa", choices=["sarsa", "expsarsa", "qlearning"], help="Algorithm to be used for learning")
parser.add_argument("--gamma", default=1, help="Discount factor to be used")
parser.add_argument("--endReward", default=1, help="Reward for the end state")
args = parser.parse_args()

ROWS = 7
COLS = 10
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
START = (3, 0)
END = (3, 7)
GAMMA = float(args.gamma)
ENDREWARD = float(args.endReward)
np.random.seed(int(args.seed))
LR = float(args.alpha)
ACTIONMAP = {
    0: "N", 1: "E", 2: "S", 3: "W", 4: "NE", 5: "SE", 6: "SW", 7: "NW", 8: "C"
}


class Grid:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.wind = -1*np.array(WIND)
        self.windDev = int(args.windDev)

        self.myPosI = START[0]
        self.myPosJ = START[1]

        self.epsCompleted = 0

    def move(self, direction: str):
        nextState = None
        if self.wind[self.myPosJ]:
            possibleI = np.array(range(-1 * self.windDev, self.windDev + 1)) + self.wind[self.myPosJ]
        else:
            possibleI = np.array([self.wind[self.myPosJ]])
        if direction.upper() == "N":
            possibleIN = possibleI + self.myPosI - 1
            nextState = (max(0, min(self.rows - 1, np.random.choice(possibleIN))), self.myPosJ)
        elif direction.upper() == "S":
            possibleIS = possibleI + self.myPosI + 1
            nextState = (max(0, min(self.rows - 1, np.random.choice(possibleIS))), self.myPosJ)
        elif direction.upper() == "E":
            possibleIE = possibleI + self.myPosI
            nextState = (max(0, min(self.rows - 1, np.random.choice(possibleIE))), max(0, min(self.cols - 1, self.myPosJ + 1)))
        elif direction.upper() == "W":
            possibleIE = possibleI + self.myPosI
            nextState = (max(0, min(self.rows - 1, np.random.choice(possibleIE))), max(0, min(self.cols - 1, self.myPosJ - 1)))
        elif direction.upper() == "NE":
            possibleIE = possibleI + self.myPosI - 1
            nextState = (max(0, min(self.rows - 1, np.random.choice(possibleIE))), max(0, min(self.cols - 1, self.myPosJ + 1)))
        elif direction.upper() == "SE":
            possibleIE = possibleI + self.myPosI + 1
            nextState = (max(0, min(self.rows - 1, np.random.choice(possibleIE))), max(0, min(self.cols - 1, self.myPosJ + 1)))
        elif direction.upper() == "SW":
            possibleIE = possibleI + self.myPosI + 1
            nextState = (max(0, min(self.rows - 1, np.random.choice(possibleIE))), max(0, min(self.cols - 1, self.myPosJ - 1)))
        elif direction.upper() == "NW":
            possibleIE = possibleI + self.myPosI - 1
            nextState = (max(0, min(self.rows - 1, np.random.choice(possibleIE))), max(0, min(self.cols - 1, self.myPosJ - 1)))
        elif direction.upper() == "C":
            possibleIE = possibleI + self.myPosI
            nextState = (max(0, min(self.rows - 1, np.random.choice(possibleIE))), self.myPosJ)

        self.myPosI, self.myPosJ = nextState[0], nextState[1]
        reward = -1
        if (self.myPosI == END[0]) and (self.myPosJ == END[1]):
            reward = ENDREWARD
        return self.myPosI, self.myPosJ, reward


def QLearning(Q: np.array, gamma, reward):
    target = reward + (gamma * max(Q))
    return target


def Sarsa(Q: np.array, gamma, reward, PI, nextI, nextJ):
    p = np.zeros(int(args.moves)) + (float(args.epsilon) / int(args.moves))
    p[PI[nextI, nextJ]] = p[PI[nextI, nextJ]] + 1 - float(args.epsilon)
    action = np.random.choice(list(range(int(args.moves))), p=p)
    target = reward + (gamma * Q[action])
    return target


def ExpSarsa(Q: np.array, gamma, reward, p):
    target = reward + (gamma * sum(p * Q))
    return target


def Learning(grid: Grid, maxTimeSteps: int, algo: str):
    episodesCompleted = np.zeros(maxTimeSteps)
    Q = np.zeros((ROWS, COLS, int(args.moves)))  # action 0->N, 1->E, 2->S, 3->W, 4->NE, 5->SE, 6->SW, 7->NW, 8->center
    PI = np.random.randint(int(args.moves), size=(ROWS, COLS))
    for t in range(1, maxTimeSteps + 1):
        p = np.zeros(int(args.moves)) + (float(args.epsilon) / int(args.moves))
        p[PI[grid.myPosI, grid.myPosJ]] = p[PI[grid.myPosI, grid.myPosJ]] + 1 - float(args.epsilon)
        action = np.random.choice(list(range(int(args.moves))), p=p)
        iCurr, jCurr = grid.myPosI, grid.myPosJ
        iNext, jNext, reward = grid.move(ACTIONMAP[action])
        target = None
        if algo == "qlearning":
            target = QLearning(Q[iNext, jNext], gamma=GAMMA, reward=reward)
        elif algo == "sarsa":
            target = Sarsa(Q[iNext, jNext], gamma=GAMMA, reward=reward, PI=PI, nextI=iNext, nextJ=jNext)
        elif algo == "expsarsa":
            target = ExpSarsa(Q[iNext, jNext], gamma=GAMMA, reward=reward, p=p)
        Q[iCurr, jCurr, action] = Q[iCurr, jCurr, action] + LR * (target - Q[iCurr, jCurr, action])
        # update policy
        temp = np.argwhere(Q[iCurr, jCurr] == max(Q[iCurr, jCurr]))
        PI[iCurr, jCurr] = np.random.choice(temp.reshape(temp.shape[0]))
        if reward != -1:
            grid.epsCompleted += 1
            grid.myPosI = START[0]
            grid.myPosJ = START[1]
        episodesCompleted[t-1] = grid.epsCompleted

    with open(os.path.join(os.getcwd(), "results", f"{args.algo}-{args.moves}-{args.windDev}-{args.seed}.csv"), "w+") as f:
        f.write("Time Step,Episodes\n")
        for t in range(maxTimeSteps):
            f.write(f"{t+1},{episodesCompleted[t]}\n")


if __name__=="__main__":
    if not os.path.exists("./results"):
        os.makedirs("./results")

    grid = Grid()
    Learning(grid, int(args.maxTimeSteps), args.algo)
