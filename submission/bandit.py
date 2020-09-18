import argparse, math, sys
import numpy as np

# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--instance", help="path to the instance file")
parser.add_argument("--algorithm", choices=["epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling",
                                            "thompson-sampling-with-hint"], help="Algorithm to run")
parser.add_argument("--randomSeed", help="Random seed to use. Non negative integer")
parser.add_argument("--epsilon", help="number in [0, 1]")
parser.add_argument("--horizon", help="Non-negative integer")
args = parser.parse_args()

# utility functions/classes
class MAB:
    def __init__(self, instance):
        self.instance = instance
        self.actualProb = []
        with open(self.instance, 'r') as f:
            lines = f.readlines()
            for value in lines:
                self.actualProb.append(float(value))
        self.actualProb = np.array(self.actualProb)
        self.rewards = np.zeros_like(self.actualProb)
        self.playCount = np.zeros_like(self.actualProb)

    def sampleRewardAndUpdate(self, arm):
        """
        Updates the rewards[arm] and netReward
        :param arm: Arm to sample, should be zero indexed
        """
        reward = np.random.binomial(1, self.actualProb[arm])
        self.rewards[arm] += reward
        self.playCount[arm] += 1
        return reward


# defining the functions
# every function should return the REG i.e. regret of that run
def eGreedy(instance, epsilon, horizon):
    mab = MAB(instance)
    n = mab.actualProb.shape[0]     # number of arms
    for t in range(n):
        mab.sampleRewardAndUpdate(t)
    for t in range(n, horizon):
        e = np.random.uniform(0, 1)
        if e < epsilon:
            # sample the arm uniformly
            playArm = np.random.choice(n)
            mab.sampleRewardAndUpdate(playArm)
        else:
            # sample the arm with highest emprical mean
            empMean = np.array([mab.rewards[arm]/mab.playCount[arm] for arm in range(n)])
            playArm = np.argmax(empMean)
            mab.sampleRewardAndUpdate(playArm)
    REG = (horizon*max(mab.actualProb)) - sum(mab.rewards)
    return REG


def UCB(instance, horizon):
    mab = MAB(instance)
    n = mab.actualProb.shape[0]  # number of arms
    # sample each arm once so that mab.playCount is not zero and ln(t) is not -infinity
    for t in range(n):
        mab.sampleRewardAndUpdate(t)
    for t in range(n, horizon):
        ucb = np.zeros_like(mab.actualProb)
        empProb = np.array([mab.rewards[arm]/mab.playCount[arm] for arm in range(n)])
        ucb = empProb + np.sqrt(2*math.log(t)/mab.playCount)
        playArm = np.argmax(ucb)
        mab.sampleRewardAndUpdate(playArm)
    REG = (horizon * max(mab.actualProb)) - sum(mab.rewards)
    return REG


def klUCB(instance, horizon):
    mab = MAB(instance)
    n = mab.actualProb.shape[0]  # number of arms
    def klDiv(p, q):
        # assuming log as natural log
        if (q==1) or (q==0):
            return sys.float_info.max
        if p==0:
            return math.log(1/(1-q))
        elif p==1:
            return math.log(1/q)
        return (p*math.log(p/q)) + ((1-p)*math.log((1-p)/(1-q)))

    def binarySearchQ(p, t, c=3):
        tolerance = 1e-5       # constant for this function
        RHS = math.log(t) + (c*math.log(math.log(t)))
        leftEnd = p; rightEnd = 1
        # leftVal = 0; rightVal = klDiv(p, 1)
        point = (leftEnd + rightEnd)/2; val = klDiv(p, point)
        while (RHS - val)>tolerance:
            if (RHS - val)>0:
                leftEnd = point + 0
                point = (leftEnd + rightEnd)/2; val = klDiv(p, point)
            else:
                rightEnd = point + 0
                point = (leftEnd + rightEnd)/2; val = klDiv(p, point)
        return point

    # sample each arm once so that mab.playCount is not zero and ln(t) is not -infinity
    for t in range(n):
        mab.sampleRewardAndUpdate(t)
    for t in range(n, horizon):
        empMean = [mab.rewards[arm]/mab.playCount[arm] for arm in range(n)]
        ucb_kl = np.array([binarySearchQ(empMean[arm], t) for arm in range(n)])
        playArm = np.argmax(ucb_kl)
        mab.sampleRewardAndUpdate(playArm)
    REG = (horizon * max(mab.actualProb)) - sum(mab.rewards)
    return REG


def tSampling(instance, horizon):
    mab = MAB(instance)
    n = mab.actualProb.shape[0]  # number of arms
    for t in range(horizon):
        betaSampling = np.array([np.random.beta(mab.rewards[arm]+1, mab.playCount[arm] - mab.rewards[arm] + 1) for arm in range(n)])
        playArm = np.argmax(betaSampling)
        mab.sampleRewardAndUpdate(playArm)
    REG = (horizon * max(mab.actualProb)) - sum(mab.rewards)
    # print(REG)
    return REG


def tSamplingHint(instance, horizon):
    mab = MAB(instance)
    n = mab.actualProb.shape[0]  # number of arms
    HINT = np.sort(mab.actualProb)
    armBelief = np.zeros((n, n)) + (1/n)
    print(armBelief)
    for t in range(horizon):
        probSampling = []
        for i in range(n):
            j = np.random.choice(list(range(n)), p=armBelief[i][:])
            probSampling.append(HINT[j]) # beta(mean j)
        playArm = np.argmax(probSampling)
        playArmProb = probSampling[playArm]
        # breaking tie
        commonArms = np.sum(playArmProb==np.array(probSampling))
        if commonArms!=1:
            arms = [i for i in range(n) if probSampling[i]==playArmProb]    # indexes whose probSampling is same as max value
            hintValueIndex = [i for i in range(n) if HINT[i]==playArmProb][0]
            prob_of_sampling = np.array([armBelief[i][hintValueIndex] for i in arms])
            prob_of_sampling = prob_of_sampling / np.sum(prob_of_sampling)
            # playArmIndex = np.argmax(prob_of_sampling)
            playArmIndex = np.random.choice(arms, p=prob_of_sampling)
            playArm = arms[arms.index(playArmIndex)]
        reward = mab.sampleRewardAndUpdate(playArm)
        for i in range(n):
            p = armBelief[playArm][i]
            q = HINT[i]
            armBelief[playArm][i] = p * ((q**reward)*((1-q)**(1-reward)))
        armBelief[playArm][:] = armBelief[playArm][:] / np.sum(armBelief[playArm][:])
        # if t%20==0:
        #     print(armBelief, reward, "\n")
    REG = (horizon * max(mab.actualProb)) - sum(mab.rewards)
    print(armBelief, REG)
    return REG


algo = args.algorithm
np.random.seed(int(args.randomSeed))
params = {"instance": args.instance,
          "epsilon": float(args.epsilon),
          "horizon": int(args.horizon)}
# algo = "thompson-sampling-with-hint"
# params = {"instance": '../instances/i-2.txt',
#           "epsilon": 0.02,
#           "horizon": 100}

file = open("outputDataT2.txt", "a+")
if algo == "epsilon-greedy":
    REG = eGreedy(params['instance'], params['epsilon'], params['horizon'])
    file.write(f"{params['instance']},{algo},{args.randomSeed},{params['epsilon']},{params['horizon']},{REG}\n")
elif algo == "ucb":
    REG = UCB(params['instance'], params['horizon'])
    file.write(f"{params['instance']},{algo},{args.randomSeed},{params['epsilon']},{params['horizon']},{REG}\n")
elif algo == "kl-ucb":
    REG = klUCB(params['instance'], params['horizon'])
    file.write(f"{params['instance']},{algo},{args.randomSeed},{params['epsilon']},{params['horizon']},{REG}\n")
elif algo == "thompson-sampling":
    REG = tSampling(params['instance'], params['horizon'])
    file.write(f"{params['instance']},{algo},{args.randomSeed},{params['epsilon']},{params['horizon']},{REG}\n")
elif algo == "thompson-sampling-with-hint":
    REG = tSamplingHint(params['instance'], params['horizon'])
    file.write(f"{params['instance']},{algo},{args.randomSeed},{params['epsilon']},{params['horizon']},{REG}\n")
file.close()
