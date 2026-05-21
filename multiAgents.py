# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

MAZE_DISTANCE_CACHE = {}
FOOD_MST_CACHE = {}
INFINITY = float('inf')
MAZE_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GHOST_DANGER_DISTANCE = 1
GHOST_DANGER_PENALTY = -9999999
SCARED_GHOST_BONUS = 220.0
SCARED_GHOST_TIMER_BONUS = 4.0
DISTANT_SCARED_GHOST_BONUS = 30.0
CAPSULE_COUNT_PENALTY = 8.0
CAPSULE_DANGER_DISTANCE = 6
NEAR_CAPSULE_BONUS = 120.0
FAR_CAPSULE_BONUS = 25.0


def mazeDistance(start, goal, walls):
    if start == goal:
        return 0

    layoutKey = hash(walls)
    if start <= goal:
        cacheKey = (layoutKey, start, goal)
    else:
        cacheKey = (layoutKey, goal, start)
    if cacheKey in MAZE_DISTANCE_CACHE:
        return MAZE_DISTANCE_CACHE[cacheKey]

    queue = util.Queue()
    queue.push((start, 0))
    visited = {start}

    while not queue.isEmpty():
        (x, y), distance = queue.pop()
        for dx, dy in MAZE_DIRECTIONS:
            nextPos = (x + dx, y + dy)
            if nextPos in visited or walls[nextPos[0]][nextPos[1]]:
                continue

            if nextPos == goal:
                MAZE_DISTANCE_CACHE[cacheKey] = distance + 1
                return distance + 1

            visited.add(nextPos)
            queue.push((nextPos, distance + 1))

    return INFINITY


def calculateFoodMstCost(foodPositions, walls):
    if len(foodPositions) <= 1:
        return 0

    cacheKey = (hash(walls), tuple(sorted(foodPositions)))
    if cacheKey in FOOD_MST_CACHE:
        return FOOD_MST_CACHE[cacheKey]

    remaining = set(foodPositions)
    connected = {remaining.pop()}
    mstCost = 0

    while remaining:
        bestFood = None
        bestDistance = None
        for source in connected:
            for target in remaining:
                distance = mazeDistance(source, target, walls)
                if bestDistance is None or distance < bestDistance:
                    bestDistance = distance
                    bestFood = target

        mstCost += bestDistance
        connected.add(bestFood)
        remaining.remove(bestFood)

    FOOD_MST_CACHE[cacheKey] = mstCost
    return mstCost


def foodGhostEvaluation(gameState: GameState):
    pacmanPosition = gameState.getPacmanPosition()
    foodPositions = gameState.getFood().asList()
    walls = gameState.getWalls()

    nearestFoodDistance = 0
    foodMstCost = 0
    if foodPositions:
        nearestFoodDistance = INFINITY
        for foodPosition in foodPositions:
            foodDistance = mazeDistance(pacmanPosition, foodPosition, walls)
            if foodDistance < nearestFoodDistance:
                nearestFoodDistance = foodDistance
        foodMstCost = calculateFoodMstCost(foodPositions, walls)

    minGhostDistance = INFINITY
    for ghostPosition in gameState.getGhostPositions():
        if ghostPosition is None:
            continue
        ghostDistance = mazeDistance(pacmanPosition, ghostPosition, walls)
        if ghostDistance < minGhostDistance:
            minGhostDistance = ghostDistance

    ghostDistanceScore = GHOST_DANGER_PENALTY
    if minGhostDistance > GHOST_DANGER_DISTANCE:
        ghostDistanceScore = 0

    return gameState.getScore() - (nearestFoodDistance + foodMstCost) + ghostDistanceScore


def capsuleScaredGhostBonus(gameState: GameState):
    pacmanPosition = gameState.getPacmanPosition()
    walls = gameState.getWalls()
    capsulePositions = gameState.getCapsules()
    ghostStates = gameState.getGhostStates()

    bonus = 0.0
    activeGhostDistances = []

    for ghostState in ghostStates:
        ghostPosition = ghostState.getPosition()
        if ghostPosition is None:
            continue

        ghostDistance = mazeDistance(pacmanPosition, ghostPosition, walls)
        if ghostState.scaredTimer > 0:
            if ghostDistance <= ghostState.scaredTimer:
                bonus += SCARED_GHOST_BONUS / (ghostDistance + 1)
                bonus += SCARED_GHOST_TIMER_BONUS * (ghostState.scaredTimer - ghostDistance)
            else:
                bonus += DISTANT_SCARED_GHOST_BONUS / (ghostDistance + 1)
        else:
            activeGhostDistances.append(ghostDistance)

    if capsulePositions:
        nearestCapsuleDistance = INFINITY
        for capsulePosition in capsulePositions:
            capsuleDistance = mazeDistance(pacmanPosition, capsulePosition, walls)
            if capsuleDistance < nearestCapsuleDistance:
                nearestCapsuleDistance = capsuleDistance

        nearestActiveGhostDistance = INFINITY
        for activeGhostDistance in activeGhostDistances:
            if activeGhostDistance < nearestActiveGhostDistance:
                nearestActiveGhostDistance = activeGhostDistance

        bonus -= CAPSULE_COUNT_PENALTY * len(capsulePositions)
        if nearestActiveGhostDistance <= CAPSULE_DANGER_DISTANCE:
            bonus += NEAR_CAPSULE_BONUS / (nearestCapsuleDistance + 1)
        else:
            bonus += FAR_CAPSULE_BONUS / (nearestCapsuleDistance + 1)

    return bonus

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return foodGhostEvaluation(successorGameState)

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isTerminalState(self, gameState, depth):
        if depth == self.depth:
            return True
        if gameState.isWin():
            return True
        return gameState.isLose()

    def nextAgentIndex(self, agentIndex, numAgents):
        return (agentIndex + 1) % numAgents

    def nextDepth(self, depth, nextAgentIndex):
        nextDepth = depth
        if nextAgentIndex == 0:
            nextDepth += 1
        return nextDepth

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState, depth, agentIndex, numAgents):
        if self.isTerminalState(gameState, depth):
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(gameState)

        nextAgentIndex = self.nextAgentIndex(agentIndex, numAgents)
        nextDepth = self.nextDepth(depth, nextAgentIndex)

        if agentIndex == 0:
            value = -INFINITY
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                successorValue = self.minimax(successor, nextDepth, nextAgentIndex, numAgents)
                if successorValue > value:
                    value = successorValue
            return value

        value = INFINITY
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.minimax(successor, nextDepth, nextAgentIndex, numAgents)
            if successorValue < value:
                value = successorValue
        return value

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        numAgents = gameState.getNumAgents()
        bestAction = None
        bestValue = -INFINITY

        for action in gameState.getLegalActions(0):
            value = self.minimax(gameState.generateSuccessor(0, action), 0, 1 % numAgents, numAgents)
            if bestAction is None or value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, gameState, depth, agentIndex, numAgents, alpha, beta):
        if self.isTerminalState(gameState, depth):
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(gameState)

        nextAgentIndex = self.nextAgentIndex(agentIndex, numAgents)
        nextDepth = self.nextDepth(depth, nextAgentIndex)

        if agentIndex == 0:
            value = -INFINITY
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                successorValue = self.alphabeta(successor, nextDepth, nextAgentIndex, numAgents, alpha, beta)
                if successorValue > value:
                    value = successorValue
                if value > beta:
                    return value
                if value > alpha:
                    alpha = value
            return value

        value = INFINITY
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            successorValue = self.alphabeta(successor, nextDepth, nextAgentIndex, numAgents, alpha, beta)
            if successorValue < value:
                value = successorValue
            if value < alpha:
                return value
            if value < beta:
                beta = value
        return value

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()
        bestAction = None
        bestValue = -INFINITY
        alpha = -INFINITY
        beta = INFINITY

        for action in gameState.getLegalActions(0):
            value = self.alphabeta(gameState.generateSuccessor(0, action), 0, 1 % numAgents, numAgents, alpha, beta)
            if bestAction is None or value > bestValue:
                bestValue = value
                bestAction = action
            if bestValue > alpha:
                alpha = bestValue

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, gameState, depth, agentIndex, numAgents):
        if self.isTerminalState(gameState, depth):
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(gameState)

        nextAgentIndex = self.nextAgentIndex(agentIndex, numAgents)
        nextDepth = self.nextDepth(depth, nextAgentIndex)

        if agentIndex == 0:
            value = -INFINITY
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                successorValue = self.expectimax(successor, nextDepth, nextAgentIndex, numAgents)
                if successorValue > value:
                    value = successorValue
            return value

        totalValue = 0.0
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            totalValue += self.expectimax(successor, nextDepth, nextAgentIndex, numAgents)
        return totalValue / len(legalActions)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        numAgents = gameState.getNumAgents()
        bestAction = None
        bestValue = -INFINITY

        for action in gameState.getLegalActions(0):
            value = self.expectimax(gameState.generateSuccessor(0, action), 0, 1 % numAgents, numAgents)
            if bestAction is None or value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Reuses the reflex-agent heuristic on full states, then adds
    incentives for moving toward capsules while ghosts are still dangerous and
    for chasing scared ghosts that can still be eaten in time.
    """
    return foodGhostEvaluation(currentGameState) + capsuleScaredGhostBonus(currentGameState)

# Abbreviation
better = betterEvaluationFunction
