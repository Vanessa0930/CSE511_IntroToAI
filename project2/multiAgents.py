# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
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

  def evaluationFunction(self, currentGameState, action):
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
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    
    if successorGameState.isWin():
        return 999999
    if successorGameState.isLose():
        return -999999
    
    currentPos = currentGameState.getPacmanPosition()
    currentFoodList = currentGameState.getFood().asList()
    
    # Calculate base score, which is the score of successor game state
    finalScore = successorGameState.getScore()
    
    if len(currentFoodList) < len(newFood.asList()):
        finalScore = finalScore + 10 * (len(newFood.asList()) - len(currenFoodList))
        
    minDistToNewFood = min([util.manhattanDistance(newPos, food) for food in newFood.asList()])
    currentDistToNewFood = min([util.manhattanDistance(currentPos, food) for food in newFood.asList()])
    if minDistToNewFood < currentDistToNewFood:
        finalScore = finalScore + 50 * ( currentDistToNewFood - minDistToNewFood )
    
    for ghostState in newGhostStates:
        ghostPos = ghostState.getPosition()
        scaredTimer = ghostState.scaredTimer
        if scaredTimer > ghostPos:
            finalScore = finalScore + 20 * ghostPos
    
    return finalScore

def scoreEvaluationFunction(currentGameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        
      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    # Reference: https://github.com/douglaschan32167/multiagent/blob/master
    
    actions = gameState.getLegalActions()
    self.removeStop(actions)
    
    numOfAgents = gameState.getNumAgents()
    bestAction = Directions.STOP
    score = float('-inf')
    
    for action in actions:
        successorState = gameState.generateSuccessor(0, action)
        previousScore = score
        score = max(score, self.minValue(successorState, 0, 1, numOfAgents - 1))
        
        if score > previousScore:
            bestAction = action
    
    return bestAction
     
   
  def maxValue(self, gameState, depth, index, numOfGhosts):
    if self.isTerminalState(gameState, depth):
        return self.evaluationFunction(gameState)
    
    value = float('-inf')
    actions = gameState.getLegalActions(0)
    #remove STOP action from legal action list
    self.removeStop(actions)
    
    for action in actions:
        successorState = gameState.generateSuccessor(0, action)
        value = max(value, self.minValue(successorState, depth + 1, index + 1, numOfGhosts))
    
    return value
 
  
  def minValue(self, gameState, depth, index, numOfGhosts):
    if self.isTerminalState(gameState, depth):
        return self.evaluationFunction(gameState)
    
    value = float('inf')
    actions = gameState.getLegalActions(index)
    
    if index == numOfGhosts:
        for action in actions:
            successorState = gameState.generateSuccessor(index, action)
            value = min(value, self.maxValue(successorState, depth + 1, 0, numOfGhosts))
    else:
        for action in actions:
            successorState = gameState.generateSuccessor(index, action)
            value = min(value, self.minValue(successorState, depth, index + 1, numOfGhosts))
    
    return value

  # Remove Directions.STOP from possible actions
  def removeStop(self, legalActions):
    stopAction = Directions.STOP
    if stopAction in legalActions:
        legalActions.remove(stopAction)
  
  # Return true if the gameState is a terminal state
  def isTerminalState(self, gameState, depth):
    return gameState.isWin() or gameState.isLose() or depth == self.depth
    

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    actions = gameState.getLegalActions()
    self.removeStop(actions)
    
    numOfAgents = gameState.getNumAgents()
    bestAction = Directions.STOP
    score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for action in actions:
        successorState = gameState.generateSuccessor(0, action)
        previousScore = score
        score = max(score, self.minValue(successorState, 0, 1, numOfAgents - 1, alpha, beta))
        
        if score > previousScore:
            bestAction = action
        if score >= beta:
            return bestAction
        alpha = max(alpha, score)
    
    return bestAction
    #util.raiseNotDefined()


  def maxValue(self, gameState, depth, numOfGhosts, alpha, beta):
    if self.isTerminalState(gameState, depth):
        return self.evaluationFunction(gameState)
    
    value = float('-inf')
    actions = gameState.getLegalActions(0)
    self.removeStop(actions)
    
    for action in actions:
        successorState = gameState.generateSuccessor(0, action)
        value = max(value, self.minValue(successorState, depth, 1, numOfGhosts, alpha, beta))
        if value >= beta:
            return value
        alpha = max(alpha, value)
    
    return value
    
  
  def minValue(self, gameState, depth, index, numOfGhosts, alpha, beta):
    if self.isTerminalState(gameState, depth):
        return self.evaluationFunction(gameState)
    
    value = float('inf')
    actions = gameState.getLegalActions(index)
    
    if index == numOfGhosts:
        for action in actions:
            successorState = gameState.generateSuccessor(index, action)
            value = min(value, self.maxValue(successorState, depth + 1, numOfGhosts, alpha, beta))
            if value <= alpha:
                return value
            beta = min(value, beta)
    else:
        for action in actions:
            successorState = gameState.generateSuccessor(index, action)
            value = min(value, self.minValue(successorState, depth , index+1, numOfGhosts, alpha, beta))
            if value <= alpha:
                return value
            beta = min(value, beta)
    
    return value
  
  def isTerminalState(self, gameState, depth):
    return gameState.isWin() or gameState.isLose() or depth == self.depth
  
  # Remove Directions.STOP from possible actions
  def removeStop(self, legalActions):
    stopAction = Directions.STOP
    if stopAction in legalActions:
        legalActions.remove(stopAction)

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    actions = gameState.getLegalActions()
    self.removeStop(actions)
    
    numOfAgents = gameState.getNumAgents()
    bestAction = Directions.STOP
    score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for action in actions:
        successorState = gameState.generateSuccessor(0, action)
        previousScore = score
        score = max(score, self.expValue(successorState, 0, 1, alpha, beta))
        
        if score > previousScore:
            bestAction = action
        if score > beta:
            return bestAction
        alpha = max(alpha, score)
    
    return bestAction
  
  def maxValue(self, gameState, depth, alpha, beta):
    if self.isTerminalState(gameState, depth):
        return self.evaluationFunction(gameState)
    
    value = float('-inf')
    actions = gameState.getLegalActions(0)
    self.removeStop(actions)
    
    for action in actions:
        successorState = gameState.generateSuccessor(0, action)
        value = max(value, self.expValue(successorState, depth, 1, alpha, beta))
        if value >= beta:
            return value
        alpha = max(alpha, value)
    
    return value
  
  def expValue(self, gameState, depth, index, alpha, beta):
    if self.isTerminalState(gameState, depth):
        return self.evaluationFunction(gameState)
    
    value = 0
    numOfGhosts = gameState.getNumAgents() - 1
    actions = gameState.getLegalActions(index)
    numOfActions = len(actions)
    
    if index == numOfGhosts:
        for action in actions:
            successorState = gameState.generateSuccessor(index, action)
            value = value + 1 * self.maxValue(successorState, depth + 1, alpha, beta)
            if value <= alpha:
                return value / numOfActions
            beta = min(value, beta)
    else:
        for action in actions:
            successorState = gameState.generateSuccessor(index, action)
            value = value + 1 * self.expValue(successorState, depth, index + 1, alpha, beta)
            if value <= alpha:
                return value / numOfActions
            beta = min(value, beta)
    
    return value / numOfActions
  
  def isTerminalState(self, gameState, depth):
    return gameState.isWin() or gameState.isLose() or depth == self.depth
  
  def removeStop(self, legalActions):
    stopAction = Directions.STOP
    if stopAction in legalActions:
        legalActions.remove(stopAction)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    1. Distance to closest food: score -= 2 * dist
    2. Current score: score += 100 * score
    3. Distance to closest actived ghost (negative effect):
       	Only consider actived ghosts within 5 distances
	score -= 2 * 1/dist
	The closer the ghost is, the smaller the score will be.
    4. Number of food left: socre -= 2 * num 
    5. Number of capsules left: score -= 10 * num
    Reference: http://dcalacci.net/blog/pacman-machine-learning-expectimax-agent/ 
  """
  "*** YOUR CODE HERE ***"
  def findNextClosestAgent(state, cornerList):
    if len(cornerList) == 0:
        # only one corner in the list
        return None
    
    closestCorner = cornerList[0]
    minDistance = util.manhattanDistance(state, closestCorner)
    for corner in cornerList[1:]:
        distance = util.manhattanDistance(state, corner)
        if minDistance > distance:
            minDistance = distance
            closestCorner = corner
    return closestCorner
  
  
  # Returen +/-infinity for win or loss states
  if currentGameState.isWin():
    return float('inf')
  if currentGameState.isLose():
    return float('-inf')
  
  pacmanPos = currentGameState.getPacmanPosition()
  finalScore = 0
  
  # Get current score and update finalScore
  currentScore = currentGameState.getScore()
  finalScore = finalScore + 100 * currentScore
  
  # Get the distance to closest food and add up finalScore accordingly
  foodList = currentGameState.getFood().asList()
  closestFood = findNextClosestAgent(pacmanPos, foodList)
  finalScore = finalScore - 2 * util.manhattanDistance(closestFood, pacmanPos)
  
  # Get the distance to closest actived ghost
  ghostStateList = currentGameState.getGhostStates()
  activedGhosts = []
  for ghost in ghostStateList:
    if not ghost.scaredTimer:
        activedGhosts.append(ghost.getPosition())
  if len(activedGhosts) > 0:
    closestActivedGhost = findNextClosestAgent(pacmanPos, activedGhosts)
    dist = util.manhattanDistance(pacmanPos, closestActivedGhost)
    finalScore = finalScore - 5 * 1/(max(5, dist)) # only consider ghosts within 5 distances
  
  # Get number of food left
  numOfFood = len(foodList)
  finalScore = finalScore - 2 * numOfFood
  
  # Get number of capsules left
  numOfCapsules = len(currentGameState.getCapsules())
  finalScore = finalScore - 10 * numOfCapsules
  
  return finalScore
  
  

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

