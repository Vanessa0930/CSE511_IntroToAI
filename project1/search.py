# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    
    if problem.isGoalState(problem.getStartState()):
        return []
    
    # Use the stack to implement DFS
    fringe = util.Stack()
    # Initialize result and visitedNode list
    visitedStates = []
    
    # Each node is (state, direction, parentNode)
    currentNode = (problem.getStartState(), [])
    fringe.push(currentNode)
    
    while not fringe.isEmpty():
        currentNode = fringe.pop()
        currentState = currentNode[0]
        solution = currentNode[1]
        
        if currentNode not in visitedStates:
            # Goal test
            if problem.isGoalState(currentState):
                return solution
            else:
                visitedStates.append(currentState)
                
                # Add successors to the fringe
                for successor in problem.getSuccessors(currentState):
                    state = successor[0]
                    if state not in visitedStates:
                        if problem.isGoalState(state):
                            return solution+[successor[1]]
                        fringe.push((state, solution+[successor[1]]))
    
    return []
    
"""
Q1 Answers:
        1. Yes the exploration order is what I expected
        2. No, Pacman does not go to all explored squares on his way to goal.

Reference: https://github.com/shiro873/pacman-projects/
Reference: https://github.com/jeknov/aiAlgorithmsWithPacman/
"""

    

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []
    
    fringe = util.Queue()
    explored = []
    # Insert start node into queue
    node = ((problem.getStartState(), []))
    fringe.push(node)
    
    while not fringe.isEmpty():
        node = fringe.pop()
        currentState = node[0]
        solution = node[1]
        
        if currentState not in explored:
            if problem.isGoalState(currentState):
                return solution
            else:
                explored.append(currentState)
        
                for successor in problem.getSuccessors(currentState):
                    successorState = successor[0]
                    successorAction = successor[1]
            
                    if successorState not in explored:
                        nextSolution = solution + [successorAction]
                        if problem.isGoalState(successorState):
                            return nextSolution
                        fringe.push((successorState, nextSolution))
    
    return []
    
"""
Q2 Answers:
    BFS returns a least cost solution
                BFS                 DFS
    tinyMaze    502(length of 8)    500(length of 16)
    mediumMaze  442(length of 68)   380(length of 130)
    bigMaze     300(length of 210)  300(length of 210)

Reference: https://github.com/shiro873/pacman-projects/
Reference: https://github.com/jeknov/aiAlgorithmsWithPacman/
"""


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    # node = (state, path, path-cost)
    # use parth-cost as priority, initialized as 0
    if problem.isGoalState(problem.getStartState()):
        return []
        
    pathCost = 0
    node = (problem.getStartState(), [], pathCost)
    
    fringe = util.PriorityQueue()
    explored = []
    fringe.push(node, node[2])
    
    while not fringe.isEmpty():
        node = fringe.pop()
        nodeState = node[0]
        solution = node[1]
        pathCost = node[2]
        
        # Goal test
        if problem.isGoalState(nodeState):
            return solution
        else:
            explored.append(nodeState)
            for successor in problem.getSuccessors(nodeState):
                successorState = successor[0]
                successorDirection = successor[1]
                
                if successorState not in explored:
                    nextCost = pathCost + successor[2]
                    nextSolution = solution + [successorDirection]
                    fringe.push((successorState, nextSolution, nextCost), nextCost)
    
    return []
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    #Reference: https://github.com/shiro873/pacman-projects/
    #Reference: https://github.com/jeknov/aiAlgorithmsWithPacman/
    
    pathCost = 0
    heuristicCost = heuristic(problem.getStartState(), problem) + pathCost # f = h + g
    node = (problem.getStartState(), [], pathCost)
    
    fringe = util.PriorityQueue()
    explored = []
    
    fringe.push(node, heuristicCost)
    
    while not fringe.isEmpty():
        node = fringe.pop()
        nodeState = node[0]
        solution = node[1]
        pathCost = node[2]
        
        if problem.isGoalState(nodeState):
            return solution
        else:
            explored.append(nodeState)
            
            for successor in problem.getSuccessors(nodeState):
                successorState = successor[0]
                successorDirection = successor[1]
                successorCost = successor[2]
                
                if successorState not in explored:
                    # use heuristic function to calculate distance between successor and node
                    # new path cost
                    nextPathCost = pathCost + successorCost
                    successorHeuristicsCost = heuristic(successorState, problem) + nextPathCost 
                    nextSolution = solution + [successorDirection]
                    
                    fringe.push((successorState, nextSolution, nextPathCost), successorHeuristicsCost)
    
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
