# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    def actionRequired(_from, _to):
        succs = problem.getSuccessors(_from)
        for succ in succs:
            s,d,c = succ
            if(_to == s):
                return d
        return None

    def traceResult(_trace):
        res = []
        flippedTrace = util.Stack()
        while(not _trace.isEmpty()):
            flippedTrace.push(_trace.pop())
        
        if(not flippedTrace.isEmpty()):
            node = flippedTrace.pop()
            while(not flippedTrace.isEmpty()):
                temp = node
                node = flippedTrace.pop()
                res.append(actionRequired(temp,node))
            return res
        return None

    front = util.Stack()
    trace = util.Stack()
    node = problem.getStartState()
    front.push(node) 
    visited = []
    parentMap = {}
    deadend = False
    while not front.isEmpty():
        node = front.pop()
        if(deadend):
            back = trace.pop()
            while(back != parentMap[node]):
                back = trace.pop()
                if(back == parentMap[node]):
                    trace.push(back)
            deadend = False
        trace.push(node)
        visited.append(node)
        if(problem.isGoalState(node)):
            return traceResult(trace)
        succs = problem.getSuccessors(node)
        deadend = True
        for succ in succs:
            s,d,c = succ
            parentMap[s] = node
            if(s not in visited):
                deadend = False
                front.push(s)
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    def actionRequired(_from, _to):
        succs = problem.getSuccessors(_from)
        for succ in succs:
            s,d,c = succ
            if(_to == s):
                return d
        return None

    def traceResult(path):
        res = []
        if(len(path) > 0):
            node = path.pop()
            while(len(path) > 0):
                temp = node
                node = path.pop()
                res.append(actionRequired(temp,node))
            return res
        return None

    queue = util.Queue()
    queuelist = []
    visited = []
    prevMap = {}
    queue.push(problem.getStartState())
    prevMap[problem.getStartState()] = problem.getStartState()
    while(queue):
        node = queue.pop()
        visited.append(node)
        if(problem.isGoalState(node)):
            path = []
            for i in range(len(prevMap)):
                if(node not in path):
                    path.append(node)
                    node = prevMap[node]
            return traceResult(path)
        for succ in problem.getSuccessors(node):
            s,_,_ = succ
            if(s not in visited and s not in queuelist):
                queue.push(s)
                queuelist.append(s)
                prevMap[s] = node
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"


    pQueue = util.PriorityQueue()
    visited = []
    graph_table = {}
    inQueue = {}

    node = problem.getStartState()
    pQueue.push(node,0) # insert the start node to the p-queue
    graph_table[node] = (node,0,None) #put the start node in the table with cost of 0
    while pQueue:
        node = pQueue.pop()
        visited.append(node)
        if(problem.isGoalState(node)):
            print(graph_table)
            res = []
            for key in graph_table:
                if(graph_table[key][2] is not None):
                    res.append(graph_table[key][2])
            print(res)
            return res
        succsessors = problem.getSuccessors(node) # check the node neighbors
        for succ in succsessors:
            s,d,c = succ
            if(s not in visited):
                pQueue.push(s, c)
                if(s not in graph_table or graph_table[s][1] > int(graph_table[node][1]) + c):
                    graph_table[s] = (node, graph_table[node][1] + c, d)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
