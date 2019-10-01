################################################################################
# Solving simple tic tac toe with MCTS
################################################################################

include("tictactoe.jl")

using AlphaZero.MCTS

################################################################################

const PROFILE = false
const INTERACTIVE = true

const INITIAL_TRAINING = 50_000
const TIMEOUT = 5_000

################################################################################
# Write the evaluator

struct RolloutEvaluator end

function rollout(board)
  state = State(copy(board), first_player=Red)
  while true
    reward = GI.white_reward(state)
    isnothing(reward) || (return reward)
    action = rand(GI.available_actions(state))
    GI.play!(state, action)
   end
end

function MCTS.evaluate(::RolloutEvaluator, board, available_actions)
  V = rollout(board)
  n = length(available_actions)
  P = [1 / n for a in available_actions]
  return P, V
end

################################################################################

const GobbletMCTS = MCTS.Env{State, Board, Action}

struct MonteCarloAI <: AI
  env :: GobbletMCTS
  timeout :: Int
end

import Gobblet.TicTacToe: play

function play(ai::MonteCarloAI, state)
  MCTS.explore!(ai.env, state, ai.timeout)
  actions, distr = MCTS.policy(ai.env)
  actions[argmax(distr)]
end

################################################################################

function debug_tree(env; k=10)
  pairs = collect(env.tree)
  k = min(k, length(pairs))
  most_visited = sort(pairs, by=(x->x.second.Ntot), rev=true)[1:k]
  for (b, info) in most_visited
    println("N: ", info.Ntot)
    print_board(State(b))
  end
end

################################################################################

# In our experiments, we can simulate ~10000 games per second

using Profile
using ProfileView

if PROFILE
  env = GobbletMCTS(RolloutEvaluator())
  MCTS.explore!(env, 0.1)
  Profile.clear()
  @profile MCTS.explore!(env, 2.0)
  ProfileView.svgwrite("profile_mcts.svg")
  # To examine code:
  # code_warntype(MCTS.select!, Tuple{GobbletMCTS})
end

env = GobbletMCTS(RolloutEvaluator())
state = State()
MCTS.explore!(env, state, INITIAL_TRAINING)

if INTERACTIVE
  interactive!(state, red=MonteCarloAI(env, TIMEOUT), blue=Human())
end

################################################################################