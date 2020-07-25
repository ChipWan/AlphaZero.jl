#####
##### Testing distributed self-play
#####

using Distributed
using LinearAlgebra

#BLAS.set_num_threads(1)

#addprocs(2, exeflags="--project")

@everywhere using AlphaZero

include("../distributed/include_workaround.jl")
include_everywhere("../../games/tictactoe/main.jl")
using .Tictactoe: Game, Training

using ProgressMeter

params = Training.params
network = Training.Network{Game}(Training.netparams)
env = AlphaZero.Env{Game}(params, network)

progress = Progress(params.self_play.num_games)
struct Handler end
AlphaZero.Handlers.game_played(::Handler) = next!(progress)

println("Running $(Distributed.nworkers()) distributed worker(s).")
@everywhere println("Running on $(Threads.nthreads()) thread(s).")

report, t, mem, gct = @timed AlphaZero.self_play_step!(env, Handler())

println("Memory size: $(report.memory_size)")
println("Total time: $t")
println("Spent in GC: $gct")
