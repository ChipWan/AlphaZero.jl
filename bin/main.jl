#####
##### Main script for the JuliaCon 2020 demo
#####

@everywhere ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # "binned" / "split"

# Enables running the script on a distant machine without an X server
@everywhere ENV["GKSwstype"]="nul"

using Distributed
try
  @info "nthreads() = $(Threads.nthreads())"
  @info "worker 1: nthreads() = $(fetch(@spawnat workers()[1] Threads.nthreads()))"
catch ex
  @info "nthreads failed " ex
end

@info "using AlphaZero"
@everywhere using AlphaZero
@info "used"

@everywhere const DUMMY_RUN = false

@everywhere import Pkg
@everywhere include(normpath(joinpath(dirname(Pkg.pathof(AlphaZero)), "../scripts/lib/dummy_run.jl")))
@everywhere include(normpath(joinpath(dirname(Pkg.pathof(AlphaZero)), "../games/connect-four/main.jl")))

@everywhere using .ConnectFour: Game, Training

params, benchmark = Training.params, Training.benchmark
if DUMMY_RUN
  params, benchmark = dummy_run_params(params, benchmark)
end

@info "creating session"
session = Session(
  Game,
  Training.Network{Game},
  params,
  Training.netparams,
  benchmark=benchmark,
  dir="sessions/connect-four",
  autosave=true,
  save_intermediate=false)

try
  @info String(read(`df`))
catch ex
  @info "df failed " ex
end

@info "training!"
resume!(session)
@info "success!"

try
  @info String(read(`df`))
catch ex
  @info "df failed " ex
end


# -- output
@info "tar up"
@info String(read(`tar czvf sessions.tar.gz sessions`))
try
  @info String(read(`ls -lh sessions.tar.gz`))
catch ex
  @info "ls failed" ex
end
try
  @info String(read(`du -sch sessions`))
catch ex
  @info "du failed" ex
end


ENV["RESULTS_FILE_TO_UPLOAD"] = "sessions.tar.gz"
