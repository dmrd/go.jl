# Run this if you do not already have Keras setup on your machine
# Sets up PyCall to use a miniconda distribution
Pkg.add("Conda")
ENV["PYTHON"] = ""
Pkg.build("PyCall")

using Conda
push!(Conda.CHANNELS, "https://conda.anaconda.org/jaikumarm")
Conda.add("keras")
