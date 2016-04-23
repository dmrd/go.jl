using BinDeps
using Conda

ENV["PYTHON"] = ""
push!(Conda.CHANNELS, "https://conda.anaconda.org/jaikumarm")

# Conda and BinDeps aren't playing nice, so just install manually for now...
Conda.add("keras")
Conda.add("h5py")

##
#@BinDeps.setup
#keras = library_dependency("keras")
#h5py = library_dependency("h5py")

#provides(Conda.Manager, "keras")
#provides(Conda.Manager, "h5py", h5py)

#@BinDeps.install
##

# Ensure that it is using the local Conda install
Pkg.build("PyCall")
