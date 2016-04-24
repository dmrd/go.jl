using PyCall
using HDF5

@pyimport keras.models as models
@pyimport keras.layers.core as core
@pyimport keras.layers.convolutional as kconv
@pyimport keras.callbacks as kcallbacks
@pyimport keras.utils.io_utils as k_io

"""
Temporary utility function until next release of PyCall (as of 1.4.0).
  Adds `revdims` flag
"""
function to_python{T<:PyCall.NPY_TYPES}(a::StridedArray{T}, revdims::Bool=false)
    @PyCall.npyinitialize
    size_a = revdims ? reverse(size(a)) : size(a)
    strides_a = revdims ? reverse(strides(a)) : strides(a)
    p = ccall(PyCall.npy_api[:PyArray_New], PyPtr,
              (PyPtr,Cint,Ptr{Int},Cint, Ptr{Int},Ptr{T}, Cint,Cint,PyPtr),
              PyCall.npy_api[:PyArray_Type],
              ndims(a), Int[size_a...], PyCall.npy_type(T),
              Int[strides_a...] * sizeof(eltype(a)), a, sizeof(eltype(a)),
              PyCall.NPY_ARRAY_ALIGNED | PyCall.NPY_ARRAY_WRITEABLE,
              C_NULL)
    return PyObject(p, a)
end

"""
Convert array to row major, C style arrays.
Does not modify memory during conversion
"""
function to_python_array(arr::AbstractArray)
    to_python_array(arr, size(arr)...)
end

"""
Convert array to row major, C style arrays.
Does not modify buffer during conversion

Specifically for bit arrays -> UInt8

Specify dimensions IN JULIA COORDINATES
Given Julia array with dims: (1, 2, 3)
Return Python array with dims: (3, 2, 1)
"""
function to_python_array(arr::AbstractArray, dims...)
    reshaped = reshape(arr, dims)
    rounded = round(UInt8, reshaped)
    to_python(rounded, true)
end

"Simple softmax classifier"
function CLF_LINEAR(features::Vector{Function})
    input_shape = get_input_size(features)
    KerasNetwork(models.Sequential([
                                    core.Flatten(input_shape=input_shape),
                                    core.Dense(N*N, input_dim=(N*N)),
                                    core.Activation("softmax")
                                    ]),
                 features)
end

"""
    Network from `Teaching Deep Convolutional Neural Networks to Play Go`
    Designed for 19x19, but should be fine for other sizes as well
    """
function CLF_DCNN(features::Vector{Function})
    # Reverse because julia <--> python
    input_shape = reverse(get_input_size(features))
    
    KerasNetwork(models.Sequential([
                                    kconv.Convolution2D(64, 7, 7, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(64, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(64, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(48, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(48, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(32, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(32, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    core.Flatten(),
                                    core.Dense(N*N, activation="softmax")
                                    ]),
                 features)
end

function compile(network::KerasNetwork)
    println(STDERR, "Compiling model...")
    network.model.compile(loss="categorical_crossentropy",
                          optimizer="adadelta",
                          metrics=["accuracy"])
end

function _callbacks(hf5path=nothing)
    callbacks = Vector()
    if hf5path != nothing
        checkpointer = kcallbacks.ModelCheckpoint(filepath=hf5path, verbose=1, save_best_only=true)
        push!(callbacks, checkpointer)
    end
    push!(callbacks, kcallbacks.ProgbarLogger())
    callbacks
end

# TODO: More advanced buffering and shuffling across multiple files
# Doesn't have a fixed batch size right now
@pydef type SGFGenerator
    __init__(self, filenames, features) = begin
        self[:filenames] = Vector{AbstractString}(filenames)
        self[:features] = features
        self[:buffer] = Vector()
        self[:fi] = 1
        self[:bi] = 1
        self[:tm] = 0.0
    end
    __next__(self) = self[:next]()
    next(self) = begin
        features = Vector{Function}(self[:features])
        # Load in the next file and return as one batch
        examples = Vector()
        while length(examples) == 0
            filename = AbstractString(self[:filenames][self[:fi]])
            examples = generate_training_data(filename, features=features)
            # Loop back to the beginning if we run out of files
            self[:fi] = (self[:fi] % length(self[:filenames])) + 1
        end
        X = go.batch_training_examples([x[1] for x in examples])
        Y = go.batch_training_examples([x[2][:] for x in examples])
        return (to_python_array(X),
                to_python_array(Y))
    end
end

# TODO: Add an in memory buffer to allow better shuffling
@pydef type HF5Generator
    __init__(self, hf5path, features, batch_size, start, stop) = begin
        h5 = h5open(AbstractString(hf5path))
        self[:data] = h5["data"]
        self[:label] = h5["label"]
        self[:features] = features
        self[:batch_size] = batch_size
        self[:start] = start
        self[:stop] = stop
        self[:i] = 0
    end
    __next__(self) = self[:next]()
    next(self) = begin
        features = Vector{Function}(self[:features])

        batch_size = self[:batch_size]
        start = self[:start]
        stop = self[:stop]
        i = self[:i]
        if start + i * batch_size > stop - batch_size
            i = 0
        end
        self[:i] = i + 1

        bstart = start + i * batch_size + 1
        bstop = bstart + batch_size - 1

        X = self[:data][:,:,:, bstart:bstop]
        Y = self[:label][:, bstart:bstop]

        return (to_python_array(X),
                to_python_array(Y))
    end
end


"""
Train a network that loads data from 
"""
function keras_train_generator(network::KerasNetwork, generator; epochs=20, samples_per_epoch=1000000, recompile=true, validation_data=nothing, checkpoint_path=nothing)
    if recompile
        compile(network)
    end

    println(STDERR, "Fitting model...")
    network.model.fit_generator(generator, samples_per_epoch=samples_per_epoch, nb_epoch=epochs,
                                verbose=2,
                                validation_data=validation_data,
                                callbacks=_callbacks(checkpoint_path))
end

" Train from an HDF5 file "
function keras_train_h5(network::KerasNetwork, hdf5_path::AbstractString; epochs=20, batch_size=32, recompile=true, n_validation=10000, n_train=nothing, checkpoint_path=nothing)
    if recompile
        compile(network)
    end

    #Open the file and see how many examples there are
    h5 = h5open(hdf5_path)
    X = h5["data"]
    Y = h5["label"]
    n_examples = size(X)[end]
    close(h5)

    max_n_train = n_examples - n_validation
    n_train = min(n_train, max_n_train)

    # These act like matrices
    trainX = k_io.HDF5Matrix(hdf5_path, "data", 0, n_train)
    trainY = k_io.HDF5Matrix(hdf5_path, "label", 0, n_train)

    valX = k_io.HDF5Matrix(hdf5_path, "data", n_train, n_train + n_validation)
    valY = k_io.HDF5Matrix(hdf5_path, "label", n_train, n_train + n_validation)

    network.model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batch_size, verbose=2,
                      validation_data=(valX, valY),
                      shuffle="batch", # Don't want to shuffle data too large for memory
                      callbacks=_callbacks(checkpoint_path))
end

function train_model(network::KerasNetwork, X, Y; epochs=20, batch_size=32, recompile=true, validation_split=0.25, checkpoint_path=nothing)
    if recompile
        compile(network)
    end

    # Handle the julia <--> python array conversion
    println(STDERR, "Doing python <--> Julia conversion...")
    X = to_python_array(X)
    Y = to_python_array(Y)
    println(STDERR, "Fitting model...")
    network.model.fit(X, Y, nb_epoch=epochs, batch_size=batch_size, verbose=2,
                      validation_split=validation_split,
                      callbacks=_callbacks(checkpoint_path))
end

function save_model(network::KerasNetwork, folder::AbstractString, name::AbstractString; save_yaml=true, save_weights=true, overwrite=false)
    if save_weights
        hf5path = joinpath(folder, string(name, ".hf5"))
        isfile(hf5path) && !overwrite && (println(STDERR, "File exists: $(hf5path)"); return)
        network.model.save_weights(hf5path)
    end

    if save_yaml
        ymlpath = joinpath(folder, string(name, ".yml"))
        isfile(ymlpath) && !overwrite && (println(STDERR, "File exists: $(ymlpath)"); return)
        yaml = network.model.to_yaml()

        # Add feature names
        # e.g. -> player_colors, liberties, ...
        # the split is to remove "go." - caused problems when reading in
        features = join(map(x -> split(string(x), ".")[end], network.features), ", ")
        yaml = string(yaml, "\nfeatures: [$(features)]")

        open(joinpath(folder, string(name, ".yml")), "w") do file
            write(file, yaml)
        end
    end
end
