using Mocha


function train_model(hdf5filename::AbstractString)
    data_layer = AsyncHDF5DataLayer(name="train-data", source=hdf5filename, batch_size=1, shuffle=true)
    fc_layer = InnerProductLayer(name="ip", output_dim=N*N, bottoms=[:data], tops=[:ip])
    softmax_layer = SoftmaxLossLayer(name="loss", bottoms=[:ip, :label])

    backend = DefaultBackend()
    init(backend)

    net = Net("policy-train", backend, [data_layer, fc_layer, softmax_layer])
    exp_dir = "snapshots-$(Mocha.default_backend_type)"
    method = SGD()

    params = make_solver_parameters(method, max_iter=10000, regu_coef=0.0005,
                                mom_policy=MomPolicy.Fixed(0.9),
                                lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
                                load_from=exp_dir)
    solver = Solver(method, params)

    setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)

    # report training progress every 100 iterations
    add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

    # save snapshots every 5000 iterations
    #add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

    ## show performance on test data every 1000 iterations
    #data_layer_test = HDF5DataLayer(name="test-data", source="data/test.txt", batch_size=100)
    #acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:ip2, :label])
    #test_net = Net("MNIST-test", backend, [data_layer_test, common_layers..., acc_layer])
    #add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

    solve(solver, net)

    #Profile.init(int(1e8), 0.001)
    #@profile solve(solver, net)
    #open("profile.txt", "w") do out
    #  Profile.print(out)
    #end

    destroy(net)
    destroy(test_net)
    shutdown(backend)
end

function policy()
end
