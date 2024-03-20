import numpy as np




def mk_training_data(m, n_training, seed, paulis="order 1", t_repeated=2):
    """
    Initialises all the training data as class parameters


    Parameters
    ----------
    n_training : int
        Number of training samples to use.
    seed : int
        Seed for the training samples, used for reproducible data.
    paulis : str, optional
        Determines the type of pauli strings that will be used as observables.
        Options: 'order k', 'full', 'random n'
        The default is 'order 1'.
    t_repeated : int, optional
        Number of repeated timesteps to use as data. The default is 2.

    Returns
    -------
    None.

    """

    self.set_steady_state()

    # rho_list = np.zeros((n_training,2**m,2**m),dtype = np.csingle)

    # dims = n, l, matrix
    training = np.zeros((t_repeated + 1, n_training, 2**m, 2**m), dtype=np.csingle)
    training_root = np.zeros(
        (t_repeated + 1, n_training, 2**m, 2**m), dtype=np.csingle
    )

    # dims = k, matrix
    paulis, pauli_names, pauli_id_list, pauli_indices = get_paulis(m, space=paulis)

    # dims = n, l, k (time, data, pauli)
    traces = np.zeros((t_repeated + 1, n_training, len(paulis)))
    measurements = np.zeros((t_repeated + 1, n_training, len(paulis)))

    for l in range(n_training):
        if l == 0:
            random_ket = qt.rand_ket_haar(dims=[[2**m], [1]], seed=seed)
            random_ket.dims = [[2] * m, [2] * m]
            random_bra = random_ket.dag()
            rho = (random_ket * random_bra).full()
            np.random.seed(seed)
        elif l == n_training - 1:
            rho = self.steady_state
        else:

            # =============================================================================
            #                 # Pure initialization
            #                 random_ket = qt.rand_ket_haar(dims = [[2**m], [1]], seed = seed)
            #                 random_ket.dims = [[2]*m,[2]*m]
            #                 random_bra = random_ket.dag()
            #                 rho = (random_ket * random_bra).full()
            # =============================================================================

            # Mixed initialization, randomly sets the eigenvalues s.t.
            # sum_i lambda_i = 1
            mix_factor = np.random.rand() ** 1 / 2

            evals = np.random.normal(size=2**m)
            evals = evals**2 / np.sum(evals**2)

            # print("Purity of initial state: {:.2f} with evals \n    {}".format(sum(evals**2), np.sort(evals)))

            # zero matrix
            zero_mat = np.zeros((2**m, 2**m))
            zero_mat[0, 0] = 1

            # mixed matrix
            init_matrix = mix_factor * zero_mat + (1 - mix_factor) * np.diag(evals)
            random_mixed = qt.Qobj(init_matrix, dims=[[2] * m, [2] * m])

            U = qt.random_objects.rand_unitary_haar(N=2**m, dims=[[2] * m, [2] * m])
            rho = (U * random_mixed * U.dag()).full()

        training[:, l, :, :] = np.reshape(
            self.evolution_n(t_repeated, rho), (t_repeated + 1, 2**m, 2**m)
        )

        for t_ind in range(t_repeated + 1):
            training_root[t_ind, l, :, :] = sc.linalg.sqrtm(
                training[t_ind, l, :, :]
            )
            for k, pauli in enumerate(paulis):
                traces[t_ind, l, k] = np.real(
                    np.trace(training[t_ind, l, :, :] @ pauli)
                )
                if self.error_type == "measurement":
                    prob = min(max((traces[t_ind, l, k] + 1) / 2, 0.0), 1.0)
                    measurements[t_ind, l, k] = (
                        np.random.binomial(self.n_measurements, prob)
                        / self.n_measurements
                        * 2
                        - 1
                    )

    

    # self.training_rho = rho_list
    self.training_data = training
    self.training_data_root = training_root

    self.traces = traces
    self.measurements = measurements

    self.paulis = paulis
    self.pauli_names = pauli_names
    self.pauli_id_list = pauli_id_list
    self.pauli_indices = pauli_indices
