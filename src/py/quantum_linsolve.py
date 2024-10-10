import scipy.sparse as spsp
import numpy as np
import json
import os
import pickle
import dill


def json_to_coo(json_data: str) -> spsp.coo_array:
    """Create a coo matrix out of the json data

    Args:
        json_data (str): data extracted from the json  file

    Returns:
        spsp.coo_array: final matrix
    """

    Aii = np.array(json_data["A"]["Aii"])
    Aij = np.array(json_data["A"]["Aij"])
    XLNZ = (
        np.array(json_data["A"]["XLNZ"]) - 1
    )  # (start position of each column in NZSUB)
    LNZ = (
        np.array(json_data["A"]["LNZ"]) - 1
    )  # (position of each NZSUB entry in Aij array)
    NZSUB = (
        np.array(json_data["A"]["NZSUB"]) - 1
    )  # (row index of each non-zero in each column)
    LNZ[LNZ < 0] = 0
    NZSUB[NZSUB < 0] = 0
    size = len(Aii)
    row, col, data = [], [], []

    # diagonal
    for i in range(size):
        row.append(i)
        col.append(i)
        data.append(Aii[i])

    # off diag
    for i in range(size):
        istart_col = XLNZ[i]
        iend_col = XLNZ[i + 1] if i < size - 1 else XLNZ[i] + 1
        row_idx = NZSUB[istart_col:iend_col]
        data_idx = LNZ[istart_col:iend_col]
        col_idx = [i] * len(row_idx)
        row += list(row_idx)
        col += list(col_idx)
        data += list(Aij[data_idx])

    return spsp.coo_matrix((data, (row, col)), shape=(size, size))


def process_matrix(A):
    """Process the matrix to create a symmetrize version

    Args:
        A (sp matrix): input matrix
    """

    # symmetrize the matrix
    A = A.todense()
    size = A.shape[0]

    # remove diag
    Adiag = np.diag(A)
    A = A - np.diag(Adiag)

    # set upper triangular part to 0
    A[np.triu_indices(size, k=1)] = 0

    # symmetrize
    A = A + A.T

    # add diagonal
    A = A + np.diag(Adiag)

    # make the matrix to csc format
    A = spsp.csc_array(A)

    return A


def load_json_data(file_name: str) -> (spsp.csr_array, np.ndarray, np.ndarray):  # type: ignore
    """Load a matrix from a json file corresponding to the linear system A x = b

    Args:
        file_name (str): file name

    Returns:
        tuple(spsp.csr_array, np.ndarray,np.ndarray): (A sparse CSR matrix, b array, x array)
    """

    # read the json data
    data = None
    with open(file_name) as json_file:
        json_str = json_file.read()
        data = json.loads(json_str)

    # create the matrix
    A = json_to_coo(data)

    # process matrix
    A = process_matrix(A)

    # create the rhs
    b = np.array(data["b"])

    return A, b


def save_serializable_result(result, sol_info):
    """Pickle intermediate linear solver results."""
    # check first if the object is serializable
    try:
        # attempt to pickle the object
        pickle.dumps(result)
    except (pickle.PicklingError, TypeError):
        print(f"Object {result.__class__.__name__} not serializable.")
        print(f"{sol_info} not created.")
        return

    try:
        existing_data = []
        if os.path.exists(sol_info):
            with open(sol_info, "rb") as fb:
                existing_data = pickle.load(fb)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]

        # append the new result and save it back to the file
        existing_data.append(result)
        with open(sol_info, "wb") as fb:
            pickle.dump(existing_data, fb)

    except Exception as err:
        print(f"An error occurred while saving intermediate linear solver results: {err}")


def main(debug=False, save_intermediate_linear_solver_results=True):
    # get the path o the shared folder
    epanet_tmp = os.environ["EPANET_TMP"]

    # file where the spare matrix was stored
    smat = os.path.join(epanet_tmp, "smat.json")

    # file where to store the solution
    sol = os.path.join(epanet_tmp, "sol.dat")
    sol_info = os.path.join(epanet_tmp, "sol_info.pckl")

    # file where the quantum solver is pickled
    solver_pckl = os.path.join(epanet_tmp, "solver.pckl")

    # load the data
    A, b = load_json_data(smat)

    # unpickle the solver
    with open(solver_pckl, "rb") as fb:
        solver = pickle.load(fb)

    # solve
    result = solver(A, b)

    if debug or save_intermediate_linear_solver_results:
        # save intermediate results
        save_serializable_result(result, sol_info)

    np.savetxt(sol, result.solution)
    if debug:
        print("A", A.todense())
        print("B", b)
        x = result.solution
        residue = np.linalg.norm(A @ x - b)
        print("X", x, residue)


if __name__ == "__main__":
    main(debug=False, save_intermediate_linear_solver_results=True)
