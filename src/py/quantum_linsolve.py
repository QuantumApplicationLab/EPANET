import scipy.sparse as spsp
import numpy as np
import json
import os
import pickle


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

    # create the rhs
    b = np.array(data["b"])

    return A, b


def main(debug=False):

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
    print(A.todense(), b)

    # unpickle the solver
    with open(solver_pckl, "rb") as fb:
        solver = pickle.load(fb)

    # solve
    result = solver(A.todense(), b)

    # pickle the solution
    if debug:
        try:
            with open(sol_info, "wb") as fb:
                pickle.dump(result, fb)
        except:  # noqa: E722
            print("Cannot pickle solution info")

    np.savetxt(sol, result.solution)


if __name__ == "__main__":
    main(debug=False)
