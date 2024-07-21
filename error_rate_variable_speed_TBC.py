import numpy as np
import matplotlib.pyplot as plt

def error_rate(Nx, error):
    """Compute the error rate."""

    # Compute the logs of the nodes and the errors
    logNx = np.log(np.array(Nx))
    logError = np.log(np.array(error))
    ones = np.ones(len(logNx))

    V = np.array([ones, logNx]).transpose()

    # Solve least squares system
    A = np.matmul(V.transpose(), V)
    b = np.matmul(V.transpose(), logError)

    c = np.linalg.solve(A, b)

    return c[1]


if __name__ == '__main__':

    #nodes = [32, 64, 128, 256, 512]
    #error_norms = [0.04439785618075195, 0.00603412251675461, 0.0007702538916404398, 9.67600265047024e-05, 1.2100024322051872e-05]

    nodes = [61, 121, 181, 241, 301]
    error_norms = [0.006970292693603508, 0.0009115248906955477, 0.0002734489187522113, 0.00011597791402578384, 5.952745583571874e-05]

    p = error_rate(nodes, error_norms)

    print('Error rate = {:6f} '.format(p))

    plt.loglog(nodes, error_norms, 'r-o')
    plt.xlabel('log(Nx)')
    plt.ylabel('log(error_norms)')
    plt.grid()
    plt.show()