import perceptron as perceptr
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Create NN 1)Perceptron
    pct = perceptr.Perceptron(eta=0.001, n_iter=50)

    # Define data and classes and plot all
    X = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]])
#    X = np.array([[1, 2], [1, 3], [2, 3], [2, 1], [3, 1], [3, 2]])
    y = np.array([-1, -1, -1, 1, 1, 1])

    perceptr.plot_data(X, y)

    # Train 1)Perceptron
    pct.fit(X, y)

    # Print weights
    print("Final weights values: ", end="")
    print(pct.w_)

    # Do prediction
    prediction = pct.predict(X)

    # plot weights updates for each epoch
    plt.plot(range(1, len(pct.errors_) + 1), pct.errors_, marker='o')
    plt.title("PERCEPTRON weights updates")
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

    # plot decision regions
    perceptr.plot_decision_regions(X, y, classifier=pct)
    plt.xlabel('x1 [cm]')
    plt.ylabel('x2 [cm]')
    plt.legend(loc='upper left')
    plt.show()

