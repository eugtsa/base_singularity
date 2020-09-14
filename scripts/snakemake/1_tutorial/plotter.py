import sys
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":

    input_filename,output_filename = sys.argv[1:]

    data = pd.read_csv(input_filename)

    plt.scatter(data.x, data.y, alpha=0.5)
    plt.title('Scatter plot {input_filename}'.format(input_filename=input_filename))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(output_filename)
