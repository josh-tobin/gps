from saved_examples.util import ResultLoader, ResultPlotter
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', nargs='+')
    args = parser.parse_args()
    
    loader = ResultLoader()
    plotter = ResultPlotter()
    results = loader.load(args.experiments)
    plotter.plot(results)

if __name__ == '__main__':
    main()
