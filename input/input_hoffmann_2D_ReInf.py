import json
import os
from argparse import ArgumentParser

def run_instance(counter, dim, l, r):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/input_hoffmann_2D_ReInf.json", 'r') as f:
       datastore = json.load(f)

    datastore["dim"]            = dim
    datastore["n global refinements"]            = l
    datastore["simulation reset manifold level"] = r
    datastore["paraview prefix"]                 = "results_hoffmann_2D_ReInf.%s" % (str(counter).zfill(4))

    with open("./roughness_%d/input_%s.json" % (dim, str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")

    parser.add_argument('dim', type=int)
    
    arguments = parser.parse_args()
    return arguments

def main():
    options = parseArguments()

    dim = options.dim

    if not os.path.exists("./roughness_%d" % dim):
        os.makedirs("./roughness_%d" % dim)

    max_level = 0

    if dim == 2:
        max_level = 4
    else:
        max_level = 3

    counter   = 0

    for l in reversed(range(0, max_level + 1)):

        for r in range(0, l + 1):
            run_instance(counter, dim, l, r)
            counter += 1

        run_instance(counter, dim, l, -1)
        counter += 1

if __name__== "__main__":
  main()
