import json
import os

def run_instance(counter, l, r):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/input_hoffmann_2D_ReInf.json", 'r') as f:
       datastore = json.load(f)

    datastore["n global refinements"]            = l
    datastore["simulation reset manifold level"] = r
    datastore["paraview prefix"]                 = "results_hoffmann_2D_ReInf.%s" % (str(counter).zfill(4))

    with open("./roughness/input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    if not os.path.exists("./roughness"):
        os.makedirs("./roughness")

    max_level = 4
    counter   = 0

    for l in reversed(range(0, max_level + 1)):

        for r in range(0, l + 1):
            run_instance(counter, l, r)
            counter += 1

        run_instance(counter, l, -1)
        counter += 1

if __name__== "__main__":
  main()
