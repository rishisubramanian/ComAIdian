"""
run_predictor.py

This script will read in from the database, the user frame, ratings frame, and
jokes frame. Using this information, it will run a matrix completion algorithm
on the data.

It will accept a number of jokes to predict, a user id to predict jokes
for, and database credentials as command line arguments.
This will write predicted joke ids line by line to the file 'jokes.txt'.

Usage:
    python run_predictor.py num_jokes uid database_name hostname port username password
"""
from __future__ import print_function

import sys

from predictor import Predictor
import user

def main(argv):
    if len(argv) != 8:
        print("Error: expected seven arguments", file=sys.stderr)
        print("Usage: python run_predictor.py num_jokes uid database_name"
              " hostname port username password",
              file=sys.stderr)
        sys.exit(1)

    num_jokes = int(argv[1])
    uid = int(argv[2])
    database_name = argv[3]
    hostname = argv[4]
    username = argv[5]
    password = argv[6]
    local = argv[7] # Should be passed in as either "true" or ""
    # Read in the data
    raters, ratings, jokes = user.read_clean_data(database_name, username,
                                                  hostname, password, local)
    # Construct a predictor object
    joke_predictor = Predictor(raters, ratings, jokes, picker='KNN',
                               completer='KNN', sampler='best')
    joke_predictor.set_user(uid)
    # Run the matrix completion algorithm.
    joke_predictor.update()
    # Predict the jokes
    jids = joke_predictor.get_joke(num_jokes)

    return (' '.join(map(str, jids)))

if __name__ == '__main__':
    main(sys.argv)
