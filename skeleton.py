import argparse
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from string import ascii_lowercase

VECTOR_COLS = ['len'] + [c for c in ascii_lowercase] + ['doubles',
                                                        'average_word_len', 'longest_word_len', 'shortest_word_len']

# must conform to the format of the VECTOR_COLS variable


def create_vec(sentence: str) -> pd.DataFrame:
    letters_dict = {c: np.char.count(sentence, c) for c in ascii_lowercase}
    doubles = sum(np.char.count(sentence, c + c) for c in ascii_lowercase)
    words = [word for word in sentence.split(' ') if word != '']
    word_lengths = list(map(len, words))
    average_word_length = np.average(word_lengths)
    longest_word_length = max(word_lengths)
    shortest_word_length = min(word_lengths)

    return [len(sentence)] + [letters_dict[c] for c in ascii_lowercase] + [doubles, average_word_length,  longest_word_length, shortest_word_length]


def create_params(fname, skip_amnt, train_amnt):
    cols = ['sentence', 'is_english']
    df = pd.read_csv(fname, header=None, skiprows=skip_amnt, nrows=train_amnt,
                     names=cols, dtype=pd.StringDtype())
    params = df['sentence'].apply(create_vec)
    params = pd.DataFrame((item for item in params), columns=VECTOR_COLS)
    is_english = df['is_english']
    return params, is_english


def train(train_file: str, model_file: str, train_amnt: int, classifier):
    params, is_english = create_params(train_file, 0, train_amnt)

    model = make_pipeline(
        StandardScaler(), classifier)
    model.fit(params, is_english)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    return model


def load_model(model_file: str):
    with open(model_file, "rb") as f:
        return pickle.load(f)


def run_tests(model: SVC, test_file: str, test_offset: int, test_amnt: int):
    params, is_english = create_params(test_file, test_offset, test_amnt)
    score = model.score(params, is_english)
    print(f"model has score {score}")


def label_sentence(sentence):
    """
    This is the heart of your algorithm.
    It recieves a sentence and attempts to label it as English or non English.
    Feel free to add more parameters to this function as needed.
    """
    return True


def label_all(in_file, out_file, start_line, nlines):
    """
    in_file is the path to the data.
    out_file is the path of the output file.
    nlines is how many lines (at most) to read.
    Feel free to add more parameters to this function as needed
    """

    f = open(in_file, "rb")
    o = open(out_file, "wb")
    while start_line > 0:
        f.readline()
        start_line -= 1
    line = f.readline()
    counter = 0
    while line and counter < nlines:
        line = line[:-1]  # Get rid of that "\n"
        # res should be 0/False for not English or 1/True for English
        res = label_sentence(line)
        o.write(line + ", " + str(res) + "\n")
        line = f.readline()
        counter += 1


def get_args():
    parser = argparse.ArgumentParser(description='Send data to server.')
    parser.add_argument('--train', action='store_true',
                        help='should we re-train the model')
    parser.add_argument('test_amnt', type=int,
                        help='amount of lines to test with')
    parser.add_argument('model_file', type=str,
                        help='model file path for saving/loading')
    parser.add_argument('infile', type=str,
                        help='path to file to label')
    parser.add_argument('outfile', type=str,
                        help='path to file to write the labeled lines from infile to')
    parser.add_argument('train_file', type=str,
                        help='train file path', nargs="?", default="")
    parser.add_argument('train_amnt', type=int,
                        help='amount of lines to train with', nargs="?", default=0)

    return parser.parse_args()


def main(train_mode: bool, train_file: str, train_amnt: int, test_amnt: int, model_file: str, infile: str, outfile: str):
    if train_mode:
        if train_file == "":
            raise Exception("train file not provided")
        if train_amnt == 0:
            raise Exception("train amnt not provided")
        print("training model... ")
        model = train(train_file, model_file, train_amnt,
                      SVC(gamma=0.001, C=100., kernel='sigmoid'))
    else:
        print(f"loading model from file {model_file}")
        model = load_model(model_file)

    print("running some tests")
    run_tests(model, train_file, train_amnt, test_amnt)


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args.train, args.train_file, args.train_amnt,
         args.test_amnt, args.model_file, args.infile, args.outfile)
