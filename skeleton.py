def label_sentence(sentence):
    """
    This is the heart of your algorithm.
    It recieves a sentence and attempts to label it as English or non English.
    Feel free to add more parameters to this function as needed.
    """
    return True

def label_all(in_file, out_file, nlines):
    """
    in_file is the path to the data.
    out_file is the path of the output file.
    nlines is how many lines (at most) to read.
    Feel free to add more parameters to this function as needed
    """

    f = open(in_file, "rb")
    o = open(out_file, "wb")
    line = f.readline()
    counter = 0
    while line and counter < nlines:
        line = line[:-1] # Get rid of that "\n"
        res = label_sentence(line) # res should be 0/False for not English or 1/True for English
        o.write(line + ", " + str(res) + "\n")
        line = f.readline()
        counter += 1


