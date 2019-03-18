#! /usr/bin/env python3
import argparse
import logging
import os

import stanfordnlp


# Constants
PROCESSORS = "tokenize,mwt,pos,lemma,depparse"  # All the current processors

# Setup logging
logging.basicConfig(
    filename='parser.log',
    filemode='a',
    format='[%(asctime)s] - %(levelname)s - %(message)s'
)


# Main functions
def prepare_library(lang="sl", processors="tokenize,mwt,pos,lemma,depparse",
                    pos_batch_size=1000):
    """A wrapper around the stanfordnlp.Pipeline() method"""
    nlp = stanfordnlp.Pipeline(lang=lang, processors=processors,
                               pos_batch_size=pos_batch_size)
    return nlp


def init_argparser():
    """Initializes argaprser."""
    parser = argparse.ArgumentParser(
        description="Processes txt files with the stanfordNLP neural pipline."
    )

    parser.add_argument(
        'source', type=str, help="The directory with the txt files."
    )
    parser.add_argument(
        'dest', type=str, help="The output directory."
    )
    parser.add_argument(
        '--lang', default="sl", help="The language to proces the data."
    )
    parser.add_argument('--processors', nargs='*', help="Processor, e.g pos.")
    parser.add_argument('--batch', type=int, help="Processing batch size.")
    args = parser.parse_args()
    return args


def get_original_sentence(sentence):
    """
    Attributes:
    ----------
    sentence: stanfordnlp.Sentence
        A procesed stanfordnlp Sentence object

    return
    ------
    str
        The orignal sentence.
    """

    original_sentence = ""
    words = sentence.words
    for index, word in enumerate(words):
        if index < len(words) - 1:
            if words[index + 1].upos == "PUNCT":
                original_sentence += word.text
            else:
                original_sentence += word.text + " "
        else:
            # The last "word"
            original_sentence += word.text
    return original_sentence


def tag_files(source, destination, nlp):
    """Processes txt files and writes them to a coresponding
    cupt file.

    NOTE: assumes cupt file format.

    atributes
    ---------
    source: str
        The path to the txt files.
        
    destination: str
        The path to write files.
    nlp: stanfordnlp.Pipeline
        A configured and loaded stanfordnlp.Pipeline object
    """
    files = [
        os.path.join(source, filename) for filename in os.listdir(path=source)
    ]
    for path_to_file in files:
        # Check if file exists
        filename, _ = os.path.splitext(os.path.split(path_to_file)[1])
        out_file = os.path.join(destination, filename + ".cupt")
        if os.path.exists(out_file):
            logging.info(f'File {filename} already exists.')
            continue

        # Parse file
        try:
            file_content = open(path_to_file).read()
            doc = nlp(file_content)
            cupt_content = [
                "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE",
            ]
            for sentence in doc.sentences:
                # Get the metadata for the cupt format
                cupt_content.append(f'# source_sent_id = . . {path_to_file}')
                original_sentence = get_original_sentence(sentence)
                cupt_content.append(f'# text = {original_sentence}')

                # Add the taged tokens
                word_props = []
                for word in sentence.words:
                    #  Example: <Word index=12;text=.;lemma=.;upos=PUNCT;xpos=Z;feats=_;governor=3;dependency_relation=punct>
                    word_props.append(word.index)
                    word_props.append(word.text)
                    word_props.append(word.lemma if word.lemma else "")
                    word_props.append(word.upos if word.upos else "")
                    word_props.append(word.xpos if word.xpos else "")
                    word_props.append(word.feats if word.feats else "")
                    word_props.append(word.governor if word.governor else "")
                    word_props.append(word.dependency_relation if
                                      word.dependency_relation else "")
                    word_props.append("_") # DEPS
                    word_props.append("_") # MISC
                    word_props.append("_") # PARSEME:MWE since this data is "blind"
                    word_props = [str(ent) for ent in word_props]
                    cupt_content.append("\t".join(word_props))
                    word_props = []
                    cupt_content.append("")

            # Write to file
            with open(out_file, "w") as cupt:
                for line in cupt_content:
                    cupt.write(line + "\n")
            logging.info(f'Processed {filename}, content writen to {out_file}.')
        except Exception as e:
            logging.error(f'Error {e} occured while processing {filename}.')



if __name__ == "__main__":
    args = init_argparser()
    # Get processors and check that the tokenize is loaded
    if args.processors and not "tokenize" in args.processors:
        args.processors.append("tokenize")
    processors_args = ",".join(args.processors) if args.processors else PROCESSORS
    pos_batch_size_arg = args.batch if args.batch else 1000
    nlp = prepare_library(lang=args.lang, processors=processors_args,
                          pos_batch_size=pos_batch_size_arg)
    tag_files(args.source, args.dest, nlp)

