#!/usr/bin/env python
import sys
import numpy as np
from argparse import ArgumentParser




def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-t", "--type", type=str, default="s", choices=["s", "u"], help="Type of training: s for supervised or u for unsupervised"
    )
    args = parser.parse_args()
    
    model_type = args.type

if __name__ == "__main__":
    main()