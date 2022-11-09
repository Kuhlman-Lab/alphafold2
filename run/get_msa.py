import os, argparse
from features import getRawInputs


def main(args):
    # Form query object.
    query = ('_INPUT_', [args.sequence])

    # Get MSAs
    raw_inputs = getRawInputs(
        queries=[query],
        msa_mode=args.msa_mode,
        output_dir=args.output_dir,
        design_run=True)
    
    # Write MSA file
    with open(os.path.join(args.output_dir, f'{args.out_file}.a3m'), 'w') as f:
        f.writelines(raw_inputs[args.sequence][0])


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sequence', type=str, help='Query sequence used for MSA generation. Should be a single chain.')
    parser.add_argument('--msa_mode', type=str, default='MMseqs2-U+E', choices=['MMseqs2-U+E', 'MMseqs2-U'], help='Mode '
                        'to generate MSAs. U = Uniref and E = Environmental, so MMseqs2-U+E queries the Uniref and '
                        'Environmental databases with mmseqs2 to generate the MSA.')
    parser.add_argument('--output_dir', type=str, default='./', help='Where to write the output .a3m MSA files.')
    parser.add_argument('--out_file', type=str, default='mmseqs2_msa', help='What the output MSA file should be named.')
    
    args = parser.parse_args()
    
    # Get MSAs
    main(args)