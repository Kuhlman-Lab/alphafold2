import os
from argparse import ArgumentParser
from typing import Sequence

def check_argument_validity(args: argparse.Namespace) -> Sequence[str]:

    # Validate numeric arguments.
    if args.num_gaps < 0:
        raise ValueError(f'Negative values for num_gaps are not allowed. {args.num_gaps} < 0')

    if args.linker_length < 0:
        raise ValueError(f'Negative values for linker_length are not allowed. {args.linker_length} < 0')

    if args.chain_length < 0:
        raise ValueError(f'Negative values for chain_length are not allowed. {args.chain_length} < 0')

    # Validation directory argument.
    dir_files = os.listdir(args.msa_dir)
    a3m_files = []
    for filename in dir_files:
        if filename.endswith('a3m'):
            a3m_files.append(filename)
    if len(a3m_files) == 0:
        raise ValueError(f'No .a3m files (MSA files) detected in {args.msa_dir}.')

    return a3m_files


if __name__ == '__main__':

    # Set up parser
    parser = ArgumentParser()
    parser.add_argument('--prepend',
                        action='store_true',
                        help='Prepends the desired gaps to the MSA. Default '
                        'is to append the gaps to the MSA.')
    parser.add_argument('--num_gaps',
                        default=0,
                        type=int,
                        help='Number of gaps to insert. Will be combined with '
                        'other arguments if provided.')
    parser.add_argument('--linker_length',
                        default=0,
                        type=int,
                        help='Number of residues to insert for a linker. Will '
                        'be combined with other arguments if provided.')
    parser.add_argument('--chain_length',
                        default=0,
                        type=int,
                        help='Number of residues to insert for additional chains. '
                        'Will be combined with other arguments if provided.')
    parser.add_argument('--msa_dir',
                        type=str,
                        help='Path to directory containing .a3m files (MSA files) '
                        'that will be edited.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='If included, original MSA files will be overwritten.')

    # Get arguments and validate them
    args = parser.parse_args()
    a3m_files = check_argument_validity(args)

    # Determine number of gaps to add
    total_gaps = args.num_gaps + args.linker_length + args.chain_length
    gaps = '-' * total_gaps

    # Loop over all the a3m files and add gaps to appropriately
    for a3m_file in a3m_files:
        # Read MSA lines
        with open(os.path.join(args.msa_dir, a3m_file), 'r') as f:
            a3m_lines = f.readlines()

        # Loop over lines
        i = 0
        new_lines = []
        update = False
        while i < len(a3m_lines):
            # If we're at a sequence line, add the gaps
            if update:
                # Add gap to front if prepending
                if args.prepend:
                    new_line = gaps + a3m_lines[i]
                # Add gap to back if appending
                else:
                    new_line = a3m_lines[i] + gaps
                new_lines.append(new_line)
                update = False
            # If we're at a non-sequence line, check if we're at a header
            else:
                # If at a header, mark the next line as a sequence line
                if a3m_lines[i][0] == '>':
                    update = True
                new_lines.append(a3m_lines[i])
            i += 1

        # Write the gapped MSA file, overwritting if necessary.
        if args.overwrite:
            with open(os.path.join(args.msa_dir, a3m_file), 'w') as f:
                f.write('\n'.join(new_lines))
        else:
            with open(os.path.join(args.msa_dir, a3m_file[:-4]+'_gapped.a3m'), 'w') as f:
                f.write('\n'.join(new_lines))
