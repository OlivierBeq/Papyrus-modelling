# -*- coding: utf-8 -*-

import argparse

from papyrus_scripts.download import download_papyrus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the Papyrus data.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', '--out_dir',
                        required=False,
                        default=None,
                        help=('Directory where Papyrus data will be stored\n'
                             '(default: pystow\'s home folder).'),
                        metavar='output_directory',
                        dest='outdir')
    parser.add_argument('-V', '--version',
                        default='latest',
                        required=False,
                        help=('Version of the Papyrus data to be downloaded (default latest).\n'),
                        metavar='version',
                        dest='version')
    parser.add_argument('-s', '--stereo',
                        choices=['without', 'with', 'both'],
                        default='without',
                        required=False,
                        help=('Type of data to be downloaded:\n'
                              '    - without: standardised data without stereochemistry,\n'
                              '    - with: non-standardised data with stereochemistry,\n'
                              '    - both: both standardised and non-standardised data.\n'),
                        metavar='stereochemistry',
                        dest='stereo')
    parser.add_argument('-S', '--structures',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Should structures be downloaded (SD file).',
                        dest='structs')
    parser.add_argument('-d', '--descriptors',
                        choices=['mold2', 'cddd', 'mordred', 'fingerprint', 'unirep', 'all', 'none'],
                        default='all',
                        nargs='+',
                        required=False,
                        help=('Type of descriptors to be downloaded:\n'
                              '    - mold2: 2D Mold2 molecular descriptors (777),\n'
                              '    - cddd: 2D continuous data-driven descriptors (512),\n'
                              '    - mordred: 2D or 3D mordred molecular descriptors (1613 or 1826),\n'
                              '    - fingerprint: 2D RDKit Morgan fingerprint with radius 3\n'
                              '                   and 2048 bits or extended 3-dimensional\n'
                              '                   fingerprints of level 5 with 2048 bits,\n'
                              '    - unirep: UniRep deep-learning protein sequence representation\n'
                              '              containing 64, 256 and 1900-bit average hidden states,\n'
                              '              final hidden states and final cell states (6660),\n'
                              '    - all: all descriptors for the selected stereochemistry,\n'
                              '    - none: do not download any descriptor.'),
                        metavar='descriptors',
                        dest='descs')
    args = parser.parse_args()
    download_papyrus(outdir=args.outdir,
                     version=args.version,
                     nostereo=args.stereo in ['without', 'both'],
                     stereo=args.stereo in ['with', 'both'],
                     structures=args.structs,
                     descriptors=args.descs,
                     progress=True)
