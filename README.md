# rf-structure-module
Repository for RNA Structure Module (isolated from RosettaFoldNA), authors @Sid01123 @mmagnus

This is an example:

    python rf.py ./examples/1xjr.pdb

Full help:

```
./rf.py -h
usage: rf.py [-h] [-v] [--torsion0] [--rtf0] [-o OUTPUT] [--stop-artf STOP_ARTF] [--chemicals CHEMICALS] file

positional arguments:
  file

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         be verbose
  --torsion0            be verbose
  --rtf0                be verbose
  -o OUTPUT, --output OUTPUT
                        output structure in the PDB format, by default: output.pdb
  --stop-artf STOP_ARTF
                        write a PDB file after rotation/translation frame calculation/transition, starts from 0, if 0, then you get backbone frame
                        positions only
  --chemicals CHEMICALS
                        elect chemicals.py file to be used, by default: chemicals.py
```

and I think you need biopython and PyTorch to run it
(I’m testing dependencies right now in different environment to make sure )

    pip3 install torch==2.0.0 biopython icecream
