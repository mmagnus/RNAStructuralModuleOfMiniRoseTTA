# rf-structure-module
Repository for RNA Structure Module (isolated from RosettaFoldNA), authors @Sid01123 @mmagnus

This is an example:

    python rf.py ./examples/1xjr.pdb

Full help:

```
python rf.py  -h
  usage: rf.py [-h] [-v] [--output OUTPUT] file

  positional arguments:
    file

  optional arguments:
    -h, --help       show this help message and exit
    -v, --verbose    be verbose
    --output OUTPUT
```

and I think you need biopython and PyTorch to run it
(I’m testing dependencies right now in different environment to make sure )

    pip3 install torch==2.0.0 biopython icecream

