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

# varables

```python
NATOKENS = 5 in our case, it's a number of nucleic acid residues [4; A, G, C, U] + X [unknown base]

alphas: for each residue it's 10 element tensor of sin, cos that describes the torsion angles:
        tensor([[[[-0.4466, -0.8947], 
                  [-0.2530,  0.9675],
                  [-0.9160,  0.4012],
                  [-0.8217, -0.5700],
                  [-0.7891,  0.6143],
                  [ 0.0082,  1.0000],
                  [-0.6455,  0.7637],
                  [ 0.9774,  0.2112],
                  [ 0.8674, -0.4977],
                  [-0.9995, -0.0301]],

                 [[-0.6560, -0.7548],
                  [ 0.2032, -0.9791],
                 ...
 alphas: torch.zeros((1,L,10,2), device=seq.device) # the 2 is for cos and sign components
 torch.Size([*, N, 10, 2])

 xyzs_in_base_frame: # coords of each atom in the base frame
        XYZs for the whole resides

 self.xyzs_in_base_frame: Parameter containing:
                         tensor([[[-7.3190e-01,  1.2920e+00,  0.0000e+00,  1.0000e+00],
                                  [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00],
                                  [ 1.4855e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00],
                                  [-4.9480e-01, -8.5590e-01,  1.2489e+00,  1.0000e+00],
                                  [ 7.2890e-01,  1.2185e+00,  0.0000e+00,  1.0000e+00],
                                  [ 5.5410e-01,  1.4027e+00,  0.0000e+00,  1.0000e+00],
                                  [ 4.9140e-01, -6.3380e-01, -1.2098e+00,  1.0000e+00],


RTFX where X is from 0 to 8 in our case, this is a rotation+translation (4x4) tensor::
 tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
```
