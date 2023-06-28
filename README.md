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

xyzs, absolute, not in the base frame
tensor([[[[50.1505, 76.1135, 39.1979,  1.0000],
          [50.0010, 77.2540, 40.1370,  1.0000],
          [48.8798, 77.2580, 41.1115,  1.0000],
          [51.3621, 77.4165, 40.9482,  1.0000],

        print(xyzs)
        print(xyzs[...,:3])

RTframes = torch.stack((
            RTF0,
            RTF1,RTF2,RTF3,RTF4,RTF5,RTF6,RTF7, RTF8
        ),dim=1)

rf.py:761 in <module>
RTframes: tensor([[[[[-7.5478e-01, -3.1187e-01,  5.7710e-01,  5.0001e+01],
                     [ 2.6921e-03, -8.8122e-01, -4.7270e-01,  7.7254e+01],
                     [ 6.5598e-01, -3.5523e-01,  6.6596e-01,  4.0137e+01],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
          
                    [[ 8.5453e-01, -1.4068e-01,  4.9998e-01,  5.1362e+01],
                     [ 1.0205e-01, -8.9838e-01, -4.2720e-01,  7.7417e+01],
                     [ 5.0927e-01,  4.1608e-01, -7.5334e-01,  4.0948e+01],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
          
                    [[ 3.1795e-01,  3.4531e-01, -8.8299e-01,  5.1814e+01],
                     [-7.1858e-01,  6.9532e-01,  1.3169e-02,  7.6396e+01],
                     [ 6.1850e-01,  6.3031e-01,  4.6921e-01,  4.1826e+01],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
          
                    [[ 4.3798e-01, -2.3955e-01,  8.6648e-01,  5.2474e+01],
                     [ 3.8268e-01, -8.2247e-01, -4.2082e-01,  7.6973e+01],
                     [ 8.1346e-01,  5.1590e-01, -2.6855e-01,  4.3053e+01],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
          
                    [[-2.3124e-02,  7.9412e-01,  6.0731e-01,  5.2439e+01],
                     [-5.7122e-01, -5.0904e-01,  6.4388e-01,  7.6105e+01],
                     [ 8.2047e-01, -3.3202e-01,  4.6539e-01,  4.4301e+01],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
          
                    [[-4.6933e-01,  1.8726e-01,  8.6294e-01,  5.1793e+01],
                     [ 8.3945e-01, -2.0859e-01,  5.0182e-01,  7.8192e+01],
                     [ 2.7397e-01,  9.5991e-01, -5.9296e-02,  4.3451e+01],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
          
                    [[ 1.5595e-02,  4.9670e-01,  8.6778e-01,  5.1815e+01],
                     [ 9.0845e-02, -8.6500e-01,  4.9348e-01,  7.8320e+01],
                     [ 9.9574e-01,  7.1139e-02, -5.8611e-02,  4.4858e+01],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
          
                    [[ 4.7801e-01, -1.7655e-01,  8.6042e-01,  5.2546e+01],
                     [-7.9664e-01, -4.9972e-01,  3.4004e-01,  7.7102e+01],
                     [ 3.6994e-01, -8.4800e-01, -3.7952e-01,  4.5423e+01],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
          
                    [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]]],

        - base_indices = torch.full((5,34),0, dtype=torch.long) # base frame that builds each atom
        - xyzs_in_base_frame = torch.ones((5,34,4)) # coords of each atom in the base frame
        - RTs_in_base_frame = torch.eye(4).repeat(5,10,1,1) # torsion frames ( shape is 5,10,4,4 )
        - seq_types: ['pur', 'pyr']
        - seq: sequence as a list, e.g. ['G', 'C']
        - seq_index: e.g., tensor([0, 1]) where A = 0, C = 1, G = 2, U = 3, X = 4


```
