#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
alphas: x len of sequence::
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

RTFX ::
 tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
"""

from icecream import ic
import sys
from Bio.PDB import PDBParser
import Bio.PDB 
import xpdb  # this is the module described below
import matplotlib.pyplot as plt
from statistics import mode, mean, multimode
import math
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
torch.set_printoptions(threshold=10_000)
import argparse
#from chemicals import *

def make_ideal_RTs():
    """<https://github.com/uw-ipd/RoseTTAFold2NA/blob/ca0283656f7c1205dc295a0abaee803f1457b038/network/util.py#L465>
    """
    torsion_indices = torch.full((NAATOKENS,NTOTALDOFS,4),0)

    for i in range(NAATOKENS):
        # NA BB tors
        torsion_indices[i,0,:] = torch.tensor([-5,-7,-8,1])  # epsilon_prev
        torsion_indices[i,1,:] = torch.tensor([-7,-8,1,3])   # zeta_prev
        torsion_indices[i,2,:] = torch.tensor([0,1,3,4])     # alpha (+2pi/3)
        torsion_indices[i,3,:] = torch.tensor([1,3,4,5])     # beta
        torsion_indices[i,4,:] = torch.tensor([3,4,5,7])     # gamma
        torsion_indices[i,5,:] = torch.tensor([4,5,7,8])     # delta

        # NA sugar ring tors
        torsion_indices[i,6,:] = torch.tensor([4,5,6,9])    # nu2
        torsion_indices[i,7,:] = torch.tensor([5,6,9,10])   # nu1
        torsion_indices[i,8,:] = torch.tensor([6,9,10,7])   # nu0

        # NA chi
        if torsions[i][0] is not None:
            i_l = aa2long[i]
            for k in range(4):
                a = torsions[i][0][k]
                torsion_indices[i,9,k] = i_l.index(a) # chi
            # no NA torsion flips

    # kinematic parameters
    # base frame that builds each atom
                
    # prepare tensor, put 0
    base_indices = torch.full((NAATOKENS, NTOTAL),0, dtype=torch.long)
    # coords of each atom in the base frame, prepare tensor, put 1
    xyzs_in_base_frame = torch.ones((NAATOKENS, NTOTAL,4))
    # torsion frames
    # RTs_in_base_frame
    RTs_by_torsion = torch.eye(4).repeat(NAATOKENS, NTOTALTORS,1,1)

    def make_frame(X, Y):
        """Process ideal frames
        Args:
            vectors X Y, e.g.::

                    tensor([-0.4948, -0.8559,  1.2489]), tensor([-0.7319,  1.2920,  0.0000])
          tensor, e.g.::

           tensor([[-0.3106, -0.6221, -0.7187],
                   [-0.5373,  0.7386, -0.4071],
                   [ 0.7841,  0.2597, -0.5637]])"""
        Xn = X / torch.linalg.norm(X)
        Y = Y - torch.dot(Y, Xn) * Xn
        Yn = Y / torch.linalg.norm(Y)
        Z = torch.cross(Xn,Yn)
        Zn =  Z / torch.linalg.norm(Z)
        return torch.stack((Xn,Yn,Zn), dim=-1)

    ## NUCLEIC ACIDS
    for i in range(NAATOKENS):
        i_l = aa2long[i]

        for name, base, coords in ideal_coords[i]:
            idx = i_l.index(name)
            base_indices[i,idx] = base
            xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

        # epsilon(p)/zeta(p) - like omega in protein, not used to build atoms
        #                    - keep as identity
        RTs_by_torsion[i,0,:3,:3] = torch.eye(3)
        #ic(RTs_by_torsion[i,0,:3,:3])

        RTs_by_torsion[i,0,:3,3] = torch.zeros(3)
        RTs_by_torsion[i,1,:3,:3] = torch.eye(3)
        RTs_by_torsion[i,1,:3,3] = torch.zeros(3)

        #ic(RTs_by_torsion)
        #ic(xyzs_in_base_frame[i,0,:3])
        # alpha
        #ic(xyzs_in_base_frame[i,3,:3] - xyzs_in_base_frame[i,1,:3])
        
        RTs_by_torsion[i,2,:3,:3] = make_frame(
            xyzs_in_base_frame[i,3,:3] - xyzs_in_base_frame[i,1,:3], # P->O5'
            xyzs_in_base_frame[i,0,:3] - xyzs_in_base_frame[i,1,:3]  # P<-OP1
        )
        #ic(RTs_by_torsion[i,2,:3,:3])

        RTs_by_torsion[i,2,:3,3] = xyzs_in_base_frame[i,3,:3] # O5'

        # beta
        RTs_by_torsion[i,3,:3,:3] = make_frame(
            xyzs_in_base_frame[i,4,:3] , torch.tensor([-1.,0.,0.])
        )
        RTs_by_torsion[i,3,:3,3] = xyzs_in_base_frame[i,4,:3] # C5'

        # gamma
        RTs_by_torsion[i,4,:3,:3] = make_frame(
            xyzs_in_base_frame[i,5,:3] , torch.tensor([-1.,0.,0.])
        )
        RTs_by_torsion[i,4,:3,3] = xyzs_in_base_frame[i,5,:3] # C4'

        # delta
        RTs_by_torsion[i,5,:3,:3] = make_frame(
            xyzs_in_base_frame[i,7,:3] , torch.tensor([-1.,0.,0.])
        )
        RTs_by_torsion[i,5,:3,3] = xyzs_in_base_frame[i,7,:3] # C3'

        # nu2
        RTs_by_torsion[i,6,:3,:3] = make_frame(
            xyzs_in_base_frame[i,6,:3] , torch.tensor([-1.,0.,0.])
        )
        RTs_by_torsion[i,6,:3,3] = xyzs_in_base_frame[i,6,:3] # O4'

        # nu1
        C1idx,C2idx = 9,10

        RTs_by_torsion[i,7,:3,:3] = make_frame(
            xyzs_in_base_frame[i,C1idx,:3] , torch.tensor([-1.,0.,0.])
        )
        RTs_by_torsion[i,7,:3,3] = xyzs_in_base_frame[i,C1idx,:3] # C1'

        # nu0
        RTs_by_torsion[i,8,:3,:3] = make_frame(
            xyzs_in_base_frame[i,C2idx,:3] , torch.tensor([-1.,0.,0.])
        )
        RTs_by_torsion[i,8,:3,3] = xyzs_in_base_frame[i,C2idx,:3] # C2'

        # NA chi
        if torsions[i][0] is not None:
            a2 = torsion_indices[i,9,2]
            RTs_by_torsion[i,9,:3,:3] = make_frame(
                xyzs_in_base_frame[i,a2,:3] , torch.tensor([-1.,0.,0.])
            )
            RTs_by_torsion[i,9,:3,3] = xyzs_in_base_frame[i,a2,:3] # N1/N9

    ic(base_indices)
    ic(RTs_by_torsion)
    ic(xyzs_in_base_frame)
    ic(torsion_indices)
    return base_indices, RTs_by_torsion, xyzs_in_base_frame, torsion_indices

def read_atoms(structure):
    """
    Args:
    - structure (BioPython structure): a structure

    Returns:
    - tensor: with coords::
 
         coords = tensor([[[[50.1500, 76.1130, 39.1980],
                       [50.0010, 77.2540, 40.1370],
                       [48.8780, 77.2580, 41.1130],
                       [51.3840, 77.5700, 40.8670],
                       [51.9790, 76.6200, 41.7380],
                       ....
                       [ 0.0000,  0.0000,  0.0000]],

                      [[54.6610, 73.2080, 44.3730],
                       [53.7120, 74.0580, 43.6070],
                       ....
                       [ 0.0000,  0.0000,  0.0000],
                       [ 0.0000,  0.0000,  0.0000]]]])
    """
    res =[r for r in structure.get_residues() if r.get_resname() in ["A","C","G","U"] and r.has_id('OP1') and r.has_id('OP2')]
    L = len(res)
    coords = torch.zeros((1,L,NTOTAL,3))    
    dict_res = {'A':0, 'C':1, 'G':2, 'U':3}

    for i, residue in enumerate(res):
        allatoms = aa2long[dict_res[residue.get_resname()]]
        for j, a in enumerate(allatoms):
            names = [z.get_fullname() for z in residue.get_atoms()]
            if a is not None:
                if a in names: # Heavy atoms, not including Hydrogens
                    ind = names.index(a)
                    coords[:,i,j,:] = torch.FloatTensor(list(residue.get_atoms())[ind].get_coord())
            else:
                coords[:,i,j,:] = torch.Tensor([torch.nan, torch.nan, torch.nan])
    return coords


def compute_backbone_frames(structure, frame: list):
    """
    Args:
    
    - structure (Biopython Structure): RNA structure read from PDB
    - frame: list of 6 string elements, first 3 atom names for purine frame, second 3 atom names for pyramidine frame, e.g. ["OP1","P", "OP2", "OP1", "P","OP2"]

    Return:
    
     - listRs (list): list of 3x3 Rotation matrices
     - listTs (list): list of Ts
     - seq_types (list): e.g. ['pur', 'pyr']
     - seq (list): ['G', 'C']
     - seq_index (tensor): seq: e.g., tensor([0, 1]) where A = 0, C = 1, G = 2, U = 3, X = 4
    """
    listRs, listTs = [], []
    seq_types, seq = [], []
    seq_index = []
    for residue in structure.get_residues():
        if residue.get_resname() in ["A","C","G","U"] and residue.has_id('OP1') and residue.has_id('OP2'):
            if residue.has_id('N9'): # PURINES
                x1 = residue[frame[0]]
                x2 = residue[frame[1]]
                x3 = residue[frame[2]]
                seq_types.append("pur")
                seq.append(residue.get_resname())

            else: # PYRAMIDINES
                x1 = residue[frame[3]]
                x2 = residue[frame[4]]
                x3 = residue[frame[5]]
                seq_types.append("pyr")
                seq.append(residue.get_resname())
 
            x1_arr = np.array(x1.get_coord())[:,None]
            x2_arr = np.array(x2.get_coord())[:,None]
            x3_arr = np.array(x3.get_coord())[:,None]

            seq_index.append(["A","C","G","U"].index(residue.get_resname()))

            R, t = rigidFrom3Points(x1_arr, x2_arr, x3_arr)
            listRs.append(R)
            listTs.append(t)

    seq_index = torch.tensor([seq_index])
    return listRs, listTs, seq_types, seq, seq_index


def rigidFrom3Points(x1, x2, x3):
    """
    Args:
    - x1, x2, x3 = arrays of coords, e.g.::
    
       x3: array([[48.878],
               [77.258],
               [41.113]], dtype=float32)

    Returns:
    - R, t: e.g.,::

            R: array([[-0.75477743, -0.31187248,  0.57710195],
                      [ 0.00269208, -0.88121927, -0.4727001 ],
                      [ 0.65597546, -0.35522974,  0.665964  ]], dtype=float32)
            t: array([[50.001],
                      [77.254],
                      [40.137]], dtype=float32)
    
    """
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / np.linalg.norm(v1)
    p = e1.T @ v2
    u2 = v2 - e1 @ p
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1.reshape((3,)), e2.reshape((3,))).reshape((3,1))
    R = np.hstack((e1, e2, e3))
    t = x2
    return R, t


def writepdb(filename: str, atoms: torch.tensor, seq, idx_pdb=None, bfacts=None):
    """
    Args:
    - filename (str): a filename
    - atoms: 
    """
    f = open(filename, "w")
    ctr = 1
    scpu = seq.cpu().squeeze(0)
    atomscpu = atoms.cpu().squeeze(0)

    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])
    bfacts = torch.clamp( bfacts.cpu(), 0, 100) # fix for #30

    for i, s in enumerate(scpu):
        natoms = atomscpu.shape[-2]
        if (natoms!=NHEAVY and natoms!=NTOTAL):
            print ('bad size!', natoms, NHEAVY, NTOTAL, atoms.shape)
            assert(False)

        atms = aa2long[s]

        # his prot hack
        if (s==8 and torch.linalg.norm( atomscpu[i,9,:]-atomscpu[i,5,:] ) < 1.7):
            atms = (
                " N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",
                  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1",
                " HD1",  None,  None,  None,  None,  None,  None) # his_d

        for j,atm_j in enumerate(atms):
            if (j<natoms and atm_j is not None and not torch.isnan(atomscpu[i,j,:]).any()):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, atm_j, num2aa[s], 
                    "A", idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                    1.0, bfacts[i] ) )
                ctr += 1


def make_rotX(angs, eps=1e-6):
    """Rotate about the x axis

    Not used here.
    """
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    RTs[:,:,1,1] = angs[:,:,0]/NORM
    RTs[:,:,1,2] = -angs[:,:,1]/NORM
    RTs[:,:,2,1] = angs[:,:,1]/NORM
    RTs[:,:,2,2] = angs[:,:,0]/NORM
    return RTs

def make_rotX_chi(angs, iden, eps=1e-6):
    """
    rotate about the x axis, for simplified version for single RNA looping over chain length #
    """
    angs2 = torch.squeeze(angs[:,iden,:])
    NORM = torch.linalg.norm(angs2, dim=-1) + eps
    RTs = torch.eye(4,  device=angs.device)

    RTs[1,1] = angs2[0]/NORM
    RTs[1,2] = -angs2[1]/NORM
    RTs[2,1] = angs2[1]/NORM
    RTs[2,2] = angs2[0]/NORM

    return RTs

# Calculating the torsion angles/creating the coordinate tensor
# L is number of residues in RNA, 34 is total max number of distinct RNA atoms
# First 3 is the number of atoms in a reference frame here OP1, P, OP2
# Second 3 is xyz coords
# L = 10
# xyz = torch.full((1,L,3,3),np.nan).float()

# ALGORITHM 24
# compute allatom structure from backbone frames and torsions
#
# alphas:
#    eps(p)/zeta(p): 0-1
#    alpha/beta/gamma/delta: 2-5
#    nu2/nu1/nu0: 6-8
#    chi_1(na): 9
# 
# RTs_in_base_frame:
#    eps(p)/zeta(p): 0-1
#    alpha/beta/gamma/delta: 2-5
#    nu2/nu1/nu0: 6-8
#    chi_1(na): 9
#
# RT frames (output):
#    origin: 0
#    alpha/beta/gamma/delta: 1-4
#    nu2/nu1/nu0: 5-7
#    chi_1(na): 8

class ComputeAllAtomCoords(torch.nn.Module):
    def __init__(self):
        """
        Attributes:
        
        - base_indices = torch.full((5,34),0, dtype=torch.long) # base frame that builds each atom
        - xyzs_in_base_frame = torch.ones((5,34,4)) # coords of each atom in the base frame
        - RTs_in_base_frame = torch.eye(4).repeat(5,10,1,1) # torsion frames ( shape is 5,10,4,4 )
        - seq_types: ['pur', 'pyr']
        - seq: sequence as a list, e.g. ['G', 'C']
        - seq_index: e.g., tensor([0, 1]) where A = 0, C = 1, G = 2, U = 3, X = 4
        """
        super(ComputeAllAtomCoords, self).__init__()
        self.base_indices = nn.Parameter(base_indices, requires_grad=False) 
        self.RTs_in_base_frame = nn.Parameter(RTs_by_torsion, requires_grad=False)
        self.xyzs_in_base_frame = nn.Parameter(xyzs_in_base_frame, requires_grad=False) 
        self.listRs = np.array(listRs)
        self.listTs = np.array(listTs)
        self.seq_types = seq_types
        self.seq_index = seq_index
        self.seq = seq


    def forward(self, alphas):  # self, seq, xyz, alphas
        """
        alphas: torch.zeros((1,L,10,2), device=seq.device) # the 2 is for cos and sign components
        xyz: B=1,L,34,3
        seq: range(5) A,C,G,U,RX     
        1, number of cycles, some remaining shape
        1, some remaining shape
        
        # input: xyz_t (B=1, T=maxtmpl, L, Natms, 3) B = 1, xyzt torch.full((n_templ,L,NTOTAL,3),np.nan).float(), then :maxtmpl and then unsqueeze 0
        # ouput: xyz (B, T, L, Natms, 3)
        # And then you do xyz_t[:,0] which shrinks size to B,L,Natms,3
        Finally xyz input into rigid_from_3_pts is B=1,L,3 for OP1, P, OP2 (L is number of residues)
        """
        # B,L = xyz.shape[:2] #?? batch size, len?

        # is_NA = is_nucleic(seq)
        # Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], is_NA) # Rs B,L,3,3 and Ts B,L,3
        Rs, Ts = torch.Tensor(self.listRs), torch.squeeze(torch.Tensor(self.listTs))
        
        L = len(self.listRs)
        #ic(L)

        # init RTF0 ones
        RTF0 = torch.eye(4).repeat(L,1,1).to(device=Rs.device)
        ic(RTF0)
        
        # bb
        RTF0[:,:3,:3] = Rs
        RTF0[:,:3,3] = Ts


        # rt self.xyzs_in_base_frame.shape torch.Size([32, 36, 4])
        # basexyzs.shape: torch.Size([5, 34, 4])
        basexyzs = self.xyzs_in_base_frame

        # ic(basexyzs, basexyzs.shape)

        # ignore RTs_in_base_frame[seq,0:2,:] and alphas[:,:,0:2,:] 
        # These are epsilon/zeta of previous residue, not used to build frames
        
        # USING BATCH MULTIPLICATION SYNTAX
        # # NA alpha O5' 
        # RTF1 = torch.einsum(
        #     'brij,brjk,brkl->bril', # (5,10,4,4)
        #     RTF0, self.RTs_in_base_frame[seq,2,:].repeat(B,L,1,1), make_rotX(alphas[:,:,2,:])) 

        # # NA beta 
        # RTF2 = torch.einsum(
        #     'brij,brjk,brkl->bril', 
        #     RTF1, self.RTs_in_base_frame[seq,3,:].repeat(B,L,1,1), make_rotX(alphas[:,:,3,:]))

        # # NA gamma
        # RTF3 = torch.einsum(
        #     'brij,brjk,brkl->bril', 
        #     RTF2, self.RTs_in_base_frame[seq,4,:].repeat(B,L,1,1), make_rotX(alphas[:,:,4,:]))

        # # NA delta
        # RTF4 = torch.einsum(
        #     'brij,brjk,brkl->bril', 
        #     RTF3, self.RTs_in_base_frame[seq,5,:].repeat(B,L,1,1), make_rotX(alphas[:,:,5,:]))

        # # NA nu2 - from gamma frame
        # RTF5 = torch.einsum(
        #     'brij,brjk,brkl->bril', 
        #     RTF3, self.RTs_in_base_frame[seq,6,:].repeat(B,L,1,1), make_rotX(alphas[:,:,6,:]))

        # # NA nu1
        # RTF6 = torch.einsum(
        #     'brij,brjk,brkl->bril', 
        #     RTF5, self.RTs_in_base_frame[seq,7,:].repeat(B,L,1,1), make_rotX(alphas[:,:,7,:]))

        # # NA nu0
        # RTF7 = torch.einsum(
        #     'brij,brjk,brkl->bril', 
        #     RTF6, self.RTs_in_base_frame[seq,8,:].repeat(B,L,1,1), make_rotX(alphas[:,:,8,:]))

        # # NA chi - from nu1 frame
        # RTF8 = torch.empty_like(RTF0)
        # for id, _id in enumerate(self.seq_id):
        #     if _id == "pur":
        #         RTF8[:,id,:,:]= torch.einsum(
        #             'bij,bjk,bkl->bil', 
        #             RTF6[:,id,:,:], self.RTs_in_base_frame[0,9,:].repeat(B,1,1), make_rotX_chi(alphas[:,:,9,:],id))
        #     else:
        #         RTF8[:,id,:,:]= torch.einsum(
        #             'bij,bjk,bkl->bil', 
        #             RTF6[:,id,:,:], self.RTs_in_base_frame[1,9,:].repeat(B,1,1), make_rotX_chi(alphas[:,:,9,:], id))

        # RTframes = torch.stack((
        #     RTF0,
        #     RTF1,RTF2,RTF3,RTF4,RTF5,RTF6,RTF7, RTF8
        # ),dim=2)


        # xyzs = torch.einsum(
        #     'brtij,brtj->brti', 
        #     RTframes.gather(2,self.base_indices[seq][...,None,None].repeat(1,1,1,4,4)), basexyzs.repeat(B,L,1,1)
        # )
             
        RTF1, RTF2, RTF3, RTF4, RTF5, RTF6, RTF7, RTF8 = torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0)
        # SIMPLIFIED MULTIPLICATION FOR FRAMES, REMOVING BATCH AND LOOPING OVER CHAIN LENGTH
 
        # seq_i is the index for the sequence
        # s_i is given nt in the index form [2, 2, 0, 2]
        # RTs_in_base_frame:
        #    eps(p)/zeta(p): 0-1
        #    alpha/beta/gamma/delta: 2-5
        #    nu2/nu1/nu0: 6-8
        #    chi_1(na): 9


        for seq_i, s_i in enumerate(self.seq_index.squeeze().tolist()):

            # NA alpha O5'
            RTF1[seq_i,:,:] = torch.einsum(
                'ij,jk,kl->il',
            RTF0[seq_i,:,:], self.RTs_in_base_frame[s_i,2,:], make_rotX_chi(alphas[:,:,2,:],seq_i)) 

                
            # NA beta 
            RTF2[seq_i,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF1[seq_i,:,:], self.RTs_in_base_frame[s_i,3,:], make_rotX_chi(alphas[:,:,3,:],seq_i))

            # NA gamma
            RTF3[seq_i,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF2[seq_i,:,:], self.RTs_in_base_frame[s_i,4,:], make_rotX_chi(alphas[:,:,4,:], seq_i))

            # NA delta
            RTF4[seq_i,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF3[seq_i,:,:], self.RTs_in_base_frame[s_i,5,:], make_rotX_chi(alphas[:,:,5,:],seq_i))

            # NA nu2 - from gamma frame
            RTF5[seq_i,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF3[seq_i,:,:], self.RTs_in_base_frame[s_i,6,:], make_rotX_chi(alphas[:,:,6,:],seq_i))

            # NA nu1
            RTF6[seq_i,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF5[seq_i,:,:], self.RTs_in_base_frame[s_i,7,:], make_rotX_chi(alphas[:,:,7,:], seq_i))
                
            # NA nu0
            RTF7[seq_i,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF6[seq_i,:,:], self.RTs_in_base_frame[s_i,8,:], make_rotX_chi(alphas[:,:,8,:], seq_i))


            # NA chi - from nu1 frame
            RTF8[seq_i,:,:]= torch.einsum(
                'ij,jk,kl->il', 
                RTF6[seq_i,:,:], self.RTs_in_base_frame[s_i,9,:], make_rotX_chi(alphas[:,:,9,:],seq_i))

        RTframes = torch.stack((
            RTF0,
            RTF1,RTF2,RTF3,RTF4,RTF5,RTF6,RTF7, RTF8
        ),dim=1)

        #ic(basexyzs)
        #seq = torch.tensor([[2, 1]]) # gc
        seq_index = self.seq_index
        #ic(seq)
        #ic(self.base_indices)#[seq])

        # RTframes.shape: torch.Size([2, 9, 4, 4]) # 9 frames, batch removed!
        #torch.Size([1, 2, 17, 4, 4]) #17 frames! here
        #torch.Size([2, 9, 4, 4])
        #with unsqueeze:
        #torch.Size([1, 2, 9, 4, 4])
        RTframes = torch.unsqueeze(RTframes, dim=0)
        #ic(RTframes.shape)
        # biseq.shape: 
        # rf torch.Size([1, 2, 36, 4, 4])
        #    torch.Size([1, 2, 34, 4, 4])
        biseq = self.base_indices[seq_index][...,None,None].repeat(1,1,1,4,4) #)#.shape)
        ic(biseq)
        #ic(biseq.shape)

        # torch.Size([1, 2, 36, 4, 4])
        gather = RTframes.gather(2, biseq)
        #ic(gather.shape, basexyzs.unsqueeze(dim=0).shape)

        # torch.Size([5, 34, 4])
        # torch.Size([1, 2, 36, 4]) rf
        #ic(basexyzs.shape)
        basexyzs = self.xyzs_in_base_frame[seq_index]

        xyzs = torch.einsum('brtij,brtj->brti', gather, basexyzs)
        #sys.exit(0)
        #RTframes.gather(baseq)
        #x = RTframes.gather(2, biseq) 
        #basexyzs = basexyzs.squeeze(axis, (axis=0)
        #sys.exit(0)
        #print(xyzs)
        #pbrint(xyzs[...,:3])
        return RTframes, xyzs[...,:3]

    ####Functions from util.py RFNA#####
    
    def th_dih_v(self, ab,bc,cd):
        def th_cross(a,b):
            a,b = torch.broadcast_tensors(a,b)
            return torch.cross(a,b, dim=-1)
        def th_norm(x,eps:float=1e-8):
            return x.square().sum(-1,keepdim=True).add(eps).sqrt()
        def th_N(x,alpha:float=0):
            return x/th_norm(x).add(alpha)

        ab, bc, cd = th_N(ab),th_N(bc),th_N(cd)
        n1 = th_N( th_cross(ab,bc) )
        n2 = th_N( th_cross(bc,cd) )
        sin_angle = (th_cross(n1,bc)*n2).sum(-1)
        cos_angle = (n1*n2).sum(-1)
        dih = torch.stack((cos_angle,sin_angle),-1)
        return dih

    def th_dih(self, a,b,c,d):
        return self.th_dih_v(a-b,b-c,c-d)
    
    def idealize_reference_frame(self, xyz_in):
        xyz = xyz_in.clone()
        Rs, Ts = torch.Tensor(self.listRs).unsqueeze(0), torch.squeeze(torch.Tensor(self.listTs)).unsqueeze(0)

        OP1ideal = torch.tensor([-0.7319, 1.2920, 0.000], device=xyz_in.device)
        OP2ideal = torch.tensor([1.4855, 0.000, 0.000], device=xyz_in.device)
        xyz[:,:,0,:] = torch.einsum('...ij,j->...i', Rs, OP1ideal) + Ts
        xyz[:,:,2,:] = torch.einsum('...ij,j->...i', Rs, OP2ideal) + Ts

        return xyz

    def get_torsions(self, xyz_in, torsion_indices, torsion_can_flip, ref_angles, mask_in=None):
        B,L = xyz_in.shape[:2]

        # tors_mask = get_tor_mask(seq, torsion_indices, mask_in)

        # idealize given xyz coordinates before computing torsion angles
        xyz = self.idealize_reference_frame(xyz_in)

        for _id in self.seq_types:
            if _id == "pur":
                ts = torsion_indices[0]
            else:
                ts = torsion_indices[1]
        bs = torch.arange(B, device=xyz_in.device)[:,None,None,None]
        xs = torch.arange(L, device=xyz_in.device)[None,:,None,None] - (ts<0)*1 # ts<-1 ==> prev res
        ys = torch.abs(ts)
        xyzs_bytor = xyz[bs,xs,ys,:]

        torsions = torch.zeros( (B,L,NTOTALDOFS,2), device=xyz_in.device )
        torsions[...,:,:] = self.th_dih(
            xyzs_bytor[...,:,0,:],xyzs_bytor[...,:,1,:],xyzs_bytor[...,:,2,:],xyzs_bytor[...,:,3,:]
        )
        
        mask0 = (torch.isnan(torsions[...,0])).nonzero()
        mask1 = (torch.isnan(torsions[...,1])).nonzero()
        torsions[mask0[:,0],mask0[:,1],mask0[:,2],0] = 1.0
        torsions[mask1[:,0],mask1[:,1],mask1[:,2],1] = 0.0

        # alt chis
        # torsions_alt = torsions.clone()
        # torsions_alt[torsion_can_flip[seq,:]] *= -1

        return torsions #, torsions_alt, tors_mask

def get_parser():
    parser = argparse.ArgumentParser()
       # description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    #parser.add_argument('-', "--", help="", default="")

    parser.add_argument("-v", "--verbose",
                        action="store_true", help="be verbose")
    parser.add_argument("-o", "--output", help="output structure in the PDB format, by default: output.pdb", default="output.pdb")
    parser.add_argument("--chemicals",
                        help="elect chemicals.py file to be used, by default: chemicals.py",
                        default="chemicals.py")
    parser.add_argument("file", help="", default="") # nargs='+')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    ic.configureOutput(outputFunction=lambda *a: print(*a, file=sys.stdout), includeContext=True)
    #ic.configureOutput(outputFunction=lambda *a: print(*a, file=open('out', 'w')),includeContext=True)
    ic.configureOutput(prefix='')

    ic.disable()
    if args.verbose:
        ic.enable()

    # load different chemicals
    import importlib
    ic(args.chemicals)
    #chemicals = importlib.import_module(args.chemicals, package=None)
    #with open(args.chemicals) as f:
    #     eval(f.read())
    #import imp
    #imp.load_module('chemicals', filename=args.chemicals)
    #import imp
    #imp.
    #eval('from chemicals import *')
    #'ideal_coords.py'
    #'./chemicals_OP1off.py'
    exec(open(args.chemicals).read())
    #from chemicals import *
    #ic(chemicals)
    # read PDB file
    sloppyparser = PDBParser(
        PERMISSIVE=True, structure_builder=xpdb.SloppyStructureBuilder()
    )
    structure = sloppyparser.get_structure("", args.file) 
    print('Input:' + args.file)

    base_indices, RTs_by_torsion, xyzs_in_base_frame, torsion_indices = make_ideal_RTs()

    listRs, listTs, seq_types, seq, seq_index = compute_backbone_frames(structure, frame=["OP1","P", "OP2", "OP1", "P","OP2"])
    xyzs_in = read_atoms(structure)

    # Running algorithm 24
    c = ComputeAllAtomCoords()
    alphas = c.get_torsions(xyzs_in, torsion_indices, False, None) 
    RTframes,  xyzs = c(alphas)
    # Write output file
    writepdb(args.output, xyzs, seq_index)#[0, -1])#, bfacts=best_lddt[0].float())
    print('Output:' + args.output)
