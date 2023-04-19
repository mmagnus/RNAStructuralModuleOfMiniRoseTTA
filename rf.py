#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import argparse

from chemicals import *
from make_ideal_RTs import base_indices, RTs_by_torsion, xyzs_in_base_frame, torsion_indices

ic.configureOutput(outputFunction=lambda *a: print(*a, file=sys.stderr), includeContext=True)
ic.configureOutput(prefix='')

def read_atoms(structure):
    """
    Args:
       structure (BioPython structure): a structure

    Returns:
      tensor: with coords::
 
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
     - seq_id (list): e.g. ['pur', 'pyr']
     - seq_name (list): ['G', 'C']
     - seq (tensor): seq: e.g., tensor([0, 1]) where A = 0, C = 1, G = 2, U = 3, X = 4
    """
    listRs, listTs = [], []
    seq_id, seq_name = [], []
    seq = []
    for residue in structure.get_residues():
        if residue.get_resname() in ["A","C","G","U"] and residue.has_id('OP1') and residue.has_id('OP2'):
            if residue.has_id('N9'): # PURINES
                x1 = residue[frame[0]]
                x2 = residue[frame[1]]
                x3 = residue[frame[2]]
                seq_id.append("pur")
                seq_name.append(residue.get_resname())

            else: # PYRAMIDINES
                x1 = residue[frame[3]]
                x2 = residue[frame[4]]
                x3 = residue[frame[5]]
                seq_id.append("pyr")
                seq_name.append(residue.get_resname())
 
            x1_arr = np.array(x1.get_coord())[:,None]
            x2_arr = np.array(x2.get_coord())[:,None]
            x3_arr = np.array(x3.get_coord())[:,None]

            seq.append(["A","C","G","U"].index(residue.get_resname()))

            R, t = rigidFrom3Points(x1_arr, x2_arr, x3_arr)
            listRs.append(R)
            listTs.append(t)

    seq = torch.tensor([seq])
    return listRs, listTs, seq_id, seq_name, seq


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
        - seq_id: ['pur', 'pyr']
        - seq: e.g., tensor([0, 1]) where A = 0, C = 1, G = 2, U = 3, X = 4
        """
        super(ComputeAllAtomCoords, self).__init__()
        self.base_indices = nn.Parameter(base_indices, requires_grad=False) 
        self.RTs_in_base_frame = nn.Parameter(RTs_by_torsion, requires_grad=False)
        self.xyzs_in_base_frame = nn.Parameter(xyzs_in_base_frame, requires_grad=False) 
        self.listRs = np.array(listRs)
        self.listTs = np.array(listTs)
        self.seq_id = seq_id
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
        ic(L)

        RTF0 = torch.eye(4).repeat(L,1,1).to(device=Rs.device)

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
             
        dict_res = {'A':0, 'C':1, 'G':2, 'U':3}

        RTF1, RTF2, RTF3, RTF4, RTF5, RTF6, RTF7, RTF8 = torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0),torch.empty_like(RTF0)
        # SIMPLIFIED MULTIPLICATION FOR FRAMES, REMOVING BATCH AND LOOPING OVER CHAIN LENGTH
        for inde, name in enumerate(seq_name):
            # NA alpha O5'
            RTF1[inde,:,:] = torch.einsum(
                'ij,jk,kl->il',
                RTF0[inde,:,:], self.RTs_in_base_frame[dict_res[name],2,:], make_rotX_chi(alphas[:,:,2,:],inde)) 

            # NA beta 
            RTF2[inde,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF1[inde,:,:], self.RTs_in_base_frame[dict_res[name],3,:], make_rotX_chi(alphas[:,:,3,:],inde))

            # NA gamma
            RTF3[inde,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF2[inde,:,:], self.RTs_in_base_frame[dict_res[name],4,:], make_rotX_chi(alphas[:,:,4,:], inde))

            # NA delta
            RTF4[inde,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF3[inde,:,:], self.RTs_in_base_frame[dict_res[name],5,:], make_rotX_chi(alphas[:,:,5,:],inde))

            # NA nu2 - from gamma frame
            RTF5[inde,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF3[inde,:,:], self.RTs_in_base_frame[dict_res[name],6,:], make_rotX_chi(alphas[:,:,6,:],inde))

            # NA nu1
            RTF6[inde,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF5[inde,:,:], self.RTs_in_base_frame[dict_res[name],7,:], make_rotX_chi(alphas[:,:,7,:], inde))

            # NA nu0
            RTF7[inde,:,:] = torch.einsum(
                'ij,jk,kl->il', 
                RTF6[inde,:,:], self.RTs_in_base_frame[dict_res[name],8,:], make_rotX_chi(alphas[:,:,8,:], inde))

            # NA chi - from nu1 frame
            RTF8[inde,:,:]= torch.einsum(
                'ij,jk,kl->il', 
                RTF6[inde,:,:], self.RTs_in_base_frame[dict_res[name],9,:], make_rotX_chi(alphas[:,:,9,:],inde))

        RTframes = torch.stack((
            RTF0,
            RTF1,RTF2,RTF3,RTF4,RTF5,RTF6,RTF7, RTF8
        ),dim=1)

        #ic(basexyzs)
        #seq = torch.tensor([[2, 1]]) # gc
        seq = self.seq
        ic(seq)
        #ic(self.base_indices)#[seq])

        # RTframes.shape: torch.Size([2, 9, 4, 4]) # 9 frames, batch removed!
        #torch.Size([1, 2, 17, 4, 4]) #17 frames! here
        #torch.Size([2, 9, 4, 4])
        #with unsqueeze:
        #torch.Size([1, 2, 9, 4, 4])
        RTframes = torch.unsqueeze(RTframes, dim=0)
        ic(RTframes.shape)
        # biseq.shape: 
        # rf torch.Size([1, 2, 36, 4, 4])
        #    torch.Size([1, 2, 34, 4, 4])
        biseq = self.base_indices[seq][...,None,None].repeat(1,1,1,4,4) #)#.shape)
        #ic(biseq.shape)

        # torch.Size([1, 2, 36, 4, 4])
        gather = RTframes.gather(2, biseq)
        #ic(gather.shape, basexyzs.unsqueeze(dim=0).shape)

        # torch.Size([5, 34, 4])
        # torch.Size([1, 2, 36, 4]) rf
        #ic(basexyzs.shape)
        basexyzs = self.xyzs_in_base_frame[seq]

        xyzs = torch.einsum('brtij,brtj->brti', gather, basexyzs)
        #sys.exit(0)
        #RTframes.gather(baseq)
        #x = RTframes.gather(2, biseq) 
        #basexyzs = basexyzs.squeeze(axis, (axis=0)
        #sys.exit(0)
        print(xyzs)
        print(xyzs[...,:3])

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

        for _id in self.seq_id:
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    #parser.add_argument('-', "--", help="", default="")

    parser.add_argument("-v", "--verbose",
                        action="store_true", help="be verbose")
    parser.add_argument("--output", help="", default="output.pdb")
    parser.add_argument("file", help="", default="") # nargs='+')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # read PDB file
    sloppyparser = PDBParser(
        PERMISSIVE=True, structure_builder=xpdb.SloppyStructureBuilder()
    )
    structure = sloppyparser.get_structure("", args.file) 
    ic(structure)
    ic('File:' + args.file)
    
    listRs, listTs, seq_id, seq_name, seq = compute_backbone_frames(structure, frame=["OP1","P", "OP2", "OP1", "P","OP2"])
    xyzs_in = read_atoms(structure)

    # Running algorithm 24
    c = ComputeAllAtomCoords()
    alphas = c.get_torsions(xyzs_in, torsion_indices, False, None) 
    RTframes,  xyzs = c(alphas)

    # Write output file
    writepdb(args.output, xyzs, seq)#[0, -1])#, bfacts=best_lddt[0].float())
