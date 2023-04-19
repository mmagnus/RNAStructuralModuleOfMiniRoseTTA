import torch
from chemicals import *

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
base_indices = torch.full((NAATOKENS,NTOTAL),0, dtype=torch.long) # base frame that builds each atom
xyzs_in_base_frame = torch.ones((NAATOKENS,NTOTAL,4)) # coords of each atom in the base frame
RTs_by_torsion = torch.eye(4).repeat(NAATOKENS,NTOTALTORS,1,1) # torsion frames

# process ideal frames
def make_frame(X, Y):
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
    RTs_by_torsion[i,0,:3,3] = torch.zeros(3)
    RTs_by_torsion[i,1,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,1,:3,3] = torch.zeros(3)

    # alpha
    RTs_by_torsion[i,2,:3,:3] = make_frame(
        xyzs_in_base_frame[i,3,:3] - xyzs_in_base_frame[i,1,:3], # P->O5'
        xyzs_in_base_frame[i,0,:3] - xyzs_in_base_frame[i,1,:3]  # P<-OP1
    )
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
