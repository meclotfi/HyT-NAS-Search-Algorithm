import numpy as np
import json

def copy_layers_config(vector,layers):
    #1
    layers.layer1.num_blocks=1
    layers.layer1.expand_ratio=int(vector[4])
    layers.layer1.out_channels=int(vector[9])

    layers.layer2.num_blocks=int(vector[0])
    layers.layer2.expand_ratio=int(vector[5])
    layers.layer2.out_channels=int(vector[10])

    layers.layer3.n_transformer_blocks=int(vector[1])
    layers.layer3.expand_ratio=int(vector[6])
    layers.layer3.out_channels=min(int(vector[11]),96)
    vector[11]=layers.layer3.out_channels
    layers.layer3.num_heads=int(vector[14])
    layers.layer3.patch_h=int(vector[20])
    layers.layer3.patch_w=int(vector[20])
    layers.layer3.head_dim=highestPowerof2(int(layers.layer3.out_channels))//layers.layer3.num_heads
    layers.layer3.ffn_dim=int(vector[17])*layers.layer3.num_heads*layers.layer3.head_dim

    layers.layer4.n_transformer_blocks=int(vector[2])
    layers.layer4.expand_ratio=int(vector[7])
    layers.layer4.out_channels=min(int(vector[12]),128)
    vector[12]=layers.layer4.out_channels
    layers.layer4.num_heads=int(vector[15])
    layers.layer4.patch_h=int(vector[21])
    layers.layer4.patch_w=int(vector[21])
    layers.layer4.head_dim=highestPowerof2(int(layers.layer4.out_channels))//layers.layer4.num_heads
    layers.layer4.ffn_dim=int(vector[18])*layers.layer4.num_heads*layers.layer4.head_dim

    layers.layer5.n_transformer_blocks=int(vector[3])
    layers.layer5.expand_ratio=int(vector[8])
    layers.layer5.out_channels=min(int(vector[13]),256)
    vector[13]=layers.layer3.out_channels
    layers.layer5.num_heads=int(vector[16])
    layers.layer5.patch_h=int(vector[22])
    layers.layer5.patch_w=int(vector[22])
    layers.layer5.head_dim=highestPowerof2(int(layers.layer5.out_channels))//layers.layer5.num_heads
    layers.layer5.ffn_dim=int(vector[19])*layers.layer5.num_heads*layers.layer5.head_dim
    print(layers.layer1)
    print(layers.layer2)
    print(layers.layer3)
    print(layers.layer4)
    print(layers.layer5)

def ConfigsToVector(layers):
    #1
    vector=[0 for _ in range(23)]

    vector[4]=layers.layer1.expand_ratio
    vector[9]=layers.layer1.out_channels
    
    vector[0]=int(layers.layer2.num_blocks)
    vector[5]=layers.layer2.expand_ratio
    vector[10]=layers.layer2.out_channels

    vector[1]=layers.layer3.n_transformer_blocks
    vector[6]=layers.layer3.expand_ratio
    vector[11]=layers.layer3.out_channels
    vector[14]=layers.layer3.num_heads
    vector[20]=layers.layer3.patch_h
    vector[17]=layers.layer3.ffn_dim

    vector[2]=layers.layer4.n_transformer_blocks
    vector[7]=layers.layer4.expand_ratio
    vector[12]=layers.layer4.out_channels
    vector[15]=layers.layer4.num_heads
    vector[21]=layers.layer4.patch_h
    vector[18]=layers.layer4.ffn_dim

    vector[3]=layers.layer5.n_transformer_blocks
    vector[8]=layers.layer5.expand_ratio
    vector[13]=layers.layer3.out_channels
    vector[16]=layers.layer5.num_heads
    vector[22]=layers.layer5.patch_h
    vector[19]=layers.layer5.ffn_dim
    return vector
    
    

def highestPowerof2(n):
 
    res = 0
    for i in range(n, 0, -1):
         
        # If i is a power of 2
        if ((i & (i - 1)) == 0):
            res = i
            break
    return res
def generate_vector_space():
    # number of blocks 0,3
    poss_nb=[1,2,3,4]
    nb=np.random.choice(poss_nb,4)
    
    # exp_ratio 4,8
    poss_exr=[1,2,4]
    exr=np.random.choice(poss_exr,5)
    
    #Out channel 9,13
    poss_c=[8,16,24,32]
    Oc1=np.random.choice(poss_c,1)[0]
    Ocs=[Oc1]
    Oc=Oc1
    poss_ex=[1,1.5,2]
    for i in range(4):
        Oc=np.random.choice(poss_ex,1)[0]*Oc
        Ocs.append(Oc)

    #N heads 14,16
    poss_nh=[1,2,4]
    nh=np.random.choice(poss_nh,3)
   

    #ffn_ratio 17,19
    poss_fnr=[1,1.5,2]
    fnr=np.random.choice(poss_fnr,3)

     #patches 20,22
    poss_patch=[2,4,8]
    patch=np.random.choice(poss_patch,3)

    return np.concatenate((nb,exr,Ocs,nh,fnr,patch))


