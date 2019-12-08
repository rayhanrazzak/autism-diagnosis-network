import math
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: an array of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width, depth] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        z_wid = int(math.ceil(previous_conv_size[2] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
        z_pad = (z_wid*out_pool_size[i] - previous_conv_size[2] + 1)/2
        maxpool = nn.MaxPool3d((h_wid, w_wid, z_wid), stride=(h_wid, w_wid, z_wid), padding=(h_pad, w_pad, z_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp

'''
Source: 

    @article{ouyang2018pedestrian,
  title={Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond},
  author={Ouyang, Xi and Cheng, Yu and Jiang, Yifan and Li, Chun-Liang and Zhou, Pan},
  journal={arXiv preprint arXiv:1804.02047},
  year={2018}
}

@inproceedings{he2014spatial,
  title={Spatial pyramid pooling in deep convolutional networks for visual recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={European conference on computer vision},
  pages={346--361},
  year={2014},
  organization={Springer}
}

'''
