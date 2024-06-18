from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell

def convlstm_encoder_params(in_chan=1, seq_len=8, image_size=60):
    size_l1 = image_size
    size_l2 = image_size - (image_size // 4)
    size_l3 = image_size - (image_size // 2)
    size_l4 = size_l1 - size_l2
    #print(size_l1,size_l2,size_l3,size_l4)
    #print(seq_len)

    convlstm_encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [in_chan, size_l4, 3, 1, 1]}),  # [1, 32, 3, 1, 1]
            OrderedDict({'conv2_leaky_1': [size_l3, size_l3, 3, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [size_l2, size_l2, 3, 2, 1]}),
        ],
        [
            CLSTM_cell(shape=(size_l1,size_l1), input_channels=size_l4, filter_size=5, num_features=size_l3, seq_len=seq_len),
            CLSTM_cell(shape=(size_l3,size_l3), input_channels=size_l3, filter_size=5, num_features=size_l2, seq_len=seq_len),
            CLSTM_cell(shape=(size_l4,size_l4), input_channels=size_l2, filter_size=5, num_features=size_l1, seq_len=seq_len)
        ]
    ]
    return convlstm_encoder_params

def convlstm_decoder_params(seq_len=12,image_size=60):
    size_l1 = image_size
    size_l2 = image_size - (image_size // 4)
    size_l3 = image_size - (image_size // 2)
    size_l4 = size_l1 - size_l2

    convlstm_decoder_params = [
        [
            OrderedDict({'deconv1_leaky_1': [size_l1, size_l1, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [size_l2, size_l2, 4, 2, 1]}),
            OrderedDict({
                'conv3_leaky_1': [size_l3, size_l4, 3, 1, 1],
                'conv4_leaky_1': [size_l4, 1, 1, 1, 0]
            }),
        ],
        [
            CLSTM_cell(shape=(size_l4,size_l4), input_channels=size_l1, filter_size=5, num_features=size_l1, seq_len=seq_len),
            CLSTM_cell(shape=(size_l3,size_l3), input_channels=size_l1, filter_size=5, num_features=size_l2, seq_len=seq_len),
            CLSTM_cell(shape=(size_l1,size_l1), input_channels=size_l2, filter_size=5, num_features=size_l3, seq_len=seq_len),
        ]
    ]
    return convlstm_decoder_params


# convlstm_encoder_params = convlstm_encoder_params(image_size=60)
# convlstm_decoder_params = convlstm_decoder_params(image_size=60)
# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]


# convlstm_encoder_params = [
#     [
#         OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
#         OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
#         OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
#     ],

#     [
#         CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64),
#         CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
#         CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
#     ]
# ]

# convlstm_decoder_params = [
#     [
#         OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
#         OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
#         OrderedDict({
#             'conv3_leaky_1': [64, 16, 3, 1, 1],
#             'conv4_leaky_1': [16, 1, 1, 1, 0]
#         }),
#     ],

#     [
#         CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
#         CLSTM_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
#         CLSTM_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
#     ]
# ]

# convgru_encoder_params = [
#     [
#         OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
#         OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
#         OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
#     ],

#     [
#         CGRU_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64),
#         CGRU_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
#         CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
#     ]
# ]

# convgru_decoder_params = [
#     [
#         OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
#         OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
#         OrderedDict({
#             'conv3_leaky_1': [64, 16, 3, 1, 1],
#             'conv4_leaky_1': [16, 1, 1, 1, 0]
#         }),
#     ],

#     [
#         CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
#         CGRU_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
#         CGRU_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
#     ]
# ]