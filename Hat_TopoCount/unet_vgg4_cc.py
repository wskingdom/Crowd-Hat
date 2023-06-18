import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import collections
from distutils.util import strtobool;
from sa_net_arch_utilities_pytorch import CNNArchUtilsPyTorch;

class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')

class UnetVggCC(nn.Module):
    def __init__(self, load_weights=False, kwargs=None):
        super(UnetVggCC,self).__init__()

        # predefined list of arguments
        args = {'input_img_width':-1, 'input_img_height':-1, 'pretrained':'False'
            , 'conv_init': 'he'
            , 'use_softmax':'False', 'use_relu':'False', 'use_tanh':'False'
            ,'n_layers_per_path':4, 'n_conv_blocks_in_start': 64, 'block_size':3, 'pool_size':2
            , 'dropout_keep_prob' : 1.0, 'initial_pad':0, 'interpolate':'False', 'n_classes':1, 'n_channels':3
 
        };

        if(not(kwargs is None)):
            args.update(kwargs);

        # 'conv_init': 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'he'

        # read extra argument
        #self.n_layers_per_path = int(args['n_layers_per_path']); # n_layers_per_path in contracting path + n_layers_per_path in expanding path + 1 bottleneck layer
        #self.n_conv_blocks_in_start = int(args['n_conv_blocks_in_start']);
        self.input_img_width = int(args['input_img_width']);
        self.input_img_height = int(args['input_img_height']);
        self.n_channels = int(args['n_channels']);
        self.n_classes = int(args['n_classes']);
        #dropout = args['dropout'];
        self.pretrained = bool(strtobool(args['pretrained']));
        #self.stain_init_name = str(args['stain_init_name']);
        self.conv_init = str(args['conv_init']).lower();
        self.use_softmax = bool(strtobool(args['use_softmax']));
        self.use_relu = bool(strtobool(args['use_relu']));
        self.use_tanh = bool(strtobool(args['use_tanh']));
    
        self.n_layers_per_path = int(args['n_layers_per_path']);
        self.n_conv_blocks_in_start = int(args['n_conv_blocks_in_start']);
        self.block_size = int(args['block_size']);
        self.pool_size = int(args['pool_size']);
        self.dropout_keep_prob = float(args['dropout_keep_prob'])
        self.initial_pad = int(args['initial_pad']);
        self.interpolate = bool(strtobool(args['interpolate']));

        print('self.initial_pad',self.initial_pad)

        n_blocks = self.n_conv_blocks_in_start;
        n_blocks_prev = self.n_channels;

        # Contracting Path
        #self.encoder = [];
        self.encoder = nn.Sequential()
        layer_index = 0;
        layer = nn.Sequential();
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_0', nn.Conv2d(3, 64, kernel_size=self.block_size, padding=self.initial_pad));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(64, 64, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer);

        layer_index = 1;
        layer = nn.Sequential();
        layer.add_module('encoder_maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_0', nn.Conv2d(64, 128, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(128, 128, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer);

        layer_index = 2;
        layer = nn.Sequential();
        layer.add_module('encoder_maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
        layer.add_module('encoder_conv_l_'+str(layer_index) + '_0', nn.Conv2d(128, 256, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(256, 256, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_2', nn.Conv2d(256, 256, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_2', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer);

        layer_index = 3;
        layer = nn.Sequential();
        layer.add_module('encoder_maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
        layer.add_module('encoder_conv_l_'+str(layer_index) + '_0', nn.Conv2d(256, 512, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(512, 512, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_2', nn.Conv2d(512, 512, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_2', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer);

        self.bottleneck = nn.Sequential();
        self.bottleneck.add_module('bottleneck_maxpool', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
        self.bottleneck.add_module('bottleneck_conv'+ '_0', nn.Conv2d(512, 512, kernel_size=self.block_size));
        self.bottleneck.add_module('bottleneck_relu'+'_0', nn.ReLU(inplace=True))
        self.bottleneck.add_module('bottleneck_conv'+ '_1', nn.Conv2d(512, 512, kernel_size=self.block_size));
        self.bottleneck.add_module('bottleneck_relu'+'_1', nn.ReLU(inplace=True))
        self.bottleneck.add_module('bottleneck_conv'+ '_2', nn.Conv2d(512, 512, kernel_size=self.block_size));
        self.bottleneck.add_module('bottleneck_relu'+'_2', nn.ReLU(inplace=True))
     

        # Expanding Path
        #self.decoder = [];
        self.decoder = nn.Sequential()
        layer_index = 3;
        layer = nn.Sequential();
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(512, 512, stride=self.pool_size, kernel_size=self.pool_size))
        layer.add_module('decoder_conv_l_s_'+str(layer_index)+'_0', nn.Conv2d(1024, 512, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_1', nn.Conv2d(512, 512, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(True));
        self.decoder.add_module('decoder_l_'+str(layer_index), layer);

        layer_index = 2;
        layer = nn.Sequential();
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(512, 256, stride=self.pool_size, kernel_size=self.pool_size))
        layer.add_module('decoder_conv_l_s_'+str(layer_index)+'_0', nn.Conv2d(512, 256, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_1', nn.Conv2d(256, 256, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(True));
        self.decoder.add_module('decoder_l_'+str(layer_index), layer);

        layer_index = 1;
        layer = nn.Sequential();
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(256, 128, stride=self.pool_size, kernel_size=self.pool_size))
        layer.add_module('decoder_conv_l_s_'+str(layer_index)+'_0', nn.Conv2d(256, 128, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_1', nn.Conv2d(128, 128, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(True));
        self.decoder.add_module('decoder_l_'+str(layer_index), layer);

        layer_index = 0;
        layer = nn.Sequential();
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(128, 64, stride=self.pool_size, kernel_size=self.pool_size))
        #layer.add_module('decoder_conv_l_s_'+str(layer_index)+'_0', nn.Conv2d(128, 64, kernel_size=self.block_size));
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_0', nn.Conv2d(64, 64, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_1', nn.Conv2d(64, 64, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(True));
        self.decoder.add_module('decoder_l_'+str(layer_index), layer);

        self.final_layer = nn.Sequential();
        self.final_layer.add_module('conv_final', nn.Conv2d(64, self.n_classes, kernel_size=1));
        if(self.use_relu):
            self.final_layer.add_module('relu_final', nn.ReLU(True));
        if(self.use_tanh):
            self.final_layer.add_module('tanh_final', nn.Tanh());



        # Softmax
        self.softmax_layer = torch.nn.Softmax(dim=1);

        self._initialize_weights()

        self.zero_grad() ;

        print('self.encoder',self.encoder)
        #print('self.bottleneck',self.bottleneck)
        print('self.decoder',self.decoder)

    def forward(self,x):

        encoder_out = [];
        encoder_out = [];
        for l in self.encoder:     
            x = l(x);
            encoder_out.append(x);
        x = self.bottleneck(x);
        j = len(self.decoder);
        for l in self.decoder:            
            x = l[0](x);
            j -= 1;
            corresponding_layer_indx = j;
            ## crop and concatenate
            if(j > 0):
                cropped = CNNArchUtilsPyTorch.crop_a_to_b(encoder_out[corresponding_layer_indx],  x);
                x = torch.cat((cropped, x), 1) ;
            for i in range(1, len(l)):
                x = l[i](x);

        c = self.final_layer(x);

        if(self.use_softmax):
            sm = self.softmax_layer(c);
        else:
            sm = c;
        return sm;

    def _initialize_weights(self):
        BIAS_INIT = 0.1;
        for l in self.encoder:
            for layer in l:
                if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                    if(self.conv_init == 'normal'):
                        torch.nn.init.normal_(layer.weight) ;
                    elif(self.conv_init == 'xavier_uniform'):
                        torch.nn.init.xavier_uniform_(layer.weight) ;
                    elif(self.conv_init == 'xavier_normal'):
                        torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                    elif(self.conv_init == 'he'):
                        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
                        #layer.bias.data.fill_(BIAS_INIT);

        for layer in self.bottleneck:
            if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                if(self.conv_init == 'normal'):
                    torch.nn.init.normal_(layer.weight) ;
                elif(self.conv_init == 'xavier_uniform'):
                    torch.nn.init.xavier_uniform_(layer.weight) ;
                elif(self.conv_init == 'xavier_normal'):
                    torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                elif(self.conv_init == 'he'):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
                    #layer.bias.data.fill_(BIAS_INIT);


        for l in self.decoder:
            for layer in l:
                if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                    if(self.conv_init == 'normal'):
                        torch.nn.init.normal_(layer.weight) ;
                    elif(self.conv_init == 'xavier_uniform'):
                        torch.nn.init.xavier_uniform_(layer.weight) ;
                    elif(self.conv_init == 'xavier_normal'):
                        torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                    elif(self.conv_init == 'he'):
                        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
                        #layer.bias.data.fill_(BIAS_INIT);

        for layer in self.final_layer:
            if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                if(self.conv_init == 'normal'):
                    torch.nn.init.normal_(layer.weight) ;
                elif(self.conv_init == 'xavier_uniform'):
                    torch.nn.init.xavier_uniform_(layer.weight) ;
                elif(self.conv_init == 'xavier_normal'):
                    torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                elif(self.conv_init == 'he'):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
                    #layer.bias.data.fill_(BIAS_INIT);

        vgg_model = models.vgg16(pretrained = True)
        fsd=collections.OrderedDict()
        i = 0
        for m in self.encoder.state_dict().items():
            temp_key=m[0]
            print('temp_key', temp_key)
            print('vgg_key', list(vgg_model.state_dict().items())[i][0])
            fsd[temp_key]=list(vgg_model.state_dict().items())[i][1]
            i += 1
        self.encoder.load_state_dict(fsd)

        fsd=collections.OrderedDict()
        for m in self.bottleneck.state_dict().items():
            temp_key=m[0]
            print('temp_key', temp_key)
            print('vgg_key', list(vgg_model.state_dict().items())[i][0])
            fsd[temp_key]=list(vgg_model.state_dict().items())[i][1]
            i += 1
        self.bottleneck.load_state_dict(fsd)

        #del vgg_model

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False, deconv=None, pad_list=None):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    #for v in cfg:
    for i in range(len(cfg)):
        v=cfg[i]
        print('in_channels=',in_channels)
        print('v=',v)
        if(not (deconv is None)):
            print('deconv[i]=',deconv[i])
        if(pad_list is None):
            padding = d_rate;
        else:
            padding = pad_list[i];
        print('padding =', padding);
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if(deconv is None or deconv[i] == False):
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=padding, dilation = d_rate)
            else:
                conv2d = nn.ConvTranspose2d(in_channels, v, stride=2, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


