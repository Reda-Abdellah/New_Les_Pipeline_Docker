import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################
########################################

class Generator(nn.Module): #mask_generator
    #this is the implementation of assemblynet architecture from tenserflow/keras model.
    def __init__(self,nf=24,nc=1,dropout_rate=0.5,in_shape=1):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv3d(in_shape, nf, 3, padding= 1)
        self.conv1_bn = nn.BatchNorm3d(nf)
        self.conv2 = nn.Conv3d(nf, 2*nf, 3, padding= 1)
        self.conv2_bn = nn.BatchNorm3d(2*nf)
        self.conv3 = nn.Conv3d(2*nf, 2*nf, 3, padding= 1)
        self.conv3_bn = nn.BatchNorm3d(2*nf)
        self.conv4 = nn.Conv3d(2*nf, 4*nf, 3, padding= 1)
        self.conv4_bn = nn.BatchNorm3d(4*nf)
        self.conv5 = nn.Conv3d(4*nf, 4*nf, 3, padding= 1)
        self.conv5_bn = nn.BatchNorm3d(4*nf)
        self.conv6 = nn.Conv3d(4*nf, 8*nf, 3, padding= 1)
        self.conv6_bn = nn.BatchNorm3d(8*nf)
        self.conv7 = nn.Conv3d(8*nf, 16*nf, 3, padding= 1)
        #bottleneck
        self.concat1_bn = nn.BatchNorm3d(8*nf+16*nf)
        self.conv8 = nn.Conv3d((8*nf+16*nf), 8*nf, 3, padding= 1)

        self.concat2_bn = nn.BatchNorm3d(8*nf+4*nf)
        self.conv9 = nn.Conv3d(8*nf+4*nf, 4*nf, 3, padding= 1)

        self.concat3_bn = nn.BatchNorm3d(4*nf+2*nf)
        self.conv10 = nn.Conv3d(4*nf+2*nf, 4*nf, 3, padding= 1)

        self.conv_out = nn.Conv3d(4*nf, nc, 3, padding= 1)
        self.up= nn.Upsample(scale_factor=2,mode='trilinear', align_corners=False)
        #self.up= nn.Upsample(scale_factor=2,mode='nearest')
        self.pool= nn.MaxPool3d(2)
        self.dropout= nn.Dropout(dropout_rate)
        self.final_activation=nn.Sigmoid()

    def encoder(self,in_x):
        self.x1=self.conv1_bn(F.relu(self.conv1(in_x)))
        self.x1= F.relu(self.conv2(self.x1))
        self.x2= self.conv2_bn(self.dropout(self.pool(self.x1)))
        self.x2=self.conv3_bn(F.relu(self.conv3(self.x2)))
        self.x2= F.relu(self.conv4(self.x2))
        self.x3= self.conv4_bn(self.dropout(self.pool(self.x2)))
        self.x3=self.conv5_bn(F.relu(self.conv5(self.x3)))
        self.x3= F.relu(self.conv6(self.x3))
        self.x4= self.conv6_bn(self.dropout(self.pool(self.x3)))
        self.x4=F.relu(self.conv7(self.x4))#bottleneck

    def decoder(self):
        self.x5=self.up(self.x4)
        self.x5=self.concat1_bn(  torch.cat((self.x5,self.x3), dim=1)  )
        self.x5= F.relu(self.conv8(self.x5))
        self.x6=self.up(self.x5)
        self.x6=self.concat2_bn(  torch.cat((self.x6,self.x2), dim=1)  )
        self.x6= F.relu(self.conv9(self.x6))
        self.x7=self.up(self.x6)
        self.x7=self.concat3_bn(  torch.cat((self.x7,self.x1), dim=1)  )
        self.x7= F.relu(self.conv10(self.x7))
        return self.x7

    def forward(self, x):
        self.encoder(x)
        decoder_out=self.decoder()
        out= self.final_activation(self.conv_out(decoder_out))
        return out

class unet_assemblynet(nn.Module):
    #this is the implementation of assemblynet architecture from tenserflow/keras model.
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv3d(in_mod, nf, 3, padding= 1)
        self.conv1_bn = nn.BatchNorm3d(nf)
        self.conv2 = nn.Conv3d(nf, 2*nf, 3, padding= 1)
        self.conv2_bn = nn.BatchNorm3d(2*nf)
        self.conv3 = nn.Conv3d(2*nf, 2*nf, 3, padding= 1)
        self.conv3_bn = nn.BatchNorm3d(2*nf)
        self.conv4 = nn.Conv3d(2*nf, 4*nf, 3, padding= 1)
        self.conv4_bn = nn.BatchNorm3d(4*nf)
        self.conv5 = nn.Conv3d(4*nf, 4*nf, 3, padding= 1)
        self.conv5_bn = nn.BatchNorm3d(4*nf)
        self.conv6 = nn.Conv3d(4*nf, 8*nf, 3, padding= 1)
        self.conv6_bn = nn.BatchNorm3d(8*nf)
        self.conv7 = nn.Conv3d(8*nf, 16*nf, 3, padding= 1)
        #bottleneck
        self.concat1_bn = nn.BatchNorm3d(8*nf+16*nf)
        #print('hak swalhak')
        self.conv8 = nn.Conv3d((8*nf+16*nf), 8*nf, 3, padding= 1)

        self.concat2_bn = nn.BatchNorm3d(8*nf+4*nf)
        self.conv9 = nn.Conv3d(8*nf+4*nf, 4*nf, 3, padding= 1)

        self.concat3_bn = nn.BatchNorm3d(4*nf+2*nf)
        self.conv10 = nn.Conv3d(4*nf+2*nf, 4*nf, 3, padding= 1)

        self.conv_out = nn.Conv3d(4*nf, nc, 3, padding= 1)
        self.up= nn.Upsample(scale_factor=2,mode='trilinear', align_corners=False)
        #self.up= nn.Upsample(scale_factor=2,mode='nearest')
        self.pool= nn.MaxPool3d(2)
        self.dropout= nn.Dropout(dropout_rate)
        self.softmax=nn.Softmax(dim=1)

    def encoder(self,in_x):
        self.x1=self.conv1_bn(F.relu(self.conv1(in_x)))
        self.x1= F.relu(self.conv2(self.x1))
        self.x2= self.conv2_bn(self.dropout(self.pool(self.x1)))
        self.x2=self.conv3_bn(F.relu(self.conv3(self.x2)))
        self.x2= F.relu(self.conv4(self.x2))
        self.x3= self.conv4_bn(self.dropout(self.pool(self.x2)))
        self.x3=self.conv5_bn(F.relu(self.conv5(self.x3)))
        self.x3= F.relu(self.conv6(self.x3))
        self.x4= self.conv6_bn(self.dropout(self.pool(self.x3)))
        self.x4=F.relu(self.conv7(self.x4))#bottleneck
        return self.x4,self.x3,self.x2,self.x1

    def decoder(self,x4,x3,x2,x1):
        self.x5=self.up(x4)
        self.x5=self.concat1_bn(  torch.cat((self.x5,x3), dim=1)  )
        #self.x5=self.cat1_bn(  torch.cat((self.x5,x3), dim=1)  )
        self.x5= F.relu(self.conv8(self.x5))
        self.x6=self.up(self.x5)
        self.x6=self.concat2_bn(  torch.cat((self.x6,x2), dim=1)  )
        #self.x6=self.cat2_bn(  torch.cat((self.x6,x2), dim=1)  )
        self.x6= F.relu(self.conv9(self.x6))
        self.x7=self.up(self.x6)
        self.x7=self.concat3_bn(  torch.cat((self.x7,x1), dim=1)  )
        #self.x7=self.cat3_bn(  torch.cat((self.x7,x1), dim=1)  )
        self.x7= F.relu(self.conv10(self.x7))
        return self.softmax(self.conv_out(self.x7))

    def forward(self, x):
        x4,x3,x2,x1 = self.encoder(x)
        decoder_out=self.decoder(x4,x3,x2,x1)
        return decoder_out

class unet_assemblynet_groupnorm(unet_assemblynet):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1, group_number=3):
        super().__init__(nf,nc,dropout_rate,in_mod)
        self.conv1_bn = nn.GroupNorm(group_number, nf)
        self.conv2_bn = nn.GroupNorm(2*group_number, 2*nf)
        self.conv3_bn = nn.GroupNorm(2*group_number, 2*nf)
        self.conv4_bn = nn.GroupNorm(4*group_number, 4*nf)
        self.conv5_bn = nn.GroupNorm(4*group_number, 4*nf)
        self.conv6_bn = nn.GroupNorm(8*group_number, 8*nf)
        #bottleneck
        self.concat1_bn = nn.GroupNorm(24*group_number, 24*nf)
        self.concat2_bn = nn.GroupNorm(12*group_number, 12*nf)
        self.concat3_bn = nn.GroupNorm(6*group_number, 6*nf)



class unet_siamese(unet_assemblynet_groupnorm):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1, group_number=3):
        super().__init__(nf,nc,dropout_rate,in_mod)

    def forward(self, x):
        x4_1,x3_1,x2_1,x1_1 = self.encoder(x[:,0:1,:,:,:])
        x4_2,x3_2,x2_2,x1_2 = self.encoder(x[:,1:2,:,:,:])
        x3=F.relu(x3_2-x3_1)
        x2=F.relu(x2_2-x2_1)
        x1=F.relu(x1_2-x1_1)
        decoder_out=self.decoder(x4_1,x3,x2,x1)
        return decoder_out

class unet_siamese_abs(unet_assemblynet_groupnorm):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1, group_number=3):
        super().__init__(nf,nc,dropout_rate,in_mod)

    def forward(self, x):
        x4_1,x3_1,x2_1,x1_1 = self.encoder(x[:,0:1,:,:,:])
        x4_2,x3_2,x2_2,x1_2 = self.encoder(x[:,1:2,:,:,:])
        x4=torch.abs(x4_2-x4_1)
        x3=torch.abs(x3_2-x3_1)
        x2=torch.abs(x2_2-x2_1)
        x1=torch.abs(x1_2-x1_1)
        decoder_out=self.decoder(x4_1,x3,x2,x1)
        return decoder_out

class unet_siamese_fractal(unet_assemblynet_groupnorm):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1, group_number=3):
        super().__init__(nf,nc,dropout_rate,in_mod)
        self.fusion1=CATFusion3D(nf*2,nf*2, nf*2, norm = 'GroupNorm', norm_groups=group_number*2, ftdepth=5)
        self.fusion2=CATFusion3D(nf*4,nf*4, nf*4, norm = 'GroupNorm', norm_groups=group_number*4, ftdepth=5)
        self.fusion3=CATFusion3D(nf*8,nf*8, nf*8, norm = 'GroupNorm', norm_groups=group_number*8, ftdepth=5)
        self.fusion4=CATFusion3D(nf*16,nf*16, nf*16, norm = 'GroupNorm', norm_groups=group_number*16, ftdepth=5)

    def forward(self, x):
        x4_1,x3_1,x2_1,x1_1 = self.encoder(x[:,0:1,:,:,:])
        x4_2,x3_2,x2_2,x1_2 = self.encoder(x[:,1:2,:,:,:])
        x4=self.fusion4(x4_1, x4_2)
        x3=self.fusion3(x3_1, x3_2)
        x2=self.fusion2(x2_1, x2_2)
        x1=self.fusion1(x1_1, x1_2)
        decoder_out=self.decoder(x4,x3,x2,x1)
        return decoder_out

class generator(unet_assemblynet_groupnorm):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_shape=1):
        super().__init__(nf,nc,dropout_rate,in_shape)
    def decoder(self):
        self.x5=self.up(self.x4)
        self.x5=self.concat1_bn(  torch.cat((self.x5,self.x3), dim=1)  )
        self.x5= F.relu(self.conv8(self.x5))
        self.x6=self.up(self.x5)
        self.x6=self.concat2_bn(  torch.cat((self.x6,self.x2), dim=1)  )
        self.x6= F.relu(self.conv9(self.x6))
        self.x7=self.up(self.x6)
        self.x7=self.concat3_bn(  torch.cat((self.x7,self.x1), dim=1)  )
        self.x7= F.relu(self.conv10(self.x7))
        return self.x7
    def forward(self, x):
        self.encoder(x)
        decoder_out=self.decoder()
        out= self.conv_out(decoder_out)
        return out

class detection_decoder(nn.Module):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.combine4_bn = nn.BatchNorm3d((16*nf)*2)
        self.combine4 = nn.Conv3d((16*nf)*2, (16*nf), 3, padding= 1)
        self.combine3_bn = nn.BatchNorm3d((8*nf)*2)
        self.combine3 = nn.Conv3d((8*nf)*2, (8*nf), 3, padding= 1)
        self.combine2_bn = nn.BatchNorm3d((4*nf)*2)
        self.combine2 = nn.Conv3d((4*nf)*2, (4*nf), 3, padding= 1)
        self.combine1_bn = nn.BatchNorm3d((2*nf)*2)
        self.combine1 = nn.Conv3d((2*nf)*2, (2*nf), 3, padding= 1)

        self.concat1_bn = nn.BatchNorm3d(8*nf+16*nf)
        self.conv8 = nn.Conv3d((8*nf+16*nf), 8*nf, 3, padding= 1)
        self.concat2_bn = nn.BatchNorm3d(8*nf+4*nf)
        self.conv9 = nn.Conv3d(8*nf+4*nf, 4*nf, 3, padding= 1)
        self.concat3_bn = nn.BatchNorm3d(4*nf+2*nf)
        self.conv10 = nn.Conv3d(4*nf+2*nf, 4*nf, 3, padding= 1)
        self.conv_out = nn.Conv3d(4*nf, nc, 3, padding= 1)
        self.up= nn.Upsample(scale_factor=2,mode='trilinear', align_corners=False)
        self.pool= nn.MaxPool3d(2)
        self.dropout= nn.Dropout(dropout_rate)
        self.softmax=nn.Softmax(dim=1)
    
    def combine_FMs(self, FM1, FM2, norm_layer, conv_layer):
        out=  torch.cat((FM1,FM2), dim=1) 
        out= norm_layer(out)
        out= F.relu(conv_layer(out))
        return out
    
    def decoder(self,x4,x3,x2,x1):
        self.x5=self.up(x4)
        self.x5=self.concat1_bn(  torch.cat((self.x5,x3), dim=1)  )
        #self.x5=self.cat1_bn(  torch.cat((self.x5,x3), dim=1)  )
        self.x5= F.relu(self.conv8(self.x5))
        self.x6=self.up(self.x5)
        self.x6=self.concat2_bn(  torch.cat((self.x6,x2), dim=1)  )
        #self.x6=self.cat2_bn(  torch.cat((self.x6,x2), dim=1)  )
        self.x6= F.relu(self.conv9(self.x6))
        self.x7=self.up(self.x6)
        self.x7=self.concat3_bn(  torch.cat((self.x7,x1), dim=1)  )
        #self.x7=self.cat3_bn(  torch.cat((self.x7,x1), dim=1)  )
        self.x7= F.relu(self.conv10(self.x7))
        return self.softmax(self.conv_out(self.x7))

    def forward(self, x_1,x_2):
        [x4_1,x3_1,x2_1,x1_1] = x_1
        [x4_2,x3_2,x2_2,x1_2] = x_2
        x4_combined=self.combine_FMs(x4_1, x4_2, self.combine4_bn, self.combine4)
        x3_combined=self.combine_FMs(x3_1, x3_2, self.combine3_bn, self.combine3)
        x2_combined=self.combine_FMs(x2_1, x2_2, self.combine2_bn, self.combine2)
        x1_combined=self.combine_FMs(x1_1, x1_2, self.combine1_bn, self.combine1)
        decoder_out=self.decoder(x4_combined,x3_combined,x2_combined,x1_combined)
        return decoder_out
