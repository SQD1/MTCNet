import torch
import torch.nn as nn
from torchvision.models.resnet import resnet34

nonlinearity=nn.LeakyReLU
activaionF=nn.functional.leaky_relu
class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 n_filters=256,
                 kernel_size=3,
                 is_deconv=False,
                 ):
        super().__init__()

        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels // 4,
                               kernel_size,
                               padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding, bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)

        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4,
                               n_filters,
                               kernel_size,
                               padding=conv_padding, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x




class ChgDecoderBlock1(nn.Module):
    def __init__(self, in_channels=512, n_filters=256, if_last_block=False):
        super().__init__()
        # in_channels 为 last stage chg的通道数
        if if_last_block:  # 如果是最后一层decoder 则chg与x，y的通道数相同，都是64
            self.convx1 = nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=False)
            self.convx2 = nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=False)

            self.convy1 = nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=False)
            self.convy2 = nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=False)

        else:
            self.convx1 = nn.Conv2d(in_channels//2, in_channels, 1, padding=0, bias=False)
            self.convx2 = nn.Conv2d(in_channels // 2, in_channels, 1, padding=0, bias=False)

            self.convy1 = nn.Conv2d(in_channels // 2, in_channels, 1, padding=0, bias=False)
            self.convy2 = nn.Conv2d(in_channels // 2, in_channels, 1, padding=0, bias=False)

        self.conv_atten1 = nn.Conv2d(in_channels, 1, 3, padding=1, bias=False)
        self.conv_atten2 = nn.Conv2d(in_channels, 1, 3, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels*3,
                               in_channels*3 // 8,
                               3,
                               padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels*3 // 8)
        self.relu1 = nonlinearity(inplace=True)

        self.deconv2 = nn.Upsample(scale_factor=2)

        self.norm2 = nn.BatchNorm2d(in_channels*3 // 8)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels*3 // 8,
                               n_filters,
                               3,
                               padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)


    def forward(self, chg, x, y):
        # filters = [64, 128, 256, 512]
        # chg:512, x/y: 256
        x1 = self.convx1(x)
        x2 = self.convx2(x)

        y1 = self.convy1(y)
        y2 = self.convy2(y)

        atten1 = self.sigmoid(self.conv_atten1(chg * x1))
        atten2 = self.sigmoid(self.conv_atten2(chg * y1))

        chg_x = atten1 * x2
        chg_y = atten2 * y2

        chg_cat = torch.cat([chg, chg_x, chg_y], dim=1)

        out = self.conv1(chg_cat)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.deconv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu3(out)
        return out

# m = ChgDecoderBlock(in_channels=512, n_filters=256)
# a = torch.rand([4,512,64,64])
# b = torch.rand([4,256,64,64])
# c = torch.rand([4,256,64,64])
# out = m(a,b,c)
# print(out.shape)



class chg_decoder1(nn.Module):
    def __init__(self, num_classes, is_deconv=False, decoder_kernel_size=3):
        super(chg_decoder1, self).__init__()
        filters = [64, 128, 256, 512]

        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)

        self.decoder4 = ChgDecoderBlock1(in_channels=filters[3], n_filters=filters[2])
        self.decoder3 = ChgDecoderBlock1(in_channels=filters[2], n_filters=filters[1])
        self.decoder2 = ChgDecoderBlock1(in_channels=filters[1], n_filters=filters[0])
        self.decoder1 = ChgDecoderBlock1(in_channels=filters[0], n_filters=filters[0], if_last_block=True)

        self.finalconv = nn.Sequential(nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, num_classes, 1))



    def forward(self, enc1_out, enc2_out):
        # enc_out1  [e4, e3, e2, e1, x]    size    [8,   16,   32,   64,   128]
        # enc_out2  [e4, e3, e2, e1, x]    channel [512, 256,  128,  64,   64 ]

        center = self.center(torch.abs(enc1_out[0]-enc2_out[0])) # (16, 16)
        d4 = self.decoder4(center, enc1_out[1], enc2_out[1])     # (32, 32)
        d3 = self.decoder3(d4, enc1_out[2], enc2_out[2])         # (64, 64)
        d2 = self.decoder2(d3, enc1_out[3], enc2_out[3])         # (128, 128)
        d1 = self.decoder1(d2, enc1_out[4], enc2_out[4])         # (256, 256)
        out = self.finalconv(d1)

        return out, d1

# m = chg_decoder2(num_classes=2)
# a = torch.rand([4,512,8,8])
# b = torch.rand([4,256,16,16])
# c = torch.rand([4,128,32,32])
# d = torch.rand([4,64,64,64])
# e = torch.rand([4,64,128,128])
# enc1 = [a,b,c,d,e]
# enc2 = [a,b,c,d,e]
# out, d1 = m(enc1, enc2)
# print(out.shape, d1.shape)




class seg_decoder(nn.Module):
    def __init__(self, num_classes, is_deconv=False, decoder_kernel_size=3):
        super(seg_decoder, self).__init__()

        filters = [64, 128, 256, 512]

        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        self.finalconv = nn.Sequential(nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, num_classes, 1))

    def forward(self, enc_out):
        # enc_out  [e4, e3, e2, e1, x]
        center = self.center(enc_out[0])
        d4 = self.decoder4(torch.cat([center, enc_out[1]], 1))
        d3 = self.decoder3(torch.cat([d4, enc_out[2]], 1))
        d2 = self.decoder2(torch.cat([d3, enc_out[3]], 1))
        d1 = self.decoder1(torch.cat([d2, enc_out[4]], 1))
        out = self.finalconv(d1)
        return out, d1


class encoder(nn.Module):
    def __init__(self,num_channels=3):
        super().__init__()

        resnet = resnet34(pretrained=True)
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        return [e4, e3, e2, e1, x]         # size [8,16,32,64,128]



class ChgSegNet_V2(nn.Module):

    def __init__(self, num_classes):
        super(ChgSegNet_V2, self).__init__()

        self.encoder = encoder(num_channels=3)
        self.SegDec1 = seg_decoder(num_classes=num_classes)
        self.SegDec2 = seg_decoder(num_classes=num_classes)
        self.ChgDec = chg_decoder1(num_classes=num_classes)

        self.conv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, num_classes, 1))

    def forward(self, x1, x2):
        enc_out1 = self.encoder(x1)
        enc_out2 = self.encoder(x2)
        Seg1, _ = self.SegDec1(enc_out1)
        Seg2, f2 = self.SegDec2(enc_out2)                   # f2 / fc  [B, 64, 256, 256]
        Chg, fc = self.ChgDec(enc_out1, enc_out2)

        Seg1_ = self.conv(torch.abs(f2 - fc))

        return Seg1, Seg2, Chg, Seg1_

# a = torch.rand([2,3,128,128])
# b = torch.rand([2,3,128,128])
# net = ChgSegNet_V2(2)
# Seg1, Seg2, chg, Seg1_ = net(a,b)
# print(Seg1.shape, Seg2.shape, chg.shape, Seg1_.shape)




class ChgNet_V2(nn.Module):
    """
    ChgNet_V2: only change branch on basis of ChgSegNet_V2
    """
    def __init__(self, num_classes):
        super(ChgNet_V2, self).__init__()

        self.encoder = encoder(num_channels=3)
        self.ChgDec = chg_decoder1(num_classes=num_classes)

    def forward(self, x1, x2):
        enc_out1 = self.encoder(x1)
        enc_out2 = self.encoder(x2)
        Chg, _ = self.ChgDec(enc_out1, enc_out2)

        return Chg, Chg, Chg, Chg

# a = torch.rand([2,3,128,128])
# b = torch.rand([2,3,128,128])
# net = ChgNet_V2(2)
# chg = net(a,b)
# print(chg[0].shape)


