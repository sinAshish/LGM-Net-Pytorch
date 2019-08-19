import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MetaNetwork(nn.Module):
    def __init__(self):
        super(MetaNetwork, self).__init__()
        self.reuse = False
        self.out_size = 64
        self.ksize = [3, 3] 
    
    def forward(self, inputs, context):
        in_shape = inputs.size()
        c_dim = list(context.size())[-1]
        # split the context into mean and variance predicted by task context encoder
        z_dim = c_dim //2 
        c_mu = context[:z_dim]
        c_log_var = context[z_dim: ]

        if len(in_shape) == 4:
            is_CNN= True
        else:
            is_CNN = False
        
        if is_CNN:
            assert self.ksize[0] == self.ksize[1]
            f_size = self.ksize[0]
            in_size = in_shape[-1]

            M = f_size * f_size * in_size
            N= self.out_size
            wt_shape = [M+1, N]

        else:
            M = in_shape[-1]
            N =self.out_size
            wt_shape = [M+1, N]
        
        z_signal = torch.randn(1, z_dim)
        z_c_mu = c_mu.unsqueeze(0)
        z_c_log_var = c_log_var.unsqueeze(0)
        z_c = z_c_mu + torch.exp(z_c_log_var/2)*z_signal

        w1 = torch.empty(z_dim, (M+1)*N)
        nn.init.xavier_uniform_(w1, gain=nn.init.calculate_gain('relu'))
        b1 = torch.empty((M+1)*N)
        nn.init.constant_(b1, 0.0)
        final = torch.mm(z_c, w1)+ b1
        meta_wts = final[0, :M*N]
        meta_bias = final[0, M*N:]

        if is_CNN:
            meta_wts = torch.transpose(meta_wts.view(self.out_size, in_size, f_size, f_size), 0, 1)
        else:
            meta_wts = torch.transpose(meta_wts.view(self.out_size, M))

        if is_CNN:
            meta_wts = F.normalize(meta_wts, p=2, dim=[0, 1, 2])
        else:
            meta_wts = F.normalize(meta_wts, p=2, dim=0)
        
        return meta_wts, meta_bias

class MetaConvolution(nn.Module):
    def __init__(self):
        super(MetaConvolution, self).__init__()
        self.metanet = MetaNetwork()
        self.conv = nn.Conv2d(128, 128,1, stride=1, padding=1)
        #self.conv.data.fill_()
    def forward(self, inputs, context, filters, ksize, training= False):
        meta_conv_w, meta_conv_b = self.metanet(inputs, context)
        out =  self.conv(inputs)#+ meta_conv_b
        
        return out, meta_conv_w, meta_conv_b

class TaskTransformer(nn.Module):
    def __init__(self):
        super(TaskTransformer, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
    
    def forward(self, task_embedding):
        x = self.pool1(self.norm1(F.relu(self.conv1(task_embedding))))
        x = self.pool2(self.norm2(F.relu(self.conv2(x))))
        x = self.conv3(x)

        return x.sum(dim=[1, 2])

class TaskContextEncoder(nn.Module):
    def __init__(self, method='mean'):
        super(TaskContextEncoder, self).__init__()
        self.tasktrans = TaskTransformer()
        self.method = method
        
    def forward(self, x):
        bc, kn, w, h, c = x.size()
        x = x.view(bc*kn, w, h, c)
        if self.method == 'mean':
            x = self.tasktrans(x)
            x = x.view(bc, kn, -1)
            x = x.sum(dim=1)
        
        elif self.method == 'bilstm':
            #Todo
            pass
        
        else:
            raise TypeError('No such Methods, please use mean')
        return x

class Classifier(nn.Module):
    def __init__(self):
        self.meta_conv1 = MetaConvolution()
        self.meta_conv2 = MetaConvolution()

    def forward(self, image_embedding, task_context):
        """
        Runs the CNN producing the embeddings and the gradients.
        """
        m_conv1, m_conv1_w, m_conv1_b = self.meta_conv1(image_embedding, task_context, 64, 3)
        m_conv1 = F.relu(m_conv1)
        m_conv1 = F.max_pool2d(m_conv1, (2, 2))

        m_conv2, m_conv2_w, m_conv2_b = self.meta_conv2(m_conv1, task_context, 64, 3)
        m_conv2 = m_conv2.view(-1, 1)

        gen_wts = [m_conv1_w, m_conv1_b, m_conv2_w, m_conv2_b]

        return m_conv2, gen_wts

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.gconv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.2)
        self.gconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gconv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.gconv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm4 = nn.BatchNorm2d(64)
    
    def forward(self, support_target_images):
        bs, kn, w, h, c = support_target_images.size()
        support_target_images = support_target_images.view(bs*kn, w, h, c)
        x = self.gconv1(support_target_images)
        x = self.pool1(F.relu(self.norm1(self.gconv2(x))))
        x = self.pool2(F.relu(self.norm2(self.gconv2(x))))
        x = self.pool3(F.relu(self.norm3(self.gconv3(x))))
        x = F.relu(self.norm4(self.gconv4(x)))
        bskn, we, he, ce = x.size()

        embeddings = x.view(bs ,kn, we, he, ce)

        return embeddings

