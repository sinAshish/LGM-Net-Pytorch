import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistanceNetwork(nn.Module):
    def __init__(self, metric = 'cosine'):
        super(DistanceNetwork, self).__init__()
        self.metric = metric
    
    def forward(self, support_set, input_image):
        if self.metric == 'cosine':
            #import pdb; pdb.set_trace()
            support_set = support_set.unsqueeze(1)
            input_image = input_image.unsqueeze(1)
            norm_s = F.normalize(support_set, p=2, dim=2)
            norm_i = F.normalize(input_image, p=2, dim=2)
            similarities = torch.mm(norm_s, norm_i).sum(dim=2)

        elif metric == 'euclidean':
            input_image = input_image.unsqueeze(1)
            similarities = -torch.square(support_set - input_image).sum(dim = 2)

        return similarities

class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()
        self.softmax = nn.Softmax()
    
    def forward(self, similarities, support_set_y):
        softmax_similarities = self.softmax(similarities)
        return softmax_similarities

class MetaNetwork(nn.Module):
    def __init__(self):
        super(MetaNetwork, self).__init__()
        self.reuse = False
        self.out_size = 64
        self.ksize = [3, 3] 
    
    def forward(self, inputs, context):
        in_shape = inputs.size()
        #import pdb; pdb.set_trace()
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
            in_size = in_shape[1]

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
        self.conv = nn.Conv2d(64, 128,1, stride=1, padding=1)
        #self.conv.data.fill_()
    def forward(self, inputs, context, filters, ksize, training= False):
        #import pdb; pdb.set_trace()
        meta_conv_w, meta_conv_b = self.metanet(inputs, context)
        out =  F.conv2d(inputs, meta_conv_w, meta_conv_b, 1, 1, 1)#+ meta_conv_b
        
        return out, meta_conv_w, meta_conv_b

class TaskTransformer(nn.Module):
    def __init__(self):
        super(TaskTransformer, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
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
        #import pdb; pdb.set_trace()
        bc, kn, c, w, h = x.size()
        x = x.contiguous().view(bc*kn, c, h, w)
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
        super(Classifier, self).__init__()
        self.meta_conv1 = MetaConvolution()
        self.meta_conv2 = MetaConvolution()

    def forward(self, image_embedding, task_context):
        """
        Runs the CNN producing the embeddings and the gradients.
        """
        #import pdb; pdb.set_trace()
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
        self.gconv1 = nn.Conv2d(3, 64, 3, 1, 1)
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
        #import pdb; pdb.set_trace()
        bs, kn,spc,  h, w,c = support_target_images.size()
        support_target_images = support_target_images.view(bs*kn*spc, c, h, w)
        #x = self.gconv1(support_target_images)
        x = self.pool1(F.relu(self.norm1(self.gconv1(support_target_images))))
        x = self.pool2(F.relu(self.norm2(self.gconv2(x))))
        x = self.pool3(F.relu(self.norm3(self.gconv3(x))))
        x = F.relu(self.norm4(self.gconv4(x)))
        bskn, ce, we, he = x.size()

        embeddings = x.view(bs ,kn*spc, ce, we, he)

        return embeddings

class MetaMatchingNetwork(nn.Module):
    def __init__(self,\
        num_classes_per_set=5,\
        num_samples_per_class =1 
        ):
        super(MetaMatchingNetwork, self).__init__()
        self.Classifier = Classifier()
        self.tce = TaskContextEncoder()
        self.extractor = Extractor()
        self.dn = DistanceNetwork()
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.softmax = nn.Softmax()

    def forward(self, support_set_images, support_set_labels, target_image, target_label):
        tensor_list = []
        #import pdb; pdb.set_trace()
        b, num_classes, spc = support_set_labels.size()

        support_set_labels_ = support_set_labels.view(b, num_classes * spc)
        import pdb; pdb.set_trace()
        #support = torch.FloatTensor(b, num_classes*spc)
        #support.zero_()

        support_set_labels_.scatter_(1, torch.tensor(support_set_labels_, dtype = torch.long), self.num_classes_per_set)

        b, num_classes, spc, h, w, c= support_set_images.size()
        support_set_images_ = support_set_images.view(b, num_classes * spc, h, w, c)

        #Zeroth step
        #Extrace feature embeddings
        target_image_ = target_image.unsqueeze(1)
        #merge support set and target set in order to share the feature extractros
        support_target_images = torch.cat([support_set_images_, target_image_], dim=1 )

        support_target_embeddings = self.extractor(support_set_images)
        #First step: generate task feature representations by using support set features
        task_contexts = self.tce(support_target_embeddings[:, :-1])

        #Second step: transform images via conditional meta task convolution
        trans_support_images_list = []
        trans_target_images_list = []
        task_gen_wts_list = []
        for i, (tc, ste) in enumerate(zip(torch.unbind(task_contexts), torch.unbind(support_target_embeddings))):
            #print ("____ In task instance ", i)
            #support task image embeddings for one task
            steb, gen_wts_list = self.Classifier(image_embedding = ste, task_context = tc)
            trans_support_images_list.append(steb[:-1])
            trans_target_images_list.append(steb[:-1])
            task_gen_wts_list.append(gen_wts_list)
        
        trans_support = torch.stack(trans_support_images_list, 0)
        trans_target = torch.stack(trans_target_images_list, 0)
        #import pdb; pdb.set_trace()
        similarities = F.cosine_similarity(trans_support, trans_target, dim=2)
        #similarities = self.dn(trans_support, trans_target)
        #Produce pdfs over the support set classes fro the target set image.
        softmax_similarities = self.softmax(similarities)
        softmax_similarities = softmax_similarities[:, :num_classes*spc]
        preds = (softmax_similarities * support_set_labels_).squeeze()

        if b == 1:
            #Reshape to avoid shape error
            preds = preds.view(b, preds.size()[-1])
        
        return preds, target_label