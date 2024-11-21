from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

chsize= 28
class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """

    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B * P * P,
                              kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                           kernel_size=K, stride=stride, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out #(14 * 14 * 544)

class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """

    def __init__(self, B=32, C=32, K=3, P=4, stride=2, iters=3,
                 coor_add=False, w_shared=False):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        self.return_routes = False
        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2 * math.pi))
        # params
        # Note that \beta_u and \beta_a are per capsule type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K * K * B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1) [144, 288, 1]
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """

        r = r * a_in #->144, 288, 1]

        # r -> [36, 288, 32]
        r = r / (r.sum(dim=2, keepdim=True) + eps)

        # rSum -> [36, 1, 32]
        r_sum = r.sum(dim=1, keepdim=True)

        coeff = r / (r_sum + eps)

        # the coeff -> [36, 288, 32]
        coeff = coeff.view(b, B, C, 1)

        # the coeff -> [36, 288, 32, 1]

        # The vector -> [36, 288, 32, 16]
        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu) ** 2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum

        a_out = self.sigmoid(self._lambda * (self.beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)
        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu) ** 2 / (2 * sigma_sq) \
                   - torch.log(sigma_sq.sqrt()) \
                   - 0.5 * self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
        r = self.softmax(ln_ap)

        #print('E step during the assigment : ', r.shape)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        #print('a_in in the em routing ', a_in.shape)
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        r = torch.cuda.FloatTensor(b, B, C).fill_(1. / C)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            # Skip the last eStep since we do not want to re-calculate the rr
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)
        # mu -> [36, 1, 32, 16]
        # a_out-> [36, 32]
        return mu, a_out, r, v

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape
        assert h == w
        assert c == B * (psize + 1)
        oh = ow = int(((h - K) / stride) + 1)  # moein - changed from: oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) \
                 for k_idx in range(0, K)] \
                for h_idx in range(0, h - K + 1, stride)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P * P

        x = x.view(b, B, 1, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)

        x = x.repeat(1, 1, C, 1, 1)

        v = torch.matmul(x, w)
        v = v.view(b, B, C, P * P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = torch.arange(h, dtype=torch.float32) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h * w * B, C, psize)
        return v

    def forward(self, x, activation_mask=None):
        b, h, w, c = x.shape
        if not self.w_shared:
            if activation_mask is not None:
                p_in = x[:, :, :, :self.B * self.psize].contiguous()
                a_in = x[:, :, :, self.B * self.psize:].contiguous()
                assert a_in.shape == activation_mask.shape
                a_in *= activation_mask
                x = torch.cat((p_in, a_in), dim=-1)

            # add patches
            x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B * self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B * self.psize:].contiguous()

            p_in = p_in.view(b * oh * ow, self.K * self.K * self.B, self.psize)
            a_in = a_in.view(b * oh * ow, self.K * self.K * self.B, 1)
            v = self.transform_view(p_in, self.weights, self.C, self.P)

            # em_routing
            # pose-> [36, 1, 32, 16]
            # a-> [36, 32]
            p_out, a_out1, route, vot = self.caps_em_routing(v, a_in, self.C, self.eps)

            p_out = p_out.view(b, oh, ow, self.C * self.psize)
            #print('p_out.shape ', p_out.shape)
            a_out = a_out1.view(b, oh, ow, self.C)
            #print('aout ', a_out.shape)
            out = torch.cat([p_out, a_out], dim=3)
            #print('out after th concat  ', out.shape)
        else:
            assert c == self.B * (self.psize + 1)
            assert 1 == self.K
            assert 1 == self.stride
            #print('The input activation X to the forward opertation before a_in and p_in', x.shape)
            p_in = x[:, :, :, :self.B * self.psize].contiguous()
            a_in = x[:, :, :, self.B * self.psize:].contiguous()

            if activation_mask is not None:
                assert a_in.shape[1:] == activation_mask.shape[1:]
                a_in *= activation_mask
                #print('a_in after a_in *= activation_mask ', a_in.shape)

            p_in = p_in.view(b, h * w * self.B, self.psize)
            a_in = a_in.view(b, h * w * self.B, 1)

            #print('p_in and a_in after view', p_in.shape, 'a_in ', a_in.shape)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)
            #print('the vector shape v transforma ', v.shape)

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            p_out, a_out1, route, vot = self.caps_em_routing(v, a_in, self.C, self.eps)
            #print(p_out.shape)
            p_out = p_out.squeeze(1) #([1, 1, 10, 16])
            a_out = a_out1.unsqueeze(-1) #[1, 512, 1])
            #print(a_in.shape)
            out = torch.cat([p_out, a_out], dim=2) #[1, 10, 17])
            #print(out.shape)

        if self.return_routes:
            return out, route, vot, a_out1, a_in
        else:
            return out


class CapsNet(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 28x28x1, the feature maps change as follows:
    1. ReLU Conv1
        (_, 1, 28, 28) -> 5x5 filters, 32 out channels, stride 2 with padding
        x -> (_, 32, 14, 14)
    2. PrimaryCaps
        (_, 32, 14, 14) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 14, 14, 32x4x4), activation: (_, 14, 14, 32)
    3. ConvCaps1
        (_, 14, 14, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 6, 6, 32x4x4), activation: (_, 6, 6, 32)
    4. ConvCaps2
        (_, 6, 6, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 4, 4, 32x4x4), activation: (_, 4, 4, 32)
    5. ClassCaps
        (_, 4, 4, 32x(4x4+1)) -> 1x1 conv, 10 out capsules
        x -> pose: (_, 10x4x4), activation: (_, 10)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """

    def __init__(self, A=32, B=32, C=32, D=32, E=10, K=3, P=4, iters=3):
        super(CapsNet, self).__init__()
        channels= 1
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=A,
                               kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                  momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
                                   coor_add=True, w_shared=True)

        self.activation_mask1 = None
        self.activation_mask2 = None
        self.activation_mask3 = None

        self.decoder = nn.Sequential(
            nn.Linear(E * P ** 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, channels * chsize * chsize),
            nn.Sigmoid()
        )#784 , 2304 , 4096, 4096

    def forward(self, x, recon=False, probe=False): #def forward(self, x, y, recon=True, probe=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # 1st activations probe
        x = self.primary_caps(x)
        # 2nd activations probe
        #print('The input activations to CC1 - the output from PC', x.shape)
        x = self.conv_caps1(x, self.activation_mask1)
        #print('The input activations to CC2 the output from CC1', x.shape)
        # 3rd activations probe
        x = self.conv_caps2(x, self.activation_mask2)
        #print('The input activations to Classcaps - CC2', x.shape)
        # 4th activations probe
        x = self.class_caps(x, self.activation_mask3)
        
        # 5th activations probe
        # split into pose and activations
        p = x[:, :, :-1].contiguous()

        a = x[:, :, -1:].contiguous().squeeze(-1)

        #print(' pose ', p.shape)
        #print(' pose ', p)
        #print('---------------------------------------')
        #print(' activation ', a.shape)
        #print(' activation ', a)

        
        if recon:
            y =0
            # feed pose into decoder for reconstruction
            #print('y ', y)
            target_pose = p[torch.arange(0, p.shape[0], device=x.device), y, :]
            #print('target_pose ', target_pose.shape)
            d_in = torch.zeros(target_pose.shape[0], target_pose.shape[1] * 10, device=x.device)


            for i, _y in enumerate(y):
                
                #print(i, i + 1, ' _y ', _y)
                d_in[i, _y * 16:(_y + 1) * 16] = target_pose[i:i + 1]
                #print(' i', i, ' _y * 16', _y * 16)
                #print(' i', i, ' (_y + 1) * 16',(_y + 1) * 16)

            
            recon = self.decoder(d_in)


            return a, recon
        else:
            return a, p

    def activations(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # 1st activations probe
        yield x
        x = self.primary_caps(x)
        # 2nd activations probe
        yield x
        x = self.conv_caps1(x)
        # 3rd activations probe
        yield x
        x = self.conv_caps2(x)
        # 4th activations probe
        yield x
        x = self.class_caps(x)
        # 5th activations probe
        yield x

    def routes(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # self.primary_caps.return_routes = True
        x = self.primary_caps(x)
        # 1st routes probe
        # yield routes
        # primary caps don't do routing
        self.conv_caps1.return_routes = True
        x, routes1, vot, a_out1, aIn1 = self.conv_caps1(x)
        self.conv_caps1.return_routes = False

        self.conv_caps2.return_routes = True
        x, routes2, vot2, a_out2, aIn2 = self.conv_caps2(x)
        self.conv_caps2.return_routes = False

        self.class_caps.return_routes = True
        x, routes3, vot3, a_out3, aIn3 = self.class_caps(x)
        self.class_caps.return_routes = False
        # 4th activations probe
        return routes1, routes2, routes3, vot, vot2, vot3, a_out1, a_out2, a_out3, aIn1, aIn2, aIn3

    def modify_activations(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # 1st activations probe
        # yield x
        # x = yield
        x = self.primary_caps(x)
        # 2nd activations probe
        activation_mask = yield
        x = self.conv_caps1(x, activation_mask)
        # 3rd activations probe
        activation_mask = yield
        x = self.conv_caps2(x, activation_mask)
        # 4th activations probe
        activation_mask = yield
        x = self.class_caps(x, activation_mask)
        p = x[:, :, :-1].contiguous()
        yield p


def capsules(**kwargs):
    """Constructs a CapsNet model.
    """
    model = CapsNet(**kwargs)
    return model


if __name__ == '__main__':
    model = capsules(E=10)
    print(model)
