import os
import json
import torch
import data_utils
from sparselinear import SparseLinear
import torch.nn as nn
from modeling_finetune import MLP

# from learning_concept_layer import Sparse_attention
class CBM_model(torch.nn.Module):
    def __init__(self, backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda",args=None):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device,args)
        model.eval()
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        
        if args.use_mlp:
            self.proj_layer = ModelOracleCtoY(n_class_attr=2, input_dim=W_c['linear.weight'].shape[1], num_classes=W_c['linear2.weight'].shape[0])
            self.proj_layer.load_state_dict(W_c)
            self.proj_layer = self.proj_layer.cuda()
        else:
            self.proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False).to(device)
            self.proj_layer.load_state_dict({"weight":W_c})
            
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
    def get_feature(self,x):
        backbone_feat = self.backbone(x)
        backbone_feat = torch.flatten(backbone_feat, 1)
        x = self.proj_layer(backbone_feat)
        proj_c = (x-self.proj_mean)/self.proj_std
        final = self.final(proj_c)
        
        return backbone_feat,proj_c,final
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        return x, proj_c
class CBM_model_joint(torch.nn.Module):
    def __init__(self, backbone_name, s_W_c,t_W_c, W_g, b_g, proj_mean, proj_std, device="cuda",args=None):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device,args)
        model.eval()
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        st_W_c = torch.cat([s_W_c,t_W_c],dim=0)
            
        self.proj_layer = torch.nn.Linear(in_features=st_W_c.shape[1], out_features=st_W_c.shape[0], bias=False).to(device)
        self.proj_layer.load_state_dict({"weight":st_W_c})
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
    def get_feature(self,x):
        backbone_feat = self.backbone(x)
        backbone_feat = torch.flatten(backbone_feat, 1)
        x = self.proj_layer(backbone_feat)
        proj_c = (x-self.proj_mean)/self.proj_std
        final = self.final(proj_c)
        
        return backbone_feat,proj_c,final
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        return x, proj_c
class CBM_model_triple(torch.nn.Module):
    def __init__(self, backbone_name, s_W_c,t_W_c,p_W_c, W_g, b_g, proj_mean, proj_std, device="cuda",args=None):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device,args)
        model.eval()
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        # stp_W_c = torch.cat([s_W_c,t_W_c,p_W_c],dim=0)
            
        self.s_proj_layer = torch.nn.Linear(in_features=s_W_c.shape[1], out_features=s_W_c.shape[0], bias=False).to(device)
        self.s_proj_layer.load_state_dict({"weight":s_W_c})
        self.t_proj_layer = torch.nn.Linear(in_features=t_W_c.shape[1], out_features=t_W_c.shape[0], bias=False).to(device)
        self.t_proj_layer.load_state_dict({"weight":t_W_c})
        self.p_proj_layer = torch.nn.Linear(in_features=p_W_c.shape[1], out_features=p_W_c.shape[0], bias=False).to(device)
        self.p_proj_layer.load_state_dict({"weight":p_W_c})
        self.s_proj_mean = proj_mean[0]
        self.s_proj_std = proj_std[0]
        self.t_proj_mean = proj_mean[1]
        self.t_proj_std = proj_std[1]
        self.p_proj_mean = proj_mean[2]
        self.p_proj_std = proj_std[2]
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
    def get_feature(self,x):
        backbone_feat = self.backbone(x)
        backbone_feat = torch.flatten(backbone_feat, 1)
        s_x = self.s_proj_layer(backbone_feat)
        s_proj_c = (s_x-self.s_proj_mean)/self.s_proj_std
        
        t_x = self.t_proj_layer(backbone_feat)
        t_proj_c = (t_x-self.t_proj_mean)/self.t_proj_std
        
        p_x = self.p_proj_layer(backbone_feat)
        p_proj_c = (p_x-self.p_proj_mean)/self.p_proj_std
        
        proj_c = torch.cat([s_proj_c,t_proj_c,p_proj_c],dim=1)
        final = self.final(proj_c)
        
        return backbone_feat,proj_c,final
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        # x = self.proj_layer(x)
        s_x = self.s_proj_layer(x)
        s_proj_c = (s_x-self.s_proj_mean)/self.s_proj_std
        
        t_x = self.t_proj_layer(x)
        t_proj_c = (t_x-self.t_proj_mean)/self.t_proj_std
        
        p_x = self.p_proj_layer(x)
        p_proj_c = (p_x-self.p_proj_mean)/self.p_proj_std
        
        proj_c = torch.cat([s_proj_c,t_proj_c,p_proj_c],dim=1)
        x = self.final(proj_c)
        return x, proj_c
class CBM_model_serial(torch.nn.Module):
    def __init__(self, backbone_name, pre_W_c, post_W_c, W_g, b_g, pre_mean,pre_std,post_mean, post_std, device="cuda",args=None):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device,args)
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
            
        self.pre_proj_layer = torch.nn.Linear(in_features=pre_W_c.shape[1], out_features=pre_W_c.shape[0], bias=False).to(device)
        self.pre_proj_layer.load_state_dict({"weight":pre_W_c})

        self.post_proj_layer = torch.nn.Linear(in_features=post_W_c.shape[1], out_features=post_W_c.shape[0], bias=False).to(device)
        self.post_proj_layer.load_state_dict({"weight":post_W_c})
            
        self.pre_proj_mean = pre_mean
        self.pre_proj_std = pre_std
        
        self.post_proj_mean = post_mean
        self.post_proj_std = post_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.pre_proj_layer(x)
        pre_proj_c = (x-self.pre_proj_mean)/self.pre_proj_std
        
        post_proj_c = self.post_proj_layer(pre_proj_c)
        post_proj_c = (post_proj_c-self.post_proj_mean)/self.post_proj_std
        x = self.final(post_proj_c)
        return x, pre_proj_c,post_proj_c
class standard_model(torch.nn.Module):
    def __init__(self, backbone_name, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device)
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
            
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        return x, proj_c

class CBM_model_attention(torch.nn.Module):
    def __init__(self, backbone_name, W_c, W, proj_mean, proj_std, device="cuda",args=None, d_model=None, n_classes=None,):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device,args)
        model.eval()
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        self.sparse_linear_attention = Sparse_attention(d_model,1,None, d_model,n_classes).to(device)
        self.sparse_linear_attention.load_state_dict(W)
        self.sparse_linear_attention.eval()
        
        self.proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False).to(device)
        self.proj_layer.load_state_dict({"weight":W_c})
            
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
    def get_feature(self,x):
        backbone_feat = self.backbone(x)
        backbone_feat = torch.flatten(backbone_feat, 1)
        x = self.proj_layer(backbone_feat)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.sparse_linear_attention(proj_c)
        
        return x,proj_c
    def forward(self, x):
        num_t = x.shape[0]
        features = []
        for i in range(num_t):
            feat = self.backbone(x[i])
            feat = torch.flatten(feat, 1)
            feat = self.proj_layer(feat)
            proj_c = (feat-self.proj_mean)/self.proj_std
            features.append(proj_c)
        proj_c = torch.cat(features,dim=0)#
        x,sub_act = self.sparse_linear_attention(proj_c.unsqueeze(0))
        return x, proj_c,sub_act
def load_cbm_two_stream_attention(load_dir, device,argument):
    print('**********Load Spatio model***************')
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)
    W_c = torch.load(os.path.join(load_dir,'spatial' ,"W_c.pt"), map_location=device)
    W = torch.load(os.path.join(load_dir,'spatial', "W.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir,'spatial', "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir,'spatial', "proj_std.pt"), map_location=device)
    s_model = CBM_model_attention(args['backbone'], W_c, W, proj_mean, proj_std, device,argument,proj_mean.shape[1],args['nb_classes'])
    print('**********Load Temporal model***************')
    W_c = torch.load(os.path.join(load_dir,'temporal' ,"W_c.pt"), map_location=device)
    W = torch.load(os.path.join(load_dir,'temporal', "W.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir,'temporal', "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir,'temporal', "proj_std.pt"), map_location=device)
    t_model = CBM_model_attention(args['backbone'], W_c, W, proj_mean, proj_std, device,argument,proj_mean.shape[1],args['nb_classes'])
    return s_model,t_model
def load_cbm(load_dir, device,argument):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    W_c = torch.load(os.path.join(load_dir ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = CBM_model(args['backbone'], W_c, W_g, b_g, proj_mean, proj_std, device,argument)
    return model
def load_cbm_joint(load_dir, device,argument):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    s_W_c = torch.load(os.path.join(load_dir,'spatial',"W_c.pt"), map_location=device)
    t_W_c = torch.load(os.path.join(load_dir,'temporal',"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir,'spatio_temporal', "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,'spatio_temporal', "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir,'spatio_temporal', "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir,'spatio_temporal', "proj_std.pt"), map_location=device)

    model = CBM_model_joint(args['backbone'], s_W_c,t_W_c, W_g, b_g, proj_mean, proj_std, device,argument)
    return model,model
def load_cbm_triple(load_dir, device,argument):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    s_W_c = torch.load(os.path.join(load_dir,'spatial',"W_c.pt"), map_location=device)
    s_proj_mean = torch.load(os.path.join(load_dir,'spatial', "proj_mean.pt"), map_location=device)
    s_proj_std = torch.load(os.path.join(load_dir,'spatial', "proj_std.pt"), map_location=device)
    
    t_W_c = torch.load(os.path.join(load_dir,'temporal',"W_c.pt"), map_location=device)
    t_proj_mean = torch.load(os.path.join(load_dir,'temporal', "proj_mean.pt"), map_location=device)
    t_proj_std = torch.load(os.path.join(load_dir,'temporal', "proj_std.pt"), map_location=device)
    
    p_W_c = torch.load(os.path.join(load_dir,'place',"W_c.pt"), map_location=device)
    p_proj_mean = torch.load(os.path.join(load_dir,'place', "proj_mean.pt"), map_location=device)
    p_proj_std = torch.load(os.path.join(load_dir,'place', "proj_std.pt"), map_location=device)
    
    W_g = torch.load(os.path.join(load_dir,'spatio_temporal_place', "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,'spatio_temporal_place', "b_g.pt"), map_location=device)


    model = CBM_model_triple(args['backbone'], s_W_c,t_W_c,p_W_c, W_g, b_g, [s_proj_mean,t_proj_mean,p_proj_mean], [s_proj_std,t_proj_std,p_proj_std], device,argument)
    return model,model
def load_cbm_serial(load_dir, device,argument):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    s_W_c = torch.load(os.path.join(load_dir,"spatial","W_c.pt"), map_location=device)
    t_W_c = torch.load(os.path.join(load_dir,"temporal","W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir,"classification", "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,"classification", "b_g.pt"), map_location=device)

    pre_proj_mean = torch.load(os.path.join(load_dir,'temporal' ,"proj_mean.pt"), map_location=device)
    pre_proj_std = torch.load(os.path.join(load_dir,'temporal', "proj_std.pt"), map_location=device)
    post_proj_mean = torch.load(os.path.join(load_dir,'classification' ,"proj_mean.pt"), map_location=device)
    post_proj_std = torch.load(os.path.join(load_dir,'classification', "proj_std.pt"), map_location=device)

    model = CBM_model_serial(args['backbone'], s_W_c,t_W_c, W_g, b_g,pre_proj_mean,pre_proj_std, post_proj_mean, post_proj_std, device,argument)
    return model
def load_cbm_two_stream(load_dir, device,argument):
    print('**********Load Spatio model***************')
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)
    W_c = torch.load(os.path.join(load_dir,'spatial' ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir,'spatial', "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,'spatial', "b_g.pt"), map_location=device)
    proj_mean = torch.load(os.path.join(load_dir,'spatial', "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir,'spatial', "proj_std.pt"), map_location=device)
    s_model = CBM_model(args['backbone'], W_c, W_g, b_g, proj_mean, proj_std, device,argument)
    print('**********Load Temporal model***************')
    W_c = torch.load(os.path.join(load_dir,'temporal' ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir,'temporal', "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,'temporal', "b_g.pt"), map_location=device)
    proj_mean = torch.load(os.path.join(load_dir,'temporal', "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir,'temporal', "proj_std.pt"), map_location=device)
    t_model = CBM_model(args['backbone'], W_c, W_g, b_g, proj_mean, proj_std, device,argument)
    return s_model,t_model
def load_std(load_dir, device):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = standard_model(args['backbone'], W_g, b_g, proj_mean, proj_std, device)
    model.eval()
    return model

def ModelOracleCtoY(n_class_attr, input_dim, num_classes):
    # X -> C part is separate, this is only the C -> Y part
    import math
    expand_dim = int(math.sqrt(input_dim * num_classes))
    if n_class_attr == 3:
        model = MLP(input_dim=input_dim * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model = MLP(input_dim=input_dim, num_classes=num_classes, expand_dim=expand_dim)
    return model

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int=1, n_head: int=1, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        # self.ln_1 = LayerNorm(d_model)
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        # self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, x2:torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        return self.attn(x, x2, x2, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor,x2:torch.Tensor):
        x = self.attention(x,x2)+x
        # x = x + self.mlp((x))
        return x
    
class Sparse_attention(nn.Module):
    def __init__(self, d_model: int=1, n_head: int=1, attn_mask: torch.Tensor = None, linear_in=1, linear_out=1):
        super().__init__()
        
        self.attention = ResidualAttentionBlock(d_model,n_head,attn_mask)
        self.spatial_token = nn.Parameter(torch.randn(1,linear_in))
        self.sparse_linear = SparseLinear(linear_in,linear_out,bias=True,sparsity=0.9,alpha=0.1)
        
    def forward(self,x):
        # x1 = x1.unsqueeze(-1)
        # x2 = x2.unsqueeze(-1)
        # spatial_token = self.spatial_token.expand(1,x1.shape[0],-1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # x2 = x2.permute(1, 0, 2)  # NLD -> LND
        
        x= self.attention(x,x)
        x = x.permute(1, 0, 2)  #NLD
        # final_feat=x.mean(1)
        x = self.sparse_linear(x)
        x = x.mean(1)
        return x,None#final_feat
    def forward_feat(self,x):
    
        spatial_token = self.spatial_token.expand(1,x.shape[0],-1)
        x1 = x.permute(1, 0, 2)  # NLD -> LND
        # x2 = x2.permute(1, 0, 2)  # NLD -> LND

        sub_act= self.attention(spatial_token,x1).permute(1,0,2).squeeze(1)#N,1,D
        mean_act = x[:,2,:]#.mean(1)  #NLD
        # final_feat=x.squeeze(1)
        x = self.sparse_linear(mean_act)
        # sub_x = self.sparse_linear(sub_act)
        final = x#+sub_x*0.1
        return final,sub_act
    def forward_feat_ensemble(self,x):
    
        spatial_token = self.spatial_token.expand(1,x.shape[0],-1)
        x1 = x.permute(1, 0, 2)  # NLD -> LND
        # x2 = x2.permute(1, 0, 2)  # NLD -> LND

        sub_act= self.attention(spatial_token,x1).permute(1,0,2).squeeze(1)#N,1,D
        # mean_act = x.mean(1)  #NLD
        # final_feat=x.squeeze(1)
        x = self.sparse_linear(x)
        sub_x = self.sparse_linear(sub_act)
        final = x.mean(1)+(0.1*sub_x)
        return final,sub_act
    # def forward_feat_self(self,x1):
    
    #     x=x1[]
    #     return x,final_feat