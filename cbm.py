import os
import json
import torch
import data_utils

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

