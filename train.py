import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import CremadLipreadingDataset
from model_loader import load_model
import glob
import os
def LoRALinear(old_linear,rank=2,alpha=1.0):
    import torch.nn as nn
    import math
    lin=old_linear
    new=nn.Module()
    new.in_features=lin.in_features
    new.out_features=lin.out_features
    new.rank=rank
    new.alpha=alpha
    new.weight=nn.Parameter(lin.weight.data.clone(),requires_grad=False)
    new.bias=nn.Parameter(lin.bias.data.clone() if lin.bias is not None else torch.zeros(lin.out_features),requires_grad=False)
    new.A=nn.Parameter(torch.zeros(rank,lin.in_features))
    new.B=nn.Parameter(torch.zeros(lin.out_features,rank))
    nn.init.kaiming_uniform_(new.A,a=math.sqrt(5))
    nn.init.zeros_(new.B)
    def forward(x):
        o=x@new.weight.t()
        if new.bias is not None:
            o=o+new.bias
        l=(x@new.A.t())@new.B.t()
        return o+new.alpha*l
    new.forward=forward
    return new
def inject_lora(model):
    import torch.nn as nn
    linear_names=[]
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear):
            linear_names.append(name)
    for name in linear_names:
        *parent_names,attr=name.split(".")
        parent=model
        for pn in parent_names:
            parent=getattr(parent,pn)
        setattr(parent,attr,LoRALinear(getattr(parent,attr),rank=2,alpha=1.0))
    for pname,param in model.named_parameters():
        if "A" in pname or "B" in pname:
            param.requires_grad=True
        else:
            param.requires_grad=False
    t=sum(p.numel() for p in model.parameters() if p.requires_grad)
    T=sum(p.numel() for p in model.parameters())
    print("Trainable params:",t,"Total params:",T)
def train_model(crema_dir,task,model,epochs=1):
    video_paths=glob.glob(os.path.join(crema_dir,"VideoFlash/*/*.flv"))
    sentence_map={"IEO":"It's eleven o'clock.","TIE":"That is exactly what happened.","IOM":"I'm on my way to the meeting.","IWW":"I wonder what this is about.","TAI":"The airplane is almost full.","MTI":"Maybe tomorrow it will be cold.","IWL":"I would like a new alarm clock.","ITH":"I think I have a doctor's appointment.","DFA":"Don't forget a jacket.","ITS":"I think I've seen this before.","TSI":"The surface is slick.","WSI":"We'll stop in a couple of minutes."}
    actor_set={int(os.path.basename(p).split("_")[0]) for p in video_paths}
    sorted_actors=sorted(actor_set)
    split_idx=int(0.7*len(sorted_actors))
    train_actors=set(sorted_actors[:split_idx])
    train_list=[]
    for vp in video_paths:
        actor=int(os.path.basename(vp).split("_")[0])
        sent_code=os.path.basename(vp).split("_")[1]
        if actor in train_actors:
            train_list.append((vp,sentence_map.get(sent_code,"")))
    dataset=CremadLipreadingDataset(train_list,task)
    loader=DataLoader(dataset,batch_size=1,shuffle=True)
    optimizer=optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    model.train()
    epoch_loss=0.0
    for epoch in range(epochs):
        for i,(frames_tensor,target_tensor) in enumerate(loader):
            frames_tensor=frames_tensor.cuda()
            target_tensor=target_tensor.cuda()
            optimizer.zero_grad()
            enc_out=model.encoder(source=frames_tensor.unsqueeze(0),padding_mask=None)
            prev_tokens=target_tensor[:,:-1]
            logits,extra=model.decoder(prev_output_tokens=prev_tokens,encoder_out=enc_out)
            lprobs=F.log_softmax(logits,dim=-1)
            loss=F.nll_loss(lprobs.view(-1,lprobs.size(-1)),target_tensor[:,1:].reshape(-1),ignore_index=task.target_dictionary.pad())
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            if i%100==0:
                print("Step",i,"Loss",loss.item())
        print("Epoch",epoch,"Avg loss",epoch_loss/len(loader))
if __name__=="__main__":
    model,cfg,task=load_model()
    inject_lora(model)
    train_model("CREMA-D",task,model,epochs=1)

