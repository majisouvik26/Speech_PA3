import os
import glob
import torch
from model_loader import load_model
from utils import extract_mouth_frames, predict_frames
import jiwer
def evaluate_crema(crema_dir):
    video_paths=glob.glob(os.path.join(crema_dir,"VideoFlash/*/*.flv"))
    sentence_map={"IEO":"It's eleven o'clock.","TIE":"That is exactly what happened.","IOM":"I'm on my way to the meeting.","IWW":"I wonder what this is about.","TAI":"The airplane is almost full.","MTI":"Maybe tomorrow it will be cold.","IWL":"I would like a new alarm clock.","ITH":"I think I have a doctor's appointment.","DFA":"Don't forget a jacket.","ITS":"I think I've seen this before.","TSI":"The surface is slick.","WSI":"We'll stop in a couple of minutes."}
    actor_set={int(os.path.basename(p).split("_")[0]) for p in video_paths}
    sorted_actors=sorted(actor_set)
    split_idx=int(0.7*len(sorted_actors))
    test_actors=set(sorted_actors[split_idx:])
    test_list=[]
    for vp in video_paths:
        actor=int(os.path.basename(vp).split("_")[0])
        sent_code=os.path.basename(vp).split("_")[1]
        if actor in test_actors:
            test_list.append((vp,sentence_map.get(sent_code,"")))
    model,cfg,task=load_model()
    model.eval()
    total_wer=0.0
    count=0
    for vp,gt in test_list:
        frames=extract_mouth_frames(vp)
        pred=predict_frames(model,task,frames)
        wer_val=jiwer.wer(gt,pred)
        total_wer+=wer_val
        count+=1
    if count>0:
        print("Average WER on CREMA-D test set:",total_wer/count)
if __name__=="__main__":
    import sys
    if len(sys.argv)==2:
        evaluate_crema(sys.argv[1])

