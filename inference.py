import torch
from model_loader import load_model
from utils import extract_mouth_frames, transcribe_video_frames
def run_inference(lrs3_path,vox_path,gt=None):
    model,cfg,task=load_model()
    lrs3_frames=extract_mouth_frames(lrs3_path)
    vox_frames=extract_mouth_frames(vox_path)
    lrs3_pred=transcribe_video_frames(model,task,lrs3_frames)
    vox_pred=transcribe_video_frames(model,task,vox_frames)
    print("LRS3 sample transcription:",lrs3_pred)
    if gt:
        import jiwer
        wer_val=jiwer.wer(gt,lrs3_pred)
        print("LRS3 sample WER:",wer_val)
    print("VoxCeleb2 sample transcription:",vox_pred)
if __name__=="__main__":
    import sys
    if len(sys.argv)>=3:
        gt=None
        if len(sys.argv)==4:
            gt=sys.argv[3]
        run_inference(sys.argv[1],sys.argv[2],gt)

