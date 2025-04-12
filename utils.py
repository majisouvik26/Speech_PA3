import cv2
import numpy as np
import torch
def extract_mouth_frames(video_path,target_size=88):
    cap=cv2.VideoCapture(video_path)
    frames=[]
    face_bbox=None
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if face_bbox is None:
            faces=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml").detectMultiScale(gray,1.3,4)
            if len(faces)>0:
                face_bbox=max(faces,key=lambda r:r[2]*r[3])
        if face_bbox is None:
            continue
        x,y,w,h=face_bbox
        mx,my=x+int(0.25*w),y+int(0.6*h)
        mw,mh=int(0.5*w),int(0.4*h)
        mouth_img=gray[my:my+mh,mx:mx+mw]
        if mouth_img.size==0:
            mouth_img=gray[y:y+h,x:x+w]
        mouth_img=cv2.resize(mouth_img,(target_size,target_size))
        frames.append(mouth_img)
    cap.release()
    return np.stack(frames)
def transcribe_video_frames(model,task,frames):
    frames_tensor=torch.from_numpy(frames).unsqueeze(0).unsqueeze(2).float().cuda()
    with torch.no_grad():
        enc_out=model.encoder(source=frames_tensor,padding_mask=None)
        bos=task.target_dictionary.bos()
        eos=task.target_dictionary.eos()
        out_idx=[]
        prev_tokens=torch.tensor([[bos]],dtype=torch.long).cuda()
        for _ in range(100):
            decoder_out=model.decoder(prev_output_tokens=prev_tokens,encoder_out=enc_out)
            logits=decoder_out[0] if isinstance(decoder_out,tuple) else decoder_out
            next_token=logits[0,-1].argmax().item()
            if next_token==eos:
                break
            out_idx.append(next_token)
            prev_tokens=torch.cat([prev_tokens,torch.tensor([[next_token]]).cuda()],dim=1)
        hypothesis=task.target_dictionary.string(torch.tensor(out_idx))
    return hypothesis
def predict_frames(model,task,frames):
    frames_tensor=torch.from_numpy(frames).unsqueeze(0).unsqueeze(2).float().cuda()
    with torch.no_grad():
        enc_out=model.encoder(source=frames_tensor,padding_mask=None)
        prev_tokens=torch.tensor([[task.target_dictionary.bos()]],device='cuda')
        out_idx=[]
        for _ in range(20):
            logits,_=model.decoder(prev_output_tokens=prev_tokens,encoder_out=enc_out)
            next_token=logits[0,-1].argmax().item()
            if next_token==task.target_dictionary.eos():
                break
            out_idx.append(next_token)
            prev_tokens=torch.cat([prev_tokens,torch.tensor([[next_token]],device='cuda')],dim=1)
    return task.target_dictionary.string(torch.tensor(out_idx))

