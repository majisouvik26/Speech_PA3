import argparse
import sys
from model_loader import load_model
from inference import run_inference
from train import train_model, inject_lora
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",required=True)
    parser.add_argument("--lrs3")
    parser.add_argument("--vox")
    parser.add_argument("--crema")
    parser.add_argument("--gt")
    args=parser.parse_args()
    if args.mode=="inference":
        if args.lrs3 is None or args.vox is None:
            sys.exit(1)
        run_inference(args.lrs3,args.vox,args.gt)
    elif args.mode=="train":
        if args.crema is None:
            sys.exit(1)
        model,cfg,task=load_model()
        inject_lora(model)
        train_model(args.crema,task,model,epochs=1)
    elif args.mode=="evaluate":
        if args.crema is None:
            sys.exit(1)
        import evaluate
        evaluate.evaluate_crema(args.crema)

