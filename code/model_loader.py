import torch
import fairseq
import avhubert.hubert_asr, avhubert.hubert, avhubert.utils
def load_model():
    ckpt="/content/data/finetune-model.pt"
    models,cfg,task=fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
    model=models[0].cuda().eval()
    return model,cfg,task
if __name__=="__main__":
    m,c,t=load_model()
    print("Loaded AV-HuBERT model - input modality:",c.task.input_modality,"output vocab size:",len(t.target_dictionary))

