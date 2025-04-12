> # AV-HuBERT Lipreading with DoRA Fine-Tuning and WER Evaluation  
>
> This project demonstrates how to perform inference using the pretrained AV-HuBERT model on LRS3 and VoxCeleb2, fine-tune it on the CREMA-D dataset using DoRA (Low-Rank Adaptation), and evaluate transcription performance using Word Error Rate (WER).  
>
> ## Files  
>
> - **setup.sh**: Sets up the environment, clones the AV-HuBERT repository, installs dependencies (including jiwer), and downloads the pretrained model.  
> - **model_loader.py**: Loads the pretrained AV-HuBERT model.  
> - **utils.py**: Contains utility functions for video processing and inference.  
> - **dataset.py**: Implements the CREMA-D dataset for lipreading.  
> - **train.py**: Fine-tunes the model on CREMA-D using DoRA-based adaptation.  
> - **inference.py**: Runs inference on LRS3 and VoxCeleb2 sample videos. If a ground truth transcript is provided for the LRS3 sample (as an extra argument), it computes and prints the WER.  
> - **evaluate.py**: Evaluates the CREMA-D test set by computing the average WER.  
> - **main.py**: Entry point. Use mode "inference", "train", or "evaluate" via command-line arguments.  
> - **readme.md**: This file.  
>
> ## Requirements  
>
> Python, pip, and dependencies as specified in the AV-HuBERT repository. The setup script installs jiwer.  
>
> ## Setup  
>
> Run the following command to set up the environment:  
>
> ```bash  
> bash setup.sh  
> ```  
>
> ## Usage  
>
> **For inference** (provide paths for one LRS3 sample video and one VoxCeleb2 sample video; optionally provide a ground truth transcript for LRS3 to compute WER):  
>
> ```bash  
> python main.py --mode inference --lrs3 <path_to_lrs3_video> --vox <path_to_voxceleb_video> [--gt "<ground_truth_transcript>"]  
> ```  
>
> **For training** (fine-tune on the CREMA-D dataset; provide the path to the CREMA-D directory):  
>
> ```bash  
> python main.py --mode train --crema <path_to_CREMA-D_directory>  
> ```  
>
> **For evaluation** on the CREMA-D test set:  
>
> ```bash  
> python main.py --mode evaluate --crema <path_to_CREMA-D_directory>  
> ```  
