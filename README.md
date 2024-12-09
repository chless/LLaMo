<p align="center">
  <h1 align="center">LLaMo: Large Language Model-based Molecular Graph Assistant</h1>
  
  <p align="center">Jinyoung Park, Minseong Bae, Dohwan Ko, Hyunwoo J. Kim.
  </p>

  <h3 align="center">
    <a href="https://www.arxiv.org/pdf/2411.00871" target='_blank'><img src="https://img.shields.io/badge/arXiv-2411.00871-b31b1b.svg"></a>
  </h3>

</p>
Official PyTorch implementation of the "LLaMo: Large Language Model-based Molecular Graph Assistant".
(NeurIPS 2024)


## TODO
- [x] Release the code.
- [ ] Refactoring code to incorporate the huggingface.
- [ ] Release the pre-trained huggingface model.


## Enviroment
To install requirements, run:
```bash
git clone https://github.com/mlvlab/LLaMo.git
cd LLaMo
conda create -n llamo python==3.9
conda activate llamo
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Preparation
### Pretrained graph encoder
We utilized the pre-trained graph encoder checkpoint from the [link](https://github.com/chao1224/MoleculeSTM?tab=readme-ov-file) repository. Place the download graph model in the `MoleculeSTM/' folder.

---

## Training
You can update the training config in the `config_file` folder.
### Step1. Molecular graph-language alignment
```bash
python train.py --root_train 'data/MoleculeDesc/' --root_eval 'data/MoleculeDesc/' --devices '0,1,2,3' --filename "stage1" --max_epochs 3 --mode train --inference_batch_size 16 --batch_size 4 --config_file config_file/stage1.yaml --accumulate_grad_batches 4
```

### Step2. Instruction tuning
```bash
python train.py --root_train 'data/instruction_tuning/' --root_eval 'data/MoleculeDesc/' --devices '0,1,2,3' --filename "stage2" --max_epochs 3 --mode train --inference_batch_size 16 --batch_size 4 --config_file config_file/stage2.yaml --accumulate_grad_batches 4 --stage_path "./all_checkpoints/stage1/last.ckpt"
```
<!-- 

## Acknowledgements
This repo is built upon [Consistency Models](https://github.com/openai/consistency_models) and [CTM](https://github.com/sony/ctm). -->

## Contact
If you have any questions, please create an issue on this repository or contact at lpmn678@korea.ac.kr.

## Citation
If you find our work interesting, please consider giving a ⭐ and citation.
```bibtex
@inproceedings{park2024llamo,
  title={LLaMo: Large Language Model-based Molecular Graph Assistant},
  author={Park, Jinyoung and Bae, Minseong and Ko, Dohwan and Kim, Hyunwoo J},
  booktitle={NeurIPS},
  year={2024}
}
```
