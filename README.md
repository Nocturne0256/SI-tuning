# SI-Tuning: Structure-Information Injecting Tuning for Protein Language Models

**SI-Tuning** (Structure-Information Injecting Tuning) is a parameter-efficient fine-tuning framework that injects explicit structural information (dihedral angles and distance maps) into large-scale protein language models (PLMs), significantly improving downstream performance with minimal tunable parameters.

This implementation is based on:  **Boost Protein Language Model with Injected Structure Information through Parameter Efficient Fine-Tuning**

## 🔬 Highlights

- 🔄 **Parameter-Efficient**: Fine-tune ESM-2 with <2% parameters using LoRA.
- 🔗 **Structural Fusion**: Inject both individual-level (dihedral angle) and pairwise-level (distance map) structural features.
- 🚀 **Performance**: Outperforms full-parameter tuning and state-of-the-art structure-aware models like SaProt in classification tasks.


## 📦 Environment Setup

We recommend using `conda`:

```bash
cd SI-tuning
conda env create -n situning
bash environment.sh
````

> 💡 *You may adjust the script to install CUDA or use a package mirror based on your local setup.*


## 📁 Data & Model Preparation

### 1. Preprocessed Data

Download the processed cache files from [this link](https://drive.google.com/file/d/1Raxf7vczdMOc1dgfqZWsbIBGJS23qY8i/view?usp=drive_link)

Place all files in:

```bash
./cache/
```

### 2. ESM-2 Pretrained Weights

Download ESM-2 (35M / 650M) from [Hugging Face](https://huggingface.co/facebook/esm2_t33_650M_UR50D) and place them in the current working directory.
Path to the weights should be specified in `config.yaml`.


## 🚀 Usage

### Train and Validate

```bash
python train.py
```

Model, dataset, and training configurations can be modified in `config.yaml`.

## 📊 Results (ESM-2 650M)

| Task                  | Metric  | Full Tuning | SI-Tuning |
| --------------------- | ------- | ----------- | --------- |
| Thermostability       | ρ       | 0.680       | **0.703** |
| Metal Ion Binding     | ACC (%) | 71.56       | **76.05** |
| DeepLoc (Binary)      | ACC (%) | 91.96       | **93.95** |
| DeepLoc (Subcellular) | ACC (%) | 82.09       | **85.40** |
| EC Number             | Fmax    | 0.868       | **0.888** |


## 🧪 Datasets Used

* **Thermostability**
* **Metal Ion Binding**
* **DeepLoc** (Binary & Subcellular)
* **Gene Ontology** (MF / BP / CC)
* **Enzyme Commission (EC)**

All datasets are adapted following [SaProt](https://github.com/westlake-repl/SaProt) settings.


## 📄 License

This project is released under the [GPL-3.0 License](LICENSE).

## 🙏 Acknowledgments

* Based on [ESM](https://github.com/facebookresearch/esm) from Meta AI.
* Based on [Saprot](https://github.com/westlake-repl/SaProt) :Protein Language Modeling with Structure-aware Vocabulary

## 📬 Contact

For questions or collaborations, please feel free to open an issue or contact us via email: zhengjiayou@link.cuhk.edu.cn .

