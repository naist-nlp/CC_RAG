# Mesuring the confidence on RAG

### Setup
```bash
pip install -r requirements.txt
```

### Run
```bash
## Execution
# As for the execution, GPU is required to run.
bash scripts/execute_inference*.sh

## Evaluation
bash scripts/evaluate_results*.sh

## Plot
bash scripts/plot_*.sh
```


## Citation
If you feel this repository useful, please cite the following paper:

``` bibtex
@misc{ozaki2024understandingimpactconfidenceretrieval,
      title={Understanding the Impact of Confidence in Retrieval Augmented Generation: A Case Study in the Medical Domain}, 
      author={Shintaro Ozaki and Yuta Kato and Siyuan Feng and Masayo Tomita and Kazuki Hayashi and Ryoma Obara and Masafumi Oyamada and Katsuhiko Hayashi and Hidetaka Kamigaito and Taro Watanabe},
      year={2024},
      eprint={2412.20309},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.20309}, 
}
```
