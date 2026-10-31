# CMDPAD 中文复现指南（修复分支）

此分支保留作者的 MAT 架构，补齐丢失的 `src/utils.py` 并修复配置、训练和对话特征提取中的问题。**通过代码自检不等于复现论文指标。**

## 数据限制

数据申请：https://huggingface.co/datasets/HensonXie/CMDPAD 。截至 2026-10-31，数据卡说明需要签署研究使用协议并获得授权，且当前释放的文件只包含 train 和 valid；论文的 test 暂未公布，验证指标不得冒充测试指标。

## 1. 环境与数据检查（在项目根目录执行）

~~~bash
git clone --branch reproduce/cmdpad-baseline-fixes https://github.com/lyx888-lyx/CMDPAD.git
cd CMDPAD
conda create -n cmdpad python=3.10 -y
conda activate cmdpad
# GPU 用户需按 PyTorch 官网选择兼容 CUDA/驱动的 torch 版本。
python -m pip install torch pandas numpy scikit-learn huggingface_hub
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -m unittest discover -s tests -v
~~~

已经克隆项目时，可运行 `git fetch origin` 和 `git switch --track origin/reproduce/cmdpad-baseline-fixes`（首次切换）。

完成 HF 授权后：

~~~bash
hf auth login
hf download HensonXie/CMDPAD --repo-type dataset --local-dir ./dataset
python scripts/check_data.py --data-dir ./dataset
~~~

## 2. 单轮 AR / PR / AP 训练

~~~bash
mkdir -p logs
python -u src/main.py --task affect_recognition --modalities bert-base-chinese wav2vec2-large-robust-emotion convnext-base --num_workers 0 2>&1 | tee logs/ar.log
python -u src/main.py --task personality_recognition --modalities bert-base-chinese wav2vec2-large-robust-emotion convnext-base --epochs 80 --patience 16 --num_workers 0 2>&1 | tee logs/pr.log
python -u src/main.py --task affect_prediction --modalities bert-base-chinese wav2vec2-large-xlsr-chinese convnext-base --num_workers 0 2>&1 | tee logs/ap.log
~~~

单轮阶段使用 GPU 0（默认）；只有 CPU 时加 `--gpu -1`。权重保存在 `saved_models/`。

## 3. 提取双轮 CLS，再训练融合模型

~~~bash
python -u src/dialogue_feature_extractor_multi.py \
  --features ./dataset/multimodal_features_paired.pkl \
  --labels ./dataset/label_paired.csv \
  --model_A_aff ./saved_models/affect_recognition__bert-base-chinese_wav2vec2-large-robust-emotion_convnext-base.pt \
  --model_A_per ./saved_models/personality_recognition__bert-base-chinese_wav2vec2-large-robust-emotion_convnext-base.pt \
  --model_B ./saved_models/affect_prediction__bert-base-chinese_wav2vec2-large-xlsr-chinese_convnext-base.pt \
  --out_cache ./dataset/dialogue_cls_multi.pkl 2>&1 | tee logs/cls.log

python -u src/dialogue_train_from_cls.py \
  --cls_cache ./dataset/dialogue_cls_multi.pkl \
  --label_cols valence_prediction \
  --epochs 40 --device cuda 2>&1 | tee logs/dialogue.log
~~~

CPU 用户将最后一条中的 `--device cuda` 改为 `--device cpu`；双轮特征提取脚本会自动回退到 CPU。

### 对比论文（以下均为原文 Test 指标）

- Table 5，单轮三模态 AR：MAE **0.514**，Corr **0.647**。
- Table 6，单轮三模态 AP：MAE **0.517**，Corr **0.606**。
- Table 7，单轮 PR 五维平均：MAE **0.538**，Corr **0.533**。
- Table 8，双轮 AR+PR+AP：MAE **0.440**，Corr **0.719**，Acc2 **79.9%**，F1 **80.0%**。

**透明说明**：原代码未真正执行论文声明的“排除零标签”规则，本分支在 Acc2/F1 中排除了真实值严格等于零的样本。因此如存在零标签，这两个指标不再与旧代码完全一致。对话训练脚本默认学习率是 `1e-5`，论文统一训练设置写的是 `1e-4`；此处先保留公开代码默认值，再根据验证集做单变量消融，不把两种设置混成一次结果。

运行后请回传 `scripts/check_data.py` 的输出、四份日志、GPU/PyTorch 版本。若缺少官方 test，最后只能报告验证集复现/方法验证，不能宣称复现了论文测试集数值。
