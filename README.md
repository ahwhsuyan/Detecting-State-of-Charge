
This repository contains the implementation of the State of Charge (SoC) false reporting attacks via reinforcement learning, and the corresponding deep neural network (DNN) detection models.

## Repository Structure

### Training Scripts
- `train_dnn.py`: Code for training the detection model. This script includes comprehensive complexity analysis (FLOPs, MACs, parameters), latency testing, and detailed performance metric tracking.
- `train_rl.py`: Code for training the RL agent to generate intelligent attacks.
- `options_origin.py`: Description of arguments for training the RL agent.
- `options1.py`: Description of arguments for training the detection model.
- `soc_dataset.py`: PyTorch dataset class for defining how to retrieve samples from the dataset.

### Evaluation and Metrics
Outputs of a run will be saved to timestamped directories (e.g., `11.9begin/YYYY-MM-DD_HH-MM/`), including:
- Detailed metrics in `all_metrics.txt` (model configuration, computational overhead, inference latency).
- Training and validation loss/accuracy plots (`train_loss.pdf`, `accuracy.pdf`).
- Best model checkpoints.

## Usage

### Training the Detection Model

To train a detection model on a dataset, use the following command and input required arguments:

```bash
python train_dnn.py --lr_model LEARNING_RATE --lr_decay LEARNING_RATE_DECAY --n_epochs NUM_EPOCHS --batch_size BATCH_SIZE
```

You may also add:
```bash
--train_dataset TRAIN_PATH --val_dataset VAL_PATH --test_dataset TEST_PATH
```
to specify the training/val/test dataset.

See `options1.py` for other arguments that can be specified. Outputs of a run will be saved to the specified `save_dir` or timestamped directory.

### Training the RL Agent

To train an RL agent with `gamma=0`, use the following command and input required arguments:

```bash
python train_rl.py --lr_model LEARNING_RATE --lr_decay LEARNING_RATE_DECAY --n_epochs NUM_EPOCHS --batch_size BATCH_SIZE --exp_beta EXP_BETA
```

To train an RL agent with gamma regularization, add the arguments:
```bash
--regularize --gamma GAMMA
```

See `options_origin.py` for other arguments that can be specified. Outputs of a run will be saved to `outputs/[NUM_CARS]_[GAMMA]/run_X`.

## Evaluation

### Evaluating the Detection Model
To evaluate a trained detection model on the test dataset, run the following and add the path to the model parameters:

```bash
python train_dnn.py --eval_only --load_path PATH_TO_TRAINED_MODEL
```

### Evaluating the RL Agent
To test a trained RL agent in the charging simulation, run the following with the path to the agent's parameters:

```bash
python train_rl.py --eval_only --load_path PATH_TO_TRAINED_AGENT
```
Add the arguments `--regularize --gamma GAMMA` if the agent was trained with gamma regularization.

### Testing Synthetic Attacks
To test a synthetic attack strategy:

```bash
python train_rl.py --eval_only --attack_model attackX
```
Where `X` represents the synthetic attack type and can be any of `1-4`.

### Testing Detection Accuracy on Attacks
To test the detection accuracy of a DNN model on RL agent attacks:

```bash
python train_rl.py --eval_detect --load_path PATH_TO_TRAINED_AGENT --load_path2 PATH_TO_DETECTION_MODEL --gamma GAMMA --regularize
```

To test detection accuracy on a synthetic attack:

```bash
python train_rl.py --eval_detect --load_path PATH_TO_TRAINED_AGENT --attack_model attackX
```

## RL Environment

- `charging_env.py`: Contains code for the charging simulation.
- `reinforce_baseline.py`: Contains code for the Exponential baseline in policy gradient.

## Attack Policies

- `attack_policy/DNNAgent.py`: Model for the adversarial RL agent.
- `attack_policy/spoof_agentX.py`: Model for an agent which follows synthetic Attack strategy X (See paper for synthetic attacks considered).

## Detection Model

- `detection_model_origin.py` / `DetectionModelDNN.py`: Defines the DNN architecture of the detection model, including `EfficientAdditiveAttention`, `ECAAttention`, and `Simam_module` components.

## Dataset

The training/validation/testing datasets for the RL agent can be found in the `rl_datasets` directory.

The datasets for the detection models can be found in the `dnn_datasets` directory:
- Files with the name format `dataset_X_syn.py` correspond to datasets with both synthetic and intelligent attacks.
- Files with the name format `dataset_X.pt` correspond to datasets with intelligent attacks only.

Each sample in these datasets contains a SoC sequence of an EV over a period of 24 hours (reported every 30 min). Therefore, **each sample is of size 49, including the label** (i.e., whether the sample is malicious or not).

Feedback is welcome.
