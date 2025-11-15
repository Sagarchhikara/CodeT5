# Guide to Fine-Tuning CodeT5 on a Custom Local Dataset

This guide provides a comprehensive, step-by-step walkthrough for fine-tuning the CodeT5 model on your own local dataset for any sequence-to-sequence (seq2seq) task, such as code summarization, documentation generation, or code translation.

## Step 1: Set Up the Environment

Before you begin, you need to set up a dedicated Python environment and download the necessary pre-trained models.

### 1. Create a Virtual Environment

It is highly recommended to use a virtual environment to avoid dependency conflicts.

**Using `venv`:**
```bash
# Create the virtual environment
python3 -m venv codet5-finetune-env

# Activate it
source codet5-finetune-env/bin/activate
```

**Using `conda`:**
```bash
# Create the virtual environment
conda create -n codet5-finetune-env python=3.8

# Activate it
conda activate codet5-finetune-env
```

### 2. Install Dependencies

The `CodeT5/README.md` specifies the following required libraries. Install them using `pip`:

```bash
pip install torch==1.7.1
pip install tensorboard==2.4.1
pip install transformers==4.6.1
pip install tree-sitter==0.2.2
```

### 3. Download Pre-trained Models

You will need a pre-trained CodeT5 model to start the fine-tuning process. The repository provides `gsutil` commands to download them. Make sure you have the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and authenticated.

From the root of the repository, run the following command:

```bash
gsutil -m cp -r "gs://sfr-codet5-data-research/pretrained_models" .
```
This will download models like `codet5-small` and `codet5-base` into a `pretrained_models` directory.


## Step 2: Prepare Your Dataset

The model expects the data to be in a specific format.

### 1. Split Your Data

First, split your dataset into three sets:
- **`train`**: For training the model (e.g., 80% of your data).
- **`dev` (validation)**: For evaluating the model during training (e.g., 10%).
- **`test`**: For final evaluation of the model's performance on unseen data (e.g., 10%).

### 2. Structure Your Data Files

For each data split, create two separate plain text files:
- A `.source` file containing the input sequences.
- A `.target` file containing the corresponding output sequences.

**Crucially, each line in the `.source` file must correspond to the same line in the `.target` file.**

### 3. Create a Dataset Directory

Create a new directory for your dataset, for example, inside `CodeT5/data/`.

**Example Directory Structure:**
```
CodeT5/
├── data/
│   └── my_custom_task/
│       ├── train.source
│       ├── train.target
│       ├── dev.source
│       ├── dev.target
│       ├── test.source
│       └── test.target
└── ...
```

## Step 3: Create a Custom Data Loading Function

You need to write a Python function to load your custom dataset.

1.  **Open the `CodeT5/_utils.py` file.**
2.  **Add the following function** to the end of the file. This function will read your `.source` and `.target` files and create a list of `InputExample` objects.

```python
# In CodeT5/_utils.py
import os

# Make sure InputExample is imported from utils
# from utils import InputExample

def read_my_custom_task_data(data_dir, mode):
    """
    Load custom dataset for a seq2seq task.
    """
    print("Reading my_custom_task data from:", data_dir)
    source_file = os.path.join(data_dir, mode + '.source')
    target_file = os.path.join(data_dir, mode + '.target')
    examples = []
    with open(source_file, 'r', encoding='utf-8') as f_source, \
         open(target_file, 'r', encoding='utf-8') as f_target:
        for i, (source_line, target_line) in enumerate(zip(f_source, f_target)):
            source = source_line.strip()
            target = target_line.strip()
            if source and target:
                examples.append(
                    InputExample(idx=i, source=source, target=target)
                )
    print(f"Loaded {len(examples)} examples for mode {mode}")
    return examples
```
*Note: Replace `read_my_custom_task_data` with a name that better reflects your task (e.g., `read_python_docstring_data`).*

## Step 4: Update Configuration Files

Now, you need to "register" your new task and data loader with the training scripts.

### In `CodeT5/configs.py`:

1.  **Add your task to `TASK_LIST`**:
    ```python
    TASK_LIST = ['summarize', 'concode', 'translate', 'refine', 'defect', 'clone', 'multi_task', 'my_custom_task']
    ```

2.  **Add a configuration block for your task** inside the `Config` class `__init__` method:
    ```python
    # After another task's block
    elif args.task == 'my_custom_task':
        self.sub_task = 'none'

    # Add hyperparameter settings for your task
    if args.task == 'my_custom_task':
        self.max_source_length = 512  # Adjust as needed
        self.max_target_length = 128  # Adjust as needed
    ```

### In `CodeT5/utils.py`:

1.  **Import your new data loader**:
    ```python
    from _utils import (..., read_my_custom_task_data)
    ```

2.  **Update `get_filenames`** to point to your data directory:
    ```python
    # After another task's block
    elif task == 'my_custom_task':
        data_dir = f'{data_root}/my_custom_task'
    ```
    *Make sure the folder name (`my_custom_task`) matches the directory you created.*

3.  **Update `read_examples`** to use your data loader:
    ```python
    # After another task's block
    elif args.task == 'my_custom_task':
        examples = read_my_custom_task_data(data_dir, mode)
    ```

## Step 5: Run the Fine-Tuning Script

You are now ready to run the fine-tuning process.

1.  **Navigate to the root directory** of the repository.
2.  **Make sure your virtual environment is activated.**
3.  **Execute the `run_exp.py` script** with the correct arguments.

**Command:**
```bash
python sh/run_exp.py \
    --model_tag codet5_base \
    --task my_custom_task \
    --sub_task none
```

For a quick test run on a small subset of your data, you can add the `--data_num` argument:
```bash
python sh/run_exp.py \
    --model_tag codet5_base \
    --task my_custom_task \
    --sub_task none \
    --data_num 100
```

Once training is complete, your fine-tuned model will be saved in the `saved_models/` directory by default. You can then use it for inference on your test set.
