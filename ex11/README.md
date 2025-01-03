# Fine Tuning LLM 

1. Create a directory with these files:
- `fineTuneLM.py`
- `Dockerfile`
- `requirements.txt`
- Your dataset file (CSV or JSON)

2. Build the Docker image:
```bash
docker build -t llama-finetuning .
```

3. Run the training:
```bash
docker run --gpus all -v /path/to/your/data:/app/data -v /path/to/output:/app/output \
    llama-finetuning \
    --data_path /app/data/your_dataset.csv \
    --output_dir /app/output \
    --epochs 3 \
    --batch_size 4
```

Key features of this implementation:

1. **Data Processing**:
   - Supports both CSV and JSON formats
   - Automatically formats Q&A pairs into instruction-following format
   - Handles tokenization and batching

2. **Training Setup**:
   - Uses 16-bit precision for memory efficiency
   - Implements gradient accumulation
   - Includes validation during training
   - Saves checkpoints and logs to TensorBoard

3. **Customization Options**:
   - Adjustable learning rate
   - Configurable batch size
   - Flexible number of epochs
   - Customizable model name
