# Fine Tuning LLMs 

Fine-tuning a LLaMA 3 model for Question Answering (QA) on a laptop with the given specifications requires careful consideration of resource limitations, especially with only 8GB of GPU VRAM. Below is a step-by-step guide to fine-tune the model, including building a custom dataset from PDFs, selecting batch sizes, epochs, and quantization options.


- OS: Ubuntu 22.04.3 LTS
- NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2
- GPU VRAM: 8192MB
- System Memory: 16211524 Kb
- CPU : 12 core Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz

---

### **1. Prepare the Environment**
1. **Install Required Libraries**:
   - Install Python 3.8+ and necessary libraries:
     ```bash
     sudo apt update
     sudo apt install python3-pip
     pip install torch transformers datasets sentencepiece accelerate bitsandbytes
     ```
   - Install PDF processing libraries:
     ```bash
     pip install PyPDF2 pdfplumber
     ```

2. **Verify CUDA Installation**:
   Ensure CUDA is properly installed and accessible:
   ```bash
   nvcc --version
   ```

3. **Set Up Hugging Face Transformers**:
   Use the `transformers` library to load and fine-tune the LLaMA 3 model.

---

### **2. Build a Custom Dataset from PDFs**
1. **Extract Text from PDFs**:
   Use `pdfplumber` or `PyPDF2` to extract text from your PDFs. Save the extracted text into a structured format (e.g., JSON or CSV).

   Example script:
   ```python
   import pdfplumber
   import json

   def extract_text_from_pdf(pdf_path):
       text = ""
       with pdfplumber.open(pdf_path) as pdf:
           for page in pdf.pages:
               text += page.extract_text()
       return text

   pdf_path = "your_pdf_file.pdf"
   extracted_text = extract_text_from_pdf(pdf_path)

   # Save to JSON
   with open("dataset.json", "w") as f:
       json.dump({"text": extracted_text}, f)
   ```

2. **Format the Dataset for QA**:
   Convert the extracted text into a QA format. For example:
   ```json
   [
       {
           "context": "The LLaMA model is a large language model developed by Meta.",
           "question": "Who developed the LLaMA model?",
           "answer": "Meta"
       },
       ...
   ]
   ```

3. **Load the Dataset**:
   Use the `datasets` library to load and preprocess your custom dataset:
   ```python
   from datasets import Dataset

   with open("dataset.json", "r") as f:
       data = json.load(f)

   dataset = Dataset.from_dict(data)
   ```

---

### **3. Load the LLaMA 3 Model**
1. **Load the Pretrained Model**:
   Use the `transformers` library to load the LLaMA 3 model. Since your GPU has limited VRAM, consider using a smaller variant (e.g., `llama-3-7b`).

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_name = "meta-llama/Meta-Llama-3-7B"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
   ```

2. **Quantization**:
   Use 8-bit quantization to reduce memory usage:
   ```python
   from transformers import BitsAndBytesConfig

   quantization_config = BitsAndBytesConfig(
       load_in_8bit=True,
       llm_int8_threshold=6.0
   )
   model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
   ```

---

### **4. Fine-Tune the Model**
1. **Set Training Parameters**:
   - **Batch Size**: Use a small batch size (e.g., 1 or 2) to fit within 8GB VRAM.
   - **Epochs**: Start with 1-3 epochs to avoid overfitting.
   - **Learning Rate**: Use a low learning rate (e.g., 5e-5).

   Example:
   ```python
   from transformers import TrainingArguments, Trainer

   training_args = TrainingArguments(
       output_dir="./results",
       per_device_train_batch_size=1,
       num_train_epochs=3,
       learning_rate=5e-5,
       save_steps=500,
       save_total_limit=2,
       fp16=True,  # Use mixed precision to save memory
   )
   ```

2. **Define the Trainer**:
   ```python
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset,
       tokenizer=tokenizer,
   )
   ```

3. **Start Training**:
   ```python
   trainer.train()
   ```

---

### **5. Evaluate the Model**
1. **Test on Sample Questions**:
   After training, test the model on sample questions:
   ```python
   input_text = "What is the capital of France?"
   inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
   outputs = model.generate(**inputs, max_length=50)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

2. **Save the Fine-Tuned Model**:
   Save the model for future use:
   ```python
   model.save_pretrained("./fine-tuned-llama3-qa")
   tokenizer.save_pretrained("./fine-tuned-llama3-qa")
   ```

---

### **6. Optimize for Limited Resources**
- **Gradient Accumulation**: Use gradient accumulation to simulate larger batch sizes:
  ```python
  training_args = TrainingArguments(
      gradient_accumulation_steps=4,
      ...
  )
  ```
- **Mixed Precision**: Use `fp16` to reduce memory usage.
- **Offloading**: Use CPU offloading for layers that don’t fit in GPU memory:
  ```python
  model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", offload_folder="./offload")
  ```

---

### **7. Monitor GPU Usage**
Use `nvidia-smi` to monitor GPU memory usage during training:
```bash
watch -n 1 nvidia-smi
```

---

### **8. Additional Tips**
- **Dataset Size**: Keep the dataset small to avoid long training times.
- **Early Stopping**: Use early stopping to prevent overfitting.
- **LoRA (Low-Rank Adaptation)**: Consider using LoRA for parameter-efficient fine-tuning.

By following these steps, you can fine-tune a LLaMA 3 model for QA on your laptop while managing resource constraints effectively.