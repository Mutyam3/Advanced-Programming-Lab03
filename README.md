# Advanced-Programming

# Lab03: TensorFlow vs PyTorch

**TH Deggendorf – Campus Cham**  
**Prof. Tobias Schaffer**

This repository contains the implementation, training logs, converted models, and a brief report for **Lab03: TensorFlow and PyTorch**.

## Objective

In this lab, we:

1. Build and train the same neural network model using both TensorFlow and PyTorch.  
2. Compare training and inference times as well as performance metrics.  
3. Convert the trained models into lightweight formats for deployment (TensorFlow Lite and ONNX).

---

## Task Overview

###  Task 1: Model Implementation and Training

- **Dataset:** MNIST handwritten digits (28×28 grayscale images, 10 classes)
- **Model Architecture:**
  - Input: Flatten layer (784 features)
  - Dense layer with 64 ReLU units
  - Output layer:
    - Softmax activation (TensorFlow)
    - Logits (PyTorch)
- **Steps:**
  - Load and normalize MNIST data
  - Implement the model:
    - TensorFlow → `tf.keras.Sequential`
    - PyTorch → custom `nn.Module`
  - Train both models for 5 epochs
  - Measure training time

---

###  Task 2: Inference and Evaluation

- Run inference on test data
- Report:
  - Test accuracy
  - Inference time
- Tools:
  - TensorFlow: `model.evaluate()`
  - PyTorch: `model.eval()` with `torch.no_grad()`

---

###  Task 3: Model Conversion

#### TensorFlow → TensorFlow Lite

- Convert the trained Keras model to `.tflite` format:
  
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open('model.tflite', 'wb') as f:
      f.write(tflite_model)
  
## Export the trained PyTorch model to .onnx format using dummy input:

```python
dummy_input = torch.randn(1, 784)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"]
)


report_highlights:
  summary: "The report Lab03 Report.pdf summarizes:"
  details:
    - "Differences in code structure and development experience"
    - "Training and inference speed comparison"
    - "Ease of model export in each framework"

outputs:
  - file: "model.tflite"
    description: "TensorFlow Lite converted model"
  - file: "model.onnx"
    description: "ONNX exported PyTorch model"
  - description: "Training and inference logs"
  - description: "Accuracy and time metrics printed in console"

authors:
  - name: "Prof. Tobias Schaffer"
    role: "Lab Supervisor"
  - name: "Mutyam Bhargav Reddy"
    role: "Student Implementation"

