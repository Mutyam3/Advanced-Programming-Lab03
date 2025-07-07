# Lab03: TensorFlow vs PyTorch

**TH Deggendorf – Campus Cham**  
**Prof. Tobias Schaffer**

This repository contains the implementation, training logs, converted models, and report for **Lab03: TensorFlow and PyTorch**.

---

## Objective

In this lab, we:

- Build and train the same small neural network model using both TensorFlow and PyTorch.
- Measure and compare training time and performance.
- Convert the trained models into lightweight formats suitable for embedded deployment:
  - TensorFlow Lite
  - ONNX

---

## Tasks

### Task 1: Model Implementation and Training

- **Dataset:** MNIST handwritten digits (28×28 grayscale images, 10 classes)
- **Model Architecture:**
  - Flatten input (784 features)
  - Dense layer with 64 ReLU units
  - Output layer with 10 units
    - Softmax activation for TensorFlow
    - Logits for PyTorch
- Load and normalize MNIST data
- Implement the model:
  - TensorFlow → `tf.keras.Sequential`
  - PyTorch → custom `nn.Module`
- Train both models for 5 epochs
- Measure training time

---

### Task 2: Inference and Evaluation

- Run inference on the test set using both models
- Report test accuracy and inference time
- Tools:
  - TensorFlow: `model.evaluate()`
  - PyTorch: `model.eval()` with `torch.no_grad()`

---

### Task 3: Model Conversion

#### TensorFlow → TensorFlow Lite

- Convert the trained TensorFlow model to TensorFlow Lite:

  ```python
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  with open('model.tflite', 'wb') as f:
      f.write(tflite_model)
  
#### PyTorch → ONNX

- Export the trained PyTorch model to .onnx format using dummy input:

  ```python
  dummy_input = torch.randn(1, 784)
  torch.onnx.export(
       model,
       dummy_input,
       "model.onnx",
       input_names=["input"],
       output_names=["output"]
   )

## Report Highlights

The report Lab03 Report.pdf summarizes:

  - Differences in code structure and development experience between TensorFlow and PyTorch
  - Training and inference speed comparison
  - Ease of model export in each framework

## Outputs

This lab produces the following outputs:

- `model.tflite` – TensorFlow Lite converted model
- `model.onnx` – ONNX exported PyTorch model
- Training and inference logs
- Accuracy and time metrics printed in console

## Authors

This lab was completed by:

- **Prof. Tobias Schaffer** – Lab Supervisor
- **Mutyam Bhargav Reddy** – Student Implementation

