# Sign Language Production Model - Technical Documentation

## Project Overview

This project implements a sign language production model that translates text or gloss into 3D skeleton coordinates representing sign language gestures. The model is based on a Transformer encoder-decoder architecture and is designed for potential deployment in a mobile application.

## Prerequisites

*   Python 3.8+
*   PyTorch 1.10+
*   TensorFlow 2.5+
*   ONNX
*   onnx-tf
*   React Native
*   tensorflow-lite-react-native

Specific versions might be required depending on your environment and compatibility needs.

## PyTorch Model Creation

To create the initial PyTorch model, you need to use the code provided in the other files within this project. Specifically, you will utilize `model.py`, `training.py`, `data.py`, and the configurations defined in `Configs/Base.yaml`. These files contain the necessary components to train the model, which is a Transformer encoder-decoder architecture designed for sign language production.



## Commands to Run

Before running these commands, make sure you have:

*   Installed all the required dependencies.
*   Adapted the scripts according to your specific model and data.
*   Prepared your training data.

1.  **Train the PyTorch Model:**
```bash 
python training.py --config Configs/Base.yaml
```
You may need to adjust the path to the configuration file or add other arguments depending on the project's specific requirements.

2.  **Convert PyTorch to ONNX:**

```bash
python conversion/pytorch_to_onnx.py
```
3.  **Convert ONNX to TensorFlow Lite:**

```bash
python conversion/onnx_to_tflite.py
```

**Note:** Before running these commands, make sure you have:

*   Installed all the required dependencies.
*   Adapted the scripts according to your specific model and data.
*   Prepared your training data and configured the data paths.
    
## Model Conversion

1.  **PyTorch to ONNX:**

    *   **Script:** `conversion/pytorch_to_onnx.py`
    *   **Command:** `python conversion/pytorch_to_onnx.py`
    *   **Input Preparation:**
        *   Modify the script to load your trained PyTorch model and vocabulary.
        *   Prepare a sample input that matches your model's expected input format. The input should be a tuple containing:
            *   `src`: Source text indices (shape: `[batch_size, sequence_length]`)
            *   `trg_input`: Target input (shape: `[batch_size, sequence_length, trg_size + 1]`)
            *   `src_mask`: Source mask (shape: `[batch_size, 1, 1, sequence_length]`)
            *   `src_lengths`: Source lengths (shape: `[batch_size]`)
            *   `trg_mask`: Target mask (shape: `[batch_size, 1, sequence_length, sequence_length]`)
        *   Replace placeholder values with actual data or adapt the code to generate appropriate sample inputs.
    *   **Output:** `sign_language_model.onnx`
    *   **Note:** Ensure that the input names and dynamic axes in the `torch.onnx.export` function match your model's architecture.

2.  **ONNX to TensorFlow Lite:**

    *   **Script:** `conversion/onnx_to_tflite.py`
    *   **Command:** `python conversion/onnx_to_tflite.py`
    *   **Input:** `sign_language_model.onnx` (from the previous step)
    *   **Output:** `sign_language_model.tflite`
    *   **Quantization:** The script applies default quantization. For more advanced quantization (e.g., post-training integer quantization), modify the script to provide a representative dataset and configure the desired quantization options.

## ISL Dataset Adaptation (Optional)

1.  **Modify `data.py`:**
    *   Introduce a configuration option `num_joints` in the `data` section to specify the number of joints in the skeleton data. Default: 75.
    *   Introduce a configuration option `label_type` in the `data` section to specify the type of labels ("text" or "gloss"). Default: "gloss".
    *   Update the `load_data` function to read skeleton data and labels from files based on the configured `num_joints` and `label_type`.
    *   Modify the `SignProdDataset` class to correctly parse the skeleton data and labels.
    *   Ensure that the skeleton data is a space-separated list of joint coordinates and the labels are a space-separated list of words or glosses.
    *   Adapt the file paths and data loading logic to match the structure of your ISL dataset.

2.  **Modify `Configs/Base.yaml`:**
    *   Add `num_joints` and `label_type` options to the `data` section.
    *   Set the default value for `num_joints` to 75 and the default value for `label_type` to "gloss".

**Note:** Adapting to a new ISL dataset requires a thorough understanding of the dataset's structure and may involve significant code modifications in `data.py` and potentially other parts of the codebase.

## React Native Integration

1.  **Setup:** Create a new React Native project (if needed).

2.  **Install TensorFlow Lite React Native:**

```bash
npm install tensorflow-lite-react-native
```

3.  **Load Model:**

```javascript 
import * as tflite from 'tensorflow-lite-react-native'
```
```js
// ... inside your component ...

const [interpreter, setInterpreter] = useState(null)

useEffect(() => {
  async function loadModel() {
    try {
      const interpreter = await tflite.loadModel({
        modelPath: './assets/sign_language_model.tflite', // Place model in assets dir
      })
      setInterpreter(interpreter)
      console.log('Model loaded successfully!')
    } catch (error) {
      console.error('Failed to load the model:', error)
    }
  }
  loadModel()
}, [])
```

4.  **Preprocess Input:**

```js
async function runModel(inputText) {
  if (!interpreter) {
    console.warn('Model not loaded yet.')
    return
  }

  // --- Preprocess inputText (adapt to your specific needs) ---
  const preprocessedInput = preprocessText(inputText)
  const inputTensor = createInputTensor(preprocessedInput) // Use tflite or a helper lib

  try {
    const outputTensor = await interpreter.run(inputTensor)
    // --- Postprocess outputTensor (adapt to your model's output) ---
    const signLanguageOutput = postprocessOutput(outputTensor)
    displaySignLanguage(signLanguageOutput)
  } catch (error) {
    console.error('Error running the model:', error)
  }
continu}
```

Adapt the `preprocessText`, `createInputTensor`, `postprocessOutput`, and `displaySignLanguage` functions based on your model's input/output format and desired visualization.

5.  **Run Inference:** Use `interpreter.run(inputTensor)` to execute the model.

6.  **Postprocess Output:** Convert the output tensor to a suitable representation for visualization.

7.  **Visualize Output:** Implement a method to visualize the gestures (animations, skeleton drawing, etc.).


## Running the Model

1.  Convert the PyTorch model to TensorFlow Lite using the scripts in the `conversion` directory.
2.  (Optional) Adapt the model to a specific ISL dataset by modifying `data.py` and `Configs/Base.yaml`.
3.  Integrate the TensorFlow Lite model into your React Native application following the steps outlined above.
4.  Run your React Native application on an Android device or emulator.

## Notes and Troubleshooting

*   Ensure consistent preprocessing between training and inference. The `preprocessText` function in your React Native app must precisely match the preprocessing steps used during model training.
*   Adapt all file paths to your specific environment.
*   Verify that the input data format for the conversion scripts is correct, including the dimensions of the input tensors.
*   For errors during model conversion, double-check the input data, model architecture, and paths.
*   For errors in the React Native app, ensure that the model is loaded correctly, the input is preprocessed appropriately, and the output is handled correctly. Use console logs to debug the data flow and identify potential issues.
*   The provided code snippets are conceptual and require adaptation to your specific model and application requirements.
