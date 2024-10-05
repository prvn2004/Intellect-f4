# Approaches

## The Challenge

In the face of time constraints and the overwhelming nature of the provided data (both train and test datasets), our approach evolved through multiple iterations as we confronted various challenges. The goal was to extract and identify entities (`entity_name` and `entity_value`) from images and their corresponding metadata.

## Iterations

### Step 1: Building a Custom NER Model

Initially, we experimented with popular Named Entity Recognition (NER) models like **BERT** and **T5**, known for their state-of-the-art performance in natural language tasks.

**Proposed Approach:**

1. Extract text from images using Optical Character Recognition (OCR).
2. Use the extracted text as input to a custom NER model.
3. Map the OCR-extracted text to the corresponding `entity_value` and train the NER model on it, with predictions applied to the test dataset.

**Why This Approach Failed:**

Unfortunately, relying heavily on OCR resulted in several issues:

* **OCR Reliability:** The extracted text often lost key contextual information, which had a significant impact on predictions.
* **Loss of Context:** OCR outputs struggled to maintain the context of the images, leading to large discrepancies in our final predictions.

Even after fine-tuning both BERT and T5 models, the results were unsatisfactory. Despite their reputation for handling complex tasks, these models didn’t produce the desired F1 scores or accurate predictions.

### Step 2: Switching to LLMs for Context Preservation

Realizing the limitations of NER models, we shifted gears to **Large Language Models (LLMs)** like **Gemma2b** and **LLaMA7b**. These models demonstrated better performance than our initial NER-based approach, as they excel at maintaining context. However, they introduced a new problem:

* **Hallucination of Entities and Units:** The models occasionally generated fictitious entity names and values, leading to unpredictable results.

To mitigate this, we fine-tuned the Gemma2-2b model using the full set of `train.csv` images. While this improved the results slightly (a 2% improvement over the base model), the gain was marginal. The overall F1 score remained low at around 0.013, far from our target range of 0.4-0.5.

**The Main Challenge:** Our main roadblock was the OCR tool's performance. The accuracy of the extracted text was inconsistent, and the image quality significantly affected the results. This, combined with the loss of context in identifying entities, led us back to the drawing board.

### Step 3: Multimodal Approach

With time rapidly running out (only 1½ days left!), we needed a solution that could understand not just the text but also the context and meaning behind it, along with visual cues from the images.

**Final Approach: Using Multimodal Models (Qwen2-2b-vl)**

The multimodal model **Qwen2-2b-VL** came to our rescue. It supports both image and text input as tokens, allowing us to feed it both the image and the extracted text simultaneously, preserving context.

**Improvements:**

* After fine-tuning Qwen2-2b-vl, it produced the best predictions of all approaches. 
* We fine-tuned it on `Train.csv` data for two epochs, and it still yielded outstanding results.
* **Pros:** Qwen2-2b-vl can simultaneously understand the contextual information of images and text to predict an output.

**Further Improvements:**

To further optimize Qwen2-2b-vl, it could be trained on a larger portion of `Train.csv` data and for more epochs. This would help it fine-tune and fit the training data as well as possible. It's important to note that we only used 10,000 samples for fine-tuning, so there's potential for significant improvement with additional data and training.

## Code Files

* **Util folder:** Contains scripts to fine-tune the model.
* **d_img.ipynb:** Downloads images in batches along with progress saving.
* **Second.ipynb:** Downloads necessary packages and trains the Qwen model.
* **run_tests.ipynb:** Runs the model on the test set to generate responses, format responses, and produce the final output.


## Approach Comparison

| Approach | Efficiency | Pros | Cons |
|---|---|---|---|
| NER Models (BERT, T5) | Low | State-of-the-art performance in natural language tasks | Reliant on OCR, loss of context |
| LLMs (Gemma2b, LLaMA7b) | Medium | Better context preservation than NER models | Hallucination of entities and units |
| Multimodal Model (Qwen2-2b-vl) | High | Can understand both image and text input | Requires more training data and epochs for optimal performance |

## Team Members

* PRAVEEN KUMAR
* MOHIT MEENA
* RAJ RAUSHAN
* MEHUL SIRVI

## Signoff

Intellect-F4
