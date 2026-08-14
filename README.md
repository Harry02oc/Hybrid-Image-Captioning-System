# Hybrid-Image-Captioning-System
This project generates captions for uploaded images by combining deep learning models with the Gemini Vision API.
The system uses VGG16 for image feature extraction, YOLOv8 for object detection, and an LSTM model with Beam Search to generate the initial caption. A text-cleaning step helps reduce repetitive words, while the Gemini Vision API is used to refine the caption based on the image context and generate a natural Hindi translation.
The complete pipeline is integrated into a Streamlit dashboard, where users can upload an image and view the generated caption.

Technologies
Python
TensorFlow / Keras
VGG16
YOLOv8
LSTM
Beam Search
Gemini Vision API
OpenCV
Streamlit

Limitations

The caption generation model was trained using the Flickr8k dataset, which is relatively small for an image captioning task. Because of the limited training data and available hardware resources, the base model could sometimes generate inaccurate or repetitive captions. To improve the final output, the project uses the Gemini Vision API as an additional refinement step. This hybrid approach helps provide more context-aware and natural captions while keeping the locally trained deep learning model as part of the pipeline.
