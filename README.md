# Image Captioning with Encoder-Decoder Model

This project implements an image captioning system using a Convolutional Neural Network (CNN) as an encoder and a Recurrent Neural Network (RNN) as a decoder. The system generates captions for images by encoding the visual features using a CNN and decoding them into text using an RNN.

## Sample output generated

<img width="603" alt="Screenshot 2024-08-04 at 1 11 26 PM" src="https://github.com/user-attachments/assets/a0c4d6e9-92a3-4bb4-90c3-1344f9e34b17">

<img width="619" alt="Screenshot 2024-08-04 at 1 08 02 PM" src="https://github.com/user-attachments/assets/b0856790-25a1-4dba-aded-7fd6d201f0ec">


## Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Download the Pre-trained Models:

Download the pre-trained `encoder` and `decoder` model weights and place them in the `models` directory.

## Download the Dataset:

Download the COCO dataset and place the images in the `cocoapi/images/val2017/` directory. Ensure the image paths are correctly referenced in the code.

## Dataset

This project uses the [COCO](http://cocodataset.org/) dataset for image captioning. The dataset contains a wide variety of images with detailed annotations. The `val2017` subset is used for testing the model.

## Directory Structure

```bash
yourprojectname/
│
├── data/
│   └── vocab.pkl              # Vocabulary file
├── models/
│   ├── encoder-3.pkl          # Pre-trained encoder model
│   └── decoder-3.pkl          # Pre-trained decoder model
├── cocoapi/
│   └── images/
│       └── val2017/           # Directory containing validation images
├── inference.ipynb            # Jupyter notebook for running inference
└── README.md                  # Project README file
```

## Model Architecture

### Encoder

The encoder is a pre-trained CNN (e.g., ResNet) that extracts visual features from the input image. These features are then passed to the decoder.

### Decoder

The decoder is an RNN (specifically, an LSTM) that takes the encoded image features as input and generates a caption. The LSTM uses a vocabulary file to convert model outputs into human-readable text.

## Training

The training script trains the encoder-decoder model on the COCO dataset. Due to the computational resources required, training is typically performed on a GPU-enabled machine.

### Training Command

To train the model, use the following command (this is an example; modify according to your needs):

```bash
python train.py --epochs 10 --batch-size 32 --learning-rate 0.001 --embed-size 256 --hidden-size 512
```

## Inference

You can generate captions for images using the pre-trained model. The `inference.ipynb` notebook provides an example of how to do this.

### Running Inference

1. Open the `inference.ipynb` notebook.
2. Follow the instructions in the notebook to load an image, run the model, and generate a caption.

### Example Usage

```python
# Load the image
test_image_path = "cocoapi/images/val2017/000000000785.jpg"
image = Image.open(test_image_path)

# Generate a caption
caption = generate_caption(image)
print(f"Generated Caption: {caption}")
```

## Results

The model generates captions like "A group of people sitting at a table with a laptop." These captions are generated based on the visual features extracted by the encoder.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
