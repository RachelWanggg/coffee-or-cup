# coffee-or-cup

Image classification demo using PyTorch, ResNet18, and Apple Metal (MPS) for coffee vs cup recognition with Gradio deployment.

## Project Structure

- `app.py` - Web application interface
- `train.py` - Model training script
- `predict.py` - Prediction script
- `model.pth` - Trained model weights
- `dataset/` - Training dataset
  - `coffee/` - Coffee images
  - `cup/` - Cup images

## Usage

### Training
```bash
python train.py
```

### Making Predictions
```bash
python predict.py --image path/to/image.jpg
```

### Running the Web App
```bash
python app.py
```

## Requirements

See requirements.txt for dependencies.

## License

MIT
