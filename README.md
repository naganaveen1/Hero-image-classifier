# Hero Image Classifier

A simple deep learning-based image classification project built with PyTorch to recognize Telugu heroes from images. This model uses a fine-tuned ResNet18 architecture to classify images into one of four hero categories: **Akhil**, **Brahmi**, **Nani**, or **RamCharan**.

---

## 🔧 Project Structure

```
Hero-image-classifier/
├── Akhil/                # Folder containing images of Akhil
├── Brahmi/               # Folder containing images of Brahmi
├── Nani/                 # Folder containing images of Nani
├── RamCharan/            # Folder containing images of RamCharan
├── savedmodel.ipynb      # Jupyter Notebook for loading the model and predicting
├── telugu_hero_model.pth # Trained PyTorch model (not shown in repo)
├── README.md             # This file
```

---

## 🚀 How It Works

1. **Model Architecture**

   * Based on `ResNet18`
   * Last fully connected layer replaced: `nn.Linear(..., 4)` for 4 hero classes

2. **Image Preprocessing**

   * Resize to `(224, 224)`
   * Normalize with mean and std `[0.5, 0.5, 0.5]`

3. **Prediction Function**

```python
from PIL import Image
from torchvision import transforms
import torch

def predict_image(image_path, model, transform, class_names):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()
    return class_names[predicted_idx]
```

---

## 📦 Requirements

* Python 3.8+
* PyTorch
* torchvision
* PIL (Pillow)

You can install dependencies using:

```bash
pip install torch torchvision pillow
```

---

## ✅ Example Usage

```python
class_names = ['Akhil', 'Brahmi', 'Nani', 'RamCharan']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("telugu_hero_model.pth"))
model.eval()

image_path = "Akhil/sample.jpg"
predicted_hero = predict_image(image_path, model, transform, class_names)
print("Predicted hero:", predicted_hero)
```

---

## 🧠 Model Output

The model outputs the predicted hero name based on the highest class probability from the image input.

---

## 📌 Notes

* The `telugu_hero_model.pth` file must be present in the working directory to load the model.
* All input images should be RGB format.

---

## 💡 Future Ideas

* Add more hero classes
* Convert to web app using Flask or Streamlit
* Train with more diverse image data for better accuracy
* Add data augmentation for generalization

---

## 📜 License

This project is for educational purposes only.
