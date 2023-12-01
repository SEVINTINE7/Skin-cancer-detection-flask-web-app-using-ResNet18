from flask import Flask, render_template, request, redirect, url_for
from plot_pred import pred_and_plot_image
from model.modelClass import ResNet18
from torchvision import transforms
import torch
from pathlib import Path
from timeit import default_timer as timer
import os

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('System_Fundamental.html', result=None)


remov_directory = []

@app.route('/predict', methods=['POST'])
def predict(remov_directory=remov_directory):
    model_path = Path("model/Model2.pth")

    # Define the class names for predictions
    class_names = ['Actinic keratosis',
                   'Basal cell carcinoma',
                   'Benign keratosis',
                   'Dermatofibroma',
                   'Melanocytic nevus',
                   'Melanoma',
                   'Squamous cell carcinoma',
                   'Vascular lesion']

    # Load your deep learning model
    model = ResNet18()  # Replace with the actual class of your model

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])

    # Get the uploaded image file
    images = request.files.getlist("image")

    labels = []
    probs = []
    image_paths = []

    
    

    if not os.path.exists('static'):
        os.mkdir('static')

    start_time = timer()
    for image in images:
        img_path = "static/" + image.filename
        remov_directory.append(img_path)
        image.save(img_path)
        try:
            final = pred_and_plot_image(class_names=class_names,
                                        model=model,
                                        image_path=img_path,
                                        transform=manual_transforms)
            labels.append(final[0])
            probs.append(final[1])
        except Exception as e:
            print("Exception: ",str(e))
        image_paths.append(img_path)
        
    end_time = timer()
    total_time = end_time - start_time
    data = {"image_name": image_paths,
            "Labels": labels,
            "Probs": probs}
    tput = len(images) / (total_time)
    latency = 1 / tput
    latency = round(latency, 4)
    tput = round(tput, 4)
    total_time = round(total_time, 4)
    return render_template('testinghtml.html',
                           data=data,
                           lenght=total_time,
                           tput=tput,
                           latency=latency)


@app.route('/goback')
def go_back(remov_directory=remov_directory):
    for i in remov_directory:
        os.remove(i)
    del remov_directory[:]
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
