import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

class ImageCaptioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioner, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Linear layer to reduce ResNet output dimensions
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Linear layer to predict words
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, images, captions):
        # Extract features from image
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.embed(features)
        
        # Generate captions
        embeddings = torch.cat((features.unsqueeze(1), captions), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
    def generate_caption(self, image, vocab, max_length=20):
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Process single image
        image = transform(image).unsqueeze(0)
        
        # Generate caption
        with torch.no_grad():
            features = self.resnet(image)
            features = features.reshape(features.size(0), -1)
            features = self.embed(features)
            
            states = None
            inputs = features.unsqueeze(1)
            
            caption = []
            for i in range(max_length):
                hiddens, states = self.lstm(inputs, states)
                outputs = self.linear(hiddens.squeeze(1))
                predicted = outputs.argmax(1)
                
                if vocab.idx2word[predicted.item()] == '<end>':
                    break
                    
                caption.append(vocab.idx2word[predicted.item()])
                inputs = self.embed(predicted).unsqueeze(1)
                
        return ' '.join(caption)

# Example usage
def caption_image(image_path, model, vocab):
    image = Image.open(image_path)
    return model.generate_caption(image, vocab)