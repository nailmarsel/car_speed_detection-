from torchvision.models import mobilenet_v2


def load_mobilenetv2_model():
    model = mobilenet_v2(pretrained=True)
    model.eval()
    return model