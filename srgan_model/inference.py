import torch
from PIL import Image
import torchvision.transforms as transforms
from srgan_model.models import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path: str) -> torch.nn.Module:
    upscale_factor = 4  # Upscale factor used during training (e.g., 4)
    netG = Generator(upscale_factor=upscale_factor).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    netG.load_state_dict(checkpoint['model'])
    netG.eval()

    return netG


def preprocess(input_image_path: str) -> torch.Tensor:
    crop_size = 128
    transform = transforms.Compose([
        # transforms.Resize((1080 // 8, 1920 // 8)),
        # transforms.CenterCrop((crop_size, crop_size)),
        transforms.ToTensor(),
    ])

    input_image = Image.open(input_image_path).convert('RGB')
    lr_image = transform(input_image).unsqueeze(0).to(device)

    return lr_image


def inference(netG: torch.nn.Module, lr_image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        sr_image = netG(lr_image)

    return sr_image


def postprocess(lr_image: torch.Tensor, sr_image: torch.Tensor) -> tuple:
    lr_image = lr_image.squeeze(0).cpu()
    lr_image = torch.clamp(lr_image, 0, 1)
    lr_image = transforms.ToPILImage()(lr_image)
    print("Original image size:", lr_image.size)

    sr_image = sr_image.squeeze(0).cpu()
    sr_image = torch.clamp(sr_image, 0, 1)
    sr_image = transforms.ToPILImage()(sr_image)
    print("Super Res image size:", sr_image.size)

    return lr_image, sr_image


def save_images(lr_image: Image.Image, sr_image: Image.Image):
    lr_image.save('srgan_model/image_original_test.png')
    sr_image.save('srgan_model/image_super_resolved_test.png')

    print(f"Super-resolved image saved.")
    print(f"Original image saved.")


if __name__ == '__main__':
    # === Configuration ===
    model_path = '/home/talmacsi/BME/WeatherConverter/srgan_model/weights/swift_srgan_4x.pth.tar'  # Path to your pre-trained Generator model
    input_image_path = '/home/talmacsi/BME/WeatherConverter/srgan_model/image_original.png'  # Path to the input low-resolution image

    # Load the pre-trained Generator model
    netG = load_model(model_path)

    # Preprocess the input image
    lr_image = preprocess(input_image_path)

    # Perform super-resolution
    sr_image = inference(netG, lr_image)

    # Postprocess the images
    lr_image, sr_image = postprocess(lr_image, sr_image)

    # Save the images
    save_images(lr_image, sr_image)
