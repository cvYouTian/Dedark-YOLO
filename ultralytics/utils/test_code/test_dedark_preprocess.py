import torch
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from pathlib import Path


class MockArgs:
    """Mock arguments class for testing"""

    def __init__(self, dedark_FLAG=True):
        self.dedark_FLAG = dedark_FLAG


class TestDetectionTrainer:
    """Simplified version of DetectionTrainer for testing preprocess_batch"""

    def __init__(self, dedark_FLAG=True):
        self.args = MockArgs(dedark_FLAG)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dark_param = 0.7  # Default dark parameter

    def DarkChannel(self, im):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        return dc

    def AtmLight(self, im, dark):
        [h, w] = im.shape[:2]
        imsz = h * w
        numpx = int(max(math.floor(imsz / 1000), 1))
        darkvec = dark.reshape(imsz, 1)
        imvec = im.reshape(imsz, 3)

        indices = darkvec.argsort(0)
        indices = indices[(imsz - numpx):imsz]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A

    def DarkIcA(self, im, A):
        im3 = np.empty(im.shape, im.dtype)
        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]
        return self.DarkChannel(im3)

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['clean_img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

        if hasattr(self.args, 'dedark_FLAG') and self.args.dedark_FLAG:
            batch_size = batch['clean_img'].shape[0]
            height = batch['clean_img'].shape[2]
            width = batch['clean_img'].shape[3]

            # Convert from PyTorch tensor (BCHW) to numpy (BHWC) for processing
            clean_imgs_np = (batch['clean_img'].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            # Initialize arrays like in the first file
            defog_A = np.zeros((batch_size, 3))
            IcA = np.zeros((batch_size, height, width))

            # Process each image in batch
            for i in range(batch_size):
                dark_i = self.DarkChannel(clean_imgs_np[i])
                defog_A_i = self.AtmLight(clean_imgs_np[i], dark_i)
                IcA_i = self.DarkIcA(clean_imgs_np[i], defog_A_i)
                defog_A[i, ...] = defog_A_i
                IcA[i, ...] = IcA_i

            # Convert back to PyTorch tensors
            batch['dedark_A'] = torch.from_numpy(defog_A).float().to(self.device)
            batch['IcA'] = torch.from_numpy(np.expand_dims(IcA, axis=-1)).permute(0, 3, 1, 2).float().to(self.device)

            # Use the processed image (same as clean_img in this case, following the original logic)
            batch["img"] = batch["clean_img"]

        else:
            # Original processing when dedark_FLAG is not set
            batch["img"] = torch.pow(batch["clean_img"], self.dark_param)

        recover_loss = torch.nn.functional.mse_loss(batch["img"], batch["clean_img"])
        batch["recovery_loss_batch"] = recover_loss

        return batch


def check_image_directory(image_dir):
    """Check if the provided directory exists and contains images"""
    if not image_dir:
        return False, "No image directory provided"

    if not os.path.exists(image_dir):
        return False, f"Directory '{image_dir}' does not exist"

    if not os.path.isdir(image_dir):
        return False, f"'{image_dir}' is not a directory"

    # Check for image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    if not image_files:
        return False, f"No image files found in '{image_dir}'"

    return True, f"Found {len(image_files)} images in '{image_dir}'"


def load_images_from_directory(image_dir, target_size=(416, 416), max_images=5):
    """Load real images from a directory"""

    # Support common image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    if not image_files:
        print(f"No images found in {image_dir}")
        return None, []

    print(f"Found {len(image_files)} images in {image_dir}")

    # Limit the number of images to process
    image_files = image_files[:max_images]

    images = []
    valid_files = []

    for img_path in image_files:
        try:
            # Load image using PIL
            img_pil = Image.open(img_path).convert('RGB')

            # Resize image
            img_pil = img_pil.resize(target_size, Image.LANCZOS)

            # Convert to numpy array
            img_np = np.array(img_pil)

            images.append(img_np)
            valid_files.append(img_path)

            print(f"✓ Loaded: {os.path.basename(img_path)} - Shape: {img_np.shape}")

        except Exception as e:
            print(f"✗ Error loading {img_path}: {e}")

    if not images:
        return None, []

    return np.stack(images), valid_files


def create_test_batch_from_images(image_dir, target_size=(416, 416), max_images=5):
    """Create a test batch from real images in the specified directory"""

    # Check if directory is valid
    is_valid, message = check_image_directory(image_dir)
    if not is_valid:
        raise ValueError(f"❌ {message}")

    print(f"📁 Loading images from: {image_dir}")

    # Load images from directory
    images_np, image_files = load_images_from_directory(image_dir, target_size, max_images)

    if images_np is None or len(images_np) == 0:
        raise ValueError(f"❌ No valid images could be loaded from {image_dir}")

    batch_size = len(images_np)
    print(f"✅ Successfully loaded {batch_size} images")

    # Convert to PyTorch tensor (BCHW format)
    images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()

    # Create mock batch dict
    batch = {
        'img': images_tensor,
        'batch_idx': torch.arange(batch_size),
        'cls': torch.zeros((batch_size, 1)),
        'bboxes': torch.zeros((batch_size, 4)),
        'im_file': image_files
    }

    return batch


def visualize_results(original_batch, processed_batch, save_dir='test_output'):
    """Visualize the results of preprocessing"""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch_size = original_batch['img'].shape[0]

    for i in range(batch_size):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Get image filename for title
        img_name = os.path.basename(original_batch['im_file'][i])
        fig.suptitle(f'Image: {img_name}', fontsize=16)

        # Original image
        orig_img = original_batch['img'][i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        axes[0, 0].imshow(orig_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Clean image (normalized)
        clean_img = processed_batch['clean_img'][i].permute(1, 2, 0).cpu().numpy()
        clean_img = np.clip(clean_img, 0, 1)
        axes[0, 1].imshow(clean_img)
        axes[0, 1].set_title('Clean Image (Normalized)')
        axes[0, 1].axis('off')

        # Processed image
        processed_img = processed_batch['img'][i].permute(1, 2, 0).cpu().numpy()
        processed_img = np.clip(processed_img, 0, 1)
        axes[0, 2].imshow(processed_img)
        axes[0, 2].set_title('Processed Image')
        axes[0, 2].axis('off')

        # Show atmospheric light values if available
        if 'dedark_A' in processed_batch:
            dedark_A = processed_batch['dedark_A'][i].cpu().numpy()
            axes[1, 0].bar(['R', 'G', 'B'], dedark_A, color=['red', 'green', 'blue'])
            axes[1, 0].set_title(f'Atmospheric Light\nR:{dedark_A[0]:.3f}, G:{dedark_A[1]:.3f}, B:{dedark_A[2]:.3f}')
            axes[1, 0].set_ylim(0, max(dedark_A) * 1.2)

            # Show IcA (dark channel)
            IcA = processed_batch['IcA'][i, 0].cpu().numpy()
            im = axes[1, 1].imshow(IcA, cmap='gray')
            axes[1, 1].set_title('Dark Channel (IcA)')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        else:
            axes[1, 0].text(0.5, 0.5, 'No dedark processing\n(dedark_FLAG=False)',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')

        # Show recovery loss
        recovery_loss = processed_batch['recovery_loss_batch'].item()
        axes[1, 2].text(0.5, 0.5, f'Recovery Loss:\n{recovery_loss:.6f}',
                        ha='center', va='center', transform=axes[1, 2].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 2].axis('off')

        plt.tight_layout()

        # Save with a more descriptive filename
        safe_name = img_name.replace('.', '_').replace('/', '_')
        output_file = f'{save_dir}/result_{safe_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {output_file}")
        plt.show()


def run_test(image_dir):
    """Run the complete test with your local images"""

    if not image_dir:
        print("❌ Error: You must provide an image directory!")
        print("\n📋 Usage Instructions:")
        print("   run_test('path/to/your/images')")
        print("\n📁 Examples:")
        print("   run_test('./my_images')           # Relative path")
        print("   run_test('/home/user/Pictures')   # Linux absolute path")
        print("   run_test('C:/Users/user/Pictures') # Windows absolute path")
        print("\n📸 Supported formats: JPG, JPEG, PNG, BMP, TIFF")
        return None, None

    print("=" * 60)
    print("Testing preprocess_batch function with YOUR IMAGES")
    print("=" * 60)
    print(f"📂 Image directory: {image_dir}")

    try:
        # Create test batch from your local images
        test_batch = create_test_batch_from_images(image_dir, target_size=(512, 512), max_images=5)
        print(f"📊 Created test batch with shape: {test_batch['img'].shape}")
        print(f"📂 Processing {len(test_batch['im_file'])} images:")
        for i, file_path in enumerate(test_batch['im_file']):
            print(f"   {i + 1}. {os.path.basename(file_path)}")

    except ValueError as e:
        print(f"\n{e}")
        print("\n💡 Tips:")
        print("   - Make sure the directory path is correct")
        print("   - Check that the directory contains image files")
        print("   - Supported formats: JPG, JPEG, PNG, BMP, TIFF")
        return None, None
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return None, None

    # Test with dedark processing enabled
    print("\n🔍 1. Testing with dedark_FLAG=True")
    trainer_dedark = TestDetectionTrainer(dedark_FLAG=True)

    # Process the batch
    processed_batch_dedark = trainer_dedark.preprocess_batch(test_batch.copy())

    # Print results
    print(f"📈 Original image range: [{test_batch['img'].min():.3f}, {test_batch['img'].max():.3f}]")
    print(
        f"📈 Clean image range: [{processed_batch_dedark['clean_img'].min():.3f}, {processed_batch_dedark['clean_img'].max():.3f}]")
    print(
        f"📈 Processed image range: [{processed_batch_dedark['img'].min():.3f}, {processed_batch_dedark['img'].max():.3f}]")

    if 'dedark_A' in processed_batch_dedark:
        print(f"🌫️ Atmospheric light shape: {processed_batch_dedark['dedark_A'].shape}")
        print(f"🌫️ IcA shape: {processed_batch_dedark['IcA'].shape}")
        print("🌫️ Atmospheric light values for each image:")
        for i, img_file in enumerate(test_batch['im_file']):
            A_values = processed_batch_dedark['dedark_A'][i].cpu().numpy()
            print(f"   {os.path.basename(img_file)}: R={A_values[0]:.3f}, G={A_values[1]:.3f}, B={A_values[2]:.3f}")

    print(f"💰 Recovery loss: {processed_batch_dedark['recovery_loss_batch'].item():.6f}")

    # Test with dedark processing disabled
    print("\n🔍 2. Testing with dedark_FLAG=False")
    trainer_no_dedark = TestDetectionTrainer(dedark_FLAG=False)
    processed_batch_no_dedark = trainer_no_dedark.preprocess_batch(test_batch.copy())

    print(f"💰 Recovery loss (no dedark): {processed_batch_no_dedark['recovery_loss_batch'].item():.6f}")
    print(f"❓ Has dedark_A: {'dedark_A' in processed_batch_no_dedark}")
    print(f"❓ Has IcA: {'IcA' in processed_batch_no_dedark}")

    # Visualize results
    print("\n🎨 3. Generating visualizations...")
    visualize_results(test_batch, processed_batch_dedark, 'test_output_dedark')
    visualize_results(test_batch, processed_batch_no_dedark, 'test_output_no_dedark')

    print("\n✅ Test completed!")
    print("📁 Check the generated images in:")
    print("   - test_output_dedark/ (with dedark processing)")
    print("   - test_output_no_dedark/ (without dedark processing)")

    return processed_batch_dedark, processed_batch_no_dedark


if __name__ == "__main__":
    # Instructions and examples
    print("🖼️  Real Image Processing Test")
    print("=" * 50)
    print("This script processes YOUR local images for testing dedark preprocessing.")
    print("\n📋 Usage:")
    print("   python script.py")
    print("   Then call: run_test('path/to/your/images')")
    print("\n📁 Examples:")
    print("   run_test('./test_images')              # Relative path")
    print("   run_test('/home/user/Pictures')        # Linux")
    print("   run_test('C:/Users/user/Pictures')     # Windows")
    print("   run_test('/Users/user/Pictures')       # macOS")
    print("\n📸 Supported formats: JPG, JPEG, PNG, BMP, TIFF")
    print("\n💡 To run the test, call:")
    print("   run_test('YOUR_IMAGE_DIRECTORY_PATH')")

    # Example - uncomment and modify the path below to test with your images:
    run_test('/home/youtian/Documents/pro/pyCode/Dedark-YOLO/ultralytics/utils/test_code/sample_images')