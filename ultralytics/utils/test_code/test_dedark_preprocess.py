import torch
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image
import os


class MockArgs:
    """Mock arguments class for testing"""

    def __init__(self, fog_FLAG=True):
        self.fog_FLAG = fog_FLAG


class TestDetectionTrainer:
    """Simplified version of DetectionTrainer for testing preprocess_batch"""

    def __init__(self, fog_FLAG=True):
        self.args = MockArgs(fog_FLAG)
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

        # Apply dark channel processing similar to the first file
        if hasattr(self.args, 'fog_FLAG') and self.args.fog_FLAG:
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
            batch['defog_A'] = torch.from_numpy(defog_A).float().to(self.device)
            batch['IcA'] = torch.from_numpy(np.expand_dims(IcA, axis=-1)).permute(0, 3, 1, 2).float().to(self.device)

            # Use the processed image (same as clean_img in this case, following the original logic)
            batch["img"] = batch["clean_img"]

        else:
            # Original processing when fog_FLAG is not set
            batch["img"] = torch.pow(batch["clean_img"], self.dark_param)

        recover_loss = torch.nn.functional.mse_loss(batch["img"], batch["clean_img"])
        batch["recovery_loss_batch"] = recover_loss

        return batch


def create_test_batch(batch_size=2, height=416, width=416):
    """Create a test batch with sample images"""

    # Method 1: Create synthetic test images
    def create_synthetic_image(h, w):
        # Create a more realistic test image with different regions
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Add some geometric shapes and gradients
        # Sky region (top 1/3) - lighter
        img[:h // 3, :, :] = [180, 200, 220]  # Light blue sky

        # Middle region - objects
        img[h // 3:2 * h // 3, :w // 2, :] = [100, 150, 100]  # Green vegetation
        img[h // 3:2 * h // 3, w // 2:, :] = [120, 100, 80]  # Brown buildings

        # Ground region (bottom 1/3) - darker
        img[2 * h // 3:, :, :] = [80, 80, 80]  # Dark ground

        # Add some noise for realism
        noise = np.random.normal(0, 10, (h, w, 3))
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    # Create batch of images
    images = []
    for i in range(batch_size):
        img = create_synthetic_image(height, width)
        images.append(img)

    # Convert to PyTorch tensor (BCHW format)
    images_np = np.stack(images)  # (B, H, W, C)
    images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # Create mock batch dict
    batch = {
        'img': images_tensor,
        'batch_idx': torch.arange(batch_size),
        'cls': torch.zeros((batch_size, 1)),
        'bboxes': torch.zeros((batch_size, 4)),
        'im_file': [f'test_image_{i}.jpg' for i in range(batch_size)]
    }

    return batch


def visualize_results(original_batch, processed_batch, save_dir='test_output'):
    """Visualize the results of preprocessing"""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch_size = original_batch['img'].shape[0]

    for i in range(batch_size):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        orig_img = original_batch['img'][i].permute(1, 2, 0).cpu().numpy()
        orig_img = np.clip(orig_img / 255.0, 0, 1)
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
        if 'defog_A' in processed_batch:
            defog_A = processed_batch['defog_A'][i].cpu().numpy()
            axes[1, 0].bar(['R', 'G', 'B'], defog_A, color=['red', 'green', 'blue'])
            axes[1, 0].set_title(f'Atmospheric Light\nR:{defog_A[0]:.3f}, G:{defog_A[1]:.3f}, B:{defog_A[2]:.3f}')

            # Show IcA (dark channel)
            IcA = processed_batch['IcA'][i, 0].cpu().numpy()
            im = axes[1, 1].imshow(IcA, cmap='gray')
            axes[1, 1].set_title('Dark Channel (IcA)')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 0].text(0.5, 0.5, 'No fog processing\n(fog_FLAG=False)',
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
        plt.savefig(f'{save_dir}/test_result_image_{i}.png', dpi=150, bbox_inches='tight')
        plt.show()


def run_test():
    """Run the complete test"""
    print("=" * 50)
    print("Testing preprocess_batch function")
    print("=" * 50)

    # Test with fog processing enabled
    print("\n1. Testing with fog_FLAG=True")
    trainer_fog = TestDetectionTrainer(fog_FLAG=True)

    # Create test batch
    test_batch = create_test_batch(batch_size=2, height=256, width=256)
    print(f"Created test batch with shape: {test_batch['img'].shape}")

    # Process the batch
    processed_batch_fog = trainer_fog.preprocess_batch(test_batch.copy())

    # Print results
    print(f"Original image range: [{test_batch['img'].min():.3f}, {test_batch['img'].max():.3f}]")
    print(
        f"Clean image range: [{processed_batch_fog['clean_img'].min():.3f}, {processed_batch_fog['clean_img'].max():.3f}]")
    print(f"Processed image range: [{processed_batch_fog['img'].min():.3f}, {processed_batch_fog['img'].max():.3f}]")

    if 'defog_A' in processed_batch_fog:
        print(f"Atmospheric light shape: {processed_batch_fog['defog_A'].shape}")
        print(f"IcA shape: {processed_batch_fog['IcA'].shape}")
        print(f"Atmospheric light values (first image): {processed_batch_fog['defog_A'][0].cpu().numpy()}")

    print(f"Recovery loss: {processed_batch_fog['recovery_loss_batch'].item():.6f}")

    # Test with fog processing disabled
    print("\n2. Testing with fog_FLAG=False")
    trainer_no_fog = TestDetectionTrainer(fog_FLAG=False)
    processed_batch_no_fog = trainer_no_fog.preprocess_batch(test_batch.copy())

    print(f"Recovery loss (no fog): {processed_batch_no_fog['recovery_loss_batch'].item():.6f}")
    print(f"Has defog_A: {'defog_A' in processed_batch_no_fog}")
    print(f"Has IcA: {'IcA' in processed_batch_no_fog}")

    # Visualize results
    print("\n3. Generating visualizations...")
    visualize_results(test_batch, processed_batch_fog, 'test_output_fog')
    visualize_results(test_batch, processed_batch_no_fog, 'test_output_no_fog')

    print("\nTest completed! Check the generated images in test_output_fog/ and test_output_no_fog/ directories.")


if __name__ == "__main__":
    run_test()