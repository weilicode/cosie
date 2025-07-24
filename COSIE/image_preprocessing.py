import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import timm
import numpy as np
import pandas as pd
import tqdm
import argparse
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
import pickle
import tifffile
import scanpy as sc
from skimage.transform import rescale
from PIL import Image, ImageOps, ImageChops
from cv2 import findHomography, RANSAC, perspectiveTransform, estimateAffinePartial2D
Image.MAX_IMAGE_PIXELS = None



def load_image(filename, verbose=True):

    """
    Efficiently load an image file and convert it to a NumPy array.

    Parameters
    ----------
    filename : str
        Path to the image file. Supports common formats such as .png, .jpg, .tif.
    
    verbose : bool, optional
        Whether to print log messages during the loading process. Default is True.

    Returns
    -------
    img : np.ndarray
        A NumPy array representing the image. Shape is (H, W) for grayscale or (H, W, 3) for RGB images. The alpha channel is removed if present.
    """
    
    print('loading image...')
    Image.MAX_IMAGE_PIXELS = 2**40
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img

def rescale_image(img, scale):
    """
    Rescale an image by a given scale factor using `skimage.transform.rescale`.

    Parameters
    ----------
    img : np.ndarray
        Input image array. Can be either a 2D grayscale image of shape (H, W), or a 3D color image of shape (H, W, C).
    
    scale : float
        Rescaling factor. The same scale is applied to both height and width dimensions. If the input is 3D, the channel dimension is preserved.

    Returns
    -------
    img : np.ndarray
        The rescaled image array. The output values are kept in the original range of the input (`preserve_range=True`).
    """
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img

def save_jpg(img_array,file_name):
    """
    Convert a NumPy array into a JPEG image and save it to disk.

    Parameters
    ----------
    img_array : np.ndarray
        Input image as a NumPy array. The array should represent a grayscale or RGB image, and will be cast to `uint8` before saving.
    
    file_name : str
        The name to save the image as. The `.jpg` extension will be automatically appended.

    Returns
    -------
    None
        The image is saved as a JPEG file at the specified location.
    """
    img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save('{}.jpg'.format(file_name), format='JPEG')

def mkdir(path):
    """
    Create the parent directory for a given file or directory path if it does not already exist.

    Parameters
    ----------
    path : str
        The full file path or directory path. The function will extract the parent directory and create it if it does not exist.

    Returns
    -------
    None
        The parent directory is created in place if needed.
    """
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)

def save_pickle(x, filename):
    """
    Save a Python object to a .pkl file using the `pickle` module.

    Parameters
    ----------
    x : Any
        The Python object to serialize (e.g., list, dict, NumPy array).
    
    filename : str
        The full path to save the pickle file. If the parent directory does not exist,
        it will be created automatically.

    Returns
    -------
    None
        The object is serialized and saved as a .pkl file.
    """
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)



def load_pickle(filename, verbose=True):
    """
    Load a Python object from a .pkl file.

    Parameters
    ----------
    filename : str
        The full path of the pickle file to load.
    
    verbose : bool, optional
        Whether to print a confirmation message upon successful loading. Default is True.

    Returns
    -------
    x : Any
        The deserialized Python object.
    """
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x




def combine_mask(image_path, mask_path, output_path):
    """
    Modify a grayscale image by applying a binary mask. All pixels in the input image
    corresponding to white (255) pixels in the mask will be set to black (0).
    This is useful when a specific region needs to be recovered as black based on a mask.

    Parameters
    ----------
    image_path : str
        Path to the input grayscale image to be modified.
    
    mask_path : str
        Path to the binary mask image. Must be the same size as the input image.
    
    output_path : str
        Path to save the modified image.

    Returns
    -------
    None
        The modified image is saved to the specified output path.
    """
    # Load images
    image = Image.open(image_path).convert("L")  # Ensure grayscale
    mask = Image.open(mask_path).convert("L")  # Ensure grayscale

    # Ensure both images are the same size
    if image.size != mask.size:
        raise ValueError("The base image and mask must have the same size.")

    # Convert images to pixel data
    image_pixels = image.load()
    mask_pixels = mask.load()

    # Modify the image based on the mask
    for x in range(image.width):
        for y in range(image.height):
            if mask_pixels[x, y] == 255:  # If mask pixel is white
                image_pixels[x, y] = 0  # Set the corresponding base image pixel to black

    # Save the modified image
    image.save(output_path)
    print(f"Modified image saved to {output_path}")



def generate_pxl_location_from_mask(mask_image):
    """
    Extract pixel and spatial coordinates of 16×16 superpixels whose top-left pixels
    fall within the white (255) region of a binary mask.

    Parameters
    ----------
    mask_image : np.ndarray
        A 2D binary mask array. Pixels with value 255 are considered valid, i.e., within the tissue or region of interest (ROI).

    Returns
    -------
    filtered_coordinates : np.ndarray of shape (N, 2)
        Pixel coordinates (in image space) of valid 16×16 superpixels. Each coordinate represents the top-left corner of a 16×16 block.
        
    spatial_location : np.ndarray of shape (N, 2)
        Spatial locations obtained by dividing pixel coordinates by 16. Typically used for spatial indexing or grid-based embedding.
    """
    pixels = np.column_stack(np.where(mask_image == 255))
    pixel_coordinates = [(int(y), int(x)) for y, x in pixels]
    filtered_coordinates = [(y, x) for y, x in pixel_coordinates if x % 16 == 0 and y % 16 == 0]
    filtered_coordinates = np.array(filtered_coordinates).astype(int)
    spatial_location = filtered_coordinates//16
    # (y, x) y corresponds to height(row), x corresponds to width(col)
    return filtered_coordinates, spatial_location




class PatchDataset(Dataset):
    """
    A PyTorch-compatible dataset for extracting 224×224 image patches centered at specified cell coordinates from a high-resolution RGB image.

    Each patch is centered on a given pixel coordinate, padded if it falls near the edge
    of the image, and normalized using standard ImageNet statistics.

    Parameters
    ----------
    image : np.ndarray
        Full-resolution RGB H&E image of shape (H, W, 3).
    
    location : np.ndarray
        Array of shape (N, 2) containing N pixel coordinates for cell centers.

    Returns
    -------
    dataset : PatchDataset
        A PyTorch dataset object. Each item is a tuple (transformed_patch, coordinate), where `transformed_patch` is a normalized 3×224×224 tensor.
    """
    def __init__(self, image, location):
        self.image = image
        self.location = location   # location should be a n*2 numpy array
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        self.shape_ori = np.array(image.shape[:2])
        print('shape_ori:',self.shape_ori)
        self.total_patches = self.location.shape[0]

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):

        center_i = self.location[idx, 0]
        center_j = self.location[idx, 1]

        start_i, start_j = max(0, center_i - 112), max(0, center_j - 112)
        end_i, end_j = min(self.shape_ori[0], center_i + 112), min(self.shape_ori[1], center_j + 112)
        
        patch = self.image[start_i:end_i, start_j:end_j]
        
        # Pad if necessary to ensure 224x224 size
        if patch.shape[0] < 224 or patch.shape[1] < 224:
            padded_patch = np.zeros((224, 224, 3), dtype=patch.dtype)
            padded_patch[(224-patch.shape[0])//2:(224-patch.shape[0])//2+patch.shape[0], 
                         (224-patch.shape[1])//2:(224-patch.shape[1])//2+patch.shape[1]] = patch
            patch = padded_patch

        patch = Image.fromarray(patch.astype('uint8')).convert('RGB')
        return self.transform(patch), (center_i, center_j)


def create_model(local_dir):
    """
    Create and load a pre-trained Vision Transformer (ViT-L/16) model from a HuggingFace-compatible checkpoint.

    Parameters
    ----------
    local_dir : str
        Path to the folder containing the pre-trained model weights (e.g., `pytorch_model.bin`).

    Returns
    -------
    model : torch.nn.Module
        A ViT-Large (patch16, img224) model from the `timm` library, without classification head or global pooling.
    """
    model = timm.create_model(
        "vit_large_patch16_224", 
        img_size=224, 
        patch_size=16, 
        init_values=1e-5, 
        num_classes=0,  # This ensures no classification head
        global_pool='',  # This removes global pooling
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=False)
    return model

# v2
@torch.inference_mode()
def extract_features(model: torch.nn.Module, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts both global and local visual features from an input image batch using a pretrained model.

    Parameters
    ----------
    model : torch.nn.Module
        A pretrained vision transformer model with a method `forward_intermediates()` that allows access to token-level outputs.
    
    batch : torch.Tensor
        A batch of input images.


    Returns
    -------
    feature_emb : torch.Tensor
        The final global feature embedding.

    patch_emb : torch.Tensor
        The final local feature embedding.
    """
    feature_emb = model(batch)
    final_output, _ = model.forward_intermediates(batch, return_prefix_tokens=False)
    local_emb = final_output[:,1:]
    patch_emb = local_emb.permute(0, 2, 1).reshape(batch.shape[0], 1024, 14, 14)
    return feature_emb, patch_emb




@torch.inference_mode()
def image_feature_extraction(
    he_image, 
    uni_local_dir, 
    cell_location, 
    device = 'cuda:0', 
    batch_size=128, 
    num_workers=4):

    """
    Extract image features at cell locations using a pretrained Vision Transformer (ViT) model
    and save the output to disk as a pickle file.

    The function loads a model via `create_model`, initializes a `PatchDataset` using the input
    image and cell pixel locations, and for each patch extracts both global (224×224) and local
    (16×16) features. These features are concatenated into a single representation per cell.

    Parameters
    ----------
    he_image : np.ndarray
        RGB image of shape (H, W, 3).
    
    uni_local_dir : str
        Path to the folder containing the pretrained model weights (e.g., `pytorch_model.bin`).
    
    cell_location : np.ndarray
        An array of shape (N, 2) containing pixel coordinates for N cells.
    
    device : str, optional
        Torch device to use for inference, e.g., `"cuda:0"`. Default is `'cuda:0'`.
    
    batch_size : int, optional
        Batch size for feature extraction. Default is 128.
    
    num_workers : int, optional
        Number of worker threads for the DataLoader. Default is 4.

    Returns
    -------
    None
        The extracted features are saved to disk as a pickle file.
    """


    print('cell num:',cell_location.shape[0])

    model = create_model(uni_local_dir)
    print('Finish loading model')
    
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    
    dataset = PatchDataset(he_image, cell_location)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    patch_embeddings = []
    part_cnts = 0
    for batch_idx, (patches, positions) in enumerate(tqdm.tqdm(dataloader, total=len(dataloader))):

        patches = patches.to(device, non_blocking=True)
        if batch_idx == 0:
            print(f"Batch {batch_idx}:")
            print(f"Shape of patches: {patches.shape}")
            print(f"Shape of positions[0]: {positions[0].shape}")
            print(f"Content of positions[0][:10]: {positions[0][:10]}")
            print(f"Content of positions[1][:10]: {positions[1][:10]}")
        
        feature_emb, patch_emb = extract_features(model, patches)
        
        if batch_idx == 0:
            print(f"Shape of feature_emb: {feature_emb.shape}")
            print(f"Shape of patch_emb: {patch_emb.shape}")
        
        # Process each patch
        for idx in range(len(positions[0])):
            
            # Extract features
            center_feature = feature_emb[idx, 0]  # Use the [CLS] token as the 224-level feature
            patch_feature = patch_emb[idx, :, 7, 7]  # Use the center patch feature
            
            # Concatenate 224-level and 16-level features
            combined_feature = torch.cat([center_feature, patch_feature])
            patch_embeddings.append(combined_feature.cpu().numpy())
            

    save_pickle(patch_embeddings, 'uni_embeddings.pickle')  





def generate_homograph(keypoints_src, keypoints_dst, transform_type="rigid"):
    """
    Generate a geometric transformation matrix from source to destination keypoints,
    using either rigid or affine transformation.

    Parameters
    ----------
    keypoints_src : np.ndarray
        Source keypoints of shape (N, 2), e.g., from image A.
    
    keypoints_dst : np.ndarray
        Destination keypoints of shape (N, 2), e.g., from image B.
    
    transform_type : str, optional
        Type of transformation to apply. Must be one of {"rigid", "affine"}.
        
        - "rigid": Applies rotation and translation only.  
        - "affine": Allows for rotation, translation, and shearing/skewing.  
        
        Default is "rigid".

    Returns
    -------
    trans_mat : np.ndarray
        The transformation matrix:
        
        - Shape is (2, 3) for "rigid" transforms.  
        - Shape is (3, 3) for "affine" transforms.
    """
    if transform_type == "affine":
        trans_mat, _ = findHomography(keypoints_src, keypoints_dst, RANSAC)
    elif transform_type == "rigid":
        trans_mat, _ = estimateAffinePartial2D(keypoints_src, keypoints_dst, method=RANSAC)
        rotation_matrix = trans_mat[:, :2]
        scale = np.sqrt(np.sum(rotation_matrix ** 2, axis=0))  # Get the scale for x and y
        rotation_matrix_normalized = rotation_matrix / scale  # Remove scaling
        trans_mat[:, :2] = rotation_matrix_normalized
    return trans_mat

def transform_coordinates(coords, homography_matrix, transform_type="affine"):
    """
    Apply a geometric transformation matrix (e.g., from `generate_homograph`) to 2D coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates to be transformed, of shape (N, 2).
    
    homography_matrix : np.ndarray
        The transformation matrix. Shape must be (2, 3) for "rigid" or (3, 3) for "affine".
    
    transform_type : str, optional
        The type of transformation to apply. Must be one of {"rigid", "affine"},
        and must match the shape of `homography_matrix`. Default is "affine".

    Returns
    -------
    transformed_coords : np.ndarray
        Transformed 2D coordinates, with shape (N, 2).
    """
    if transform_type == "affine":
        return perspectiveTransform(coords.reshape(-1, 1, 2).astype(np.float32), homography_matrix)[:, 0, :]
    elif transform_type == "rigid":
        return cv2.transform(np.expand_dims(coords, axis=0), homography_matrix)[0]





def get_white_superpixel_centers(image_path, superpixel_size=16):
    """
    Identify the center coordinates of superpixels that are entirely white (255) in a binary image.

    Parameters
    ----------
    image_path : str
        Path to the binary image (grayscale) composed of superpixels.

    superpixel_size : int, optional
        Size (in pixels) of each square superpixel block. Default is 16.

    Returns
    -------
    centers : list of tuple
        List of (x, y) tuples representing the center coordinates of superpixels
        that are fully white. Coordinates are in pixel units.
    """
    # Step 1: load image
    img = Image.open(image_path).convert('L')
    mask = np.array(img)

    H, W = mask.shape
    centers = []

    # Step 2: visit all superpixels 
    for i in range(0, H, superpixel_size):
        for j in range(0, W, superpixel_size):
            patch = mask[i:i+superpixel_size, j:j+superpixel_size]

            # Step 3: check if they are white(255)
            if patch.shape == (superpixel_size, superpixel_size) and np.all(patch == 255):
                # Step 4: return the center location of each filtered superpixel
                center_y = i + superpixel_size // 2
                center_x = j + superpixel_size // 2
                centers.append((center_x, center_y))  # x should be col, y should be row

    return centers



