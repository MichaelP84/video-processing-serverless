import numpy as np
import cv2
import argparse
import tempfile
import torch
import os

def main():
    parser = argparse.ArgumentParser(description="Process a file path.")
    parser.add_argument('file_path', type=str, help='The path to the file')

    args = parser.parse_args()

    # Check if the file exists
    if os.path.isfile(args.file_path):
        print(f"The file '{args.file_path}' exists.")
    else:
        print(f"The file '{args.file_path}' does not exist.")

    deduplicate_frames(args.file_path)
    

def deduplicate_frames(input_path: str):
    # Initialize the DedupSSIM object
    dedup = DedupSSIMCuda()

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a temporary file for the output video
    temp_dir = os.path.dirname(input_path)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=temp_dir) as temp_file:
        temp_output_path = temp_file.name
        
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to a PyTorch tensor
        frame_tensor = torch.from_numpy(frame).float()  # Convert to float tensor
        
        # Check if the frame is a duplicate
        is_duplicate = dedup.run(frame_tensor)
        
        # If it's not a duplicate, write the original frame (not tensor) to the output video
        if not is_duplicate:
            out.write(frame)
        
    # Release resources
    cap.release()
    out.release()
    
    # Replace the original video with the deduplicated video
    os.replace(temp_output_path, input_path)

    print(f"Video deduplicated and saved to {input_path}")

class DedupSSIMCuda:
    def __init__(
        self,
        ssimThreshold=0.90,
        sampleSize=224,
        half=True,
    ):
        """
        A Cuda accelerated version of the SSIM deduplication method

        Args:
            ssimThreshold: float, SSIM threshold to consider two frames as duplicates
            sampleSize: int, size of the frame to be used for comparison
            half: bool, use half precision for the comparison
        """
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None
        self.half = half

        import torch
        import torch.nn.functional as F
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self.torch = torch
        self.F = F
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(">> DEVICE IS: ", self.DEVICE)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.DEVICE)

    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)
        similarity = self.ssim(self.prevFrame, frame).item()
        self.prevFrame = frame.clone()

        return similarity > self.ssimThreshold

    def processFrame(self, frame):
        frame = (
            frame
            .to(self.DEVICE)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            if not self.half
            else frame
            .to(self.DEVICE)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .half()
        ).mul(1 / 255.0)
        frame = self.F.interpolate(
            frame, (self.sampleSize, self.sampleSize), mode="nearest"
        )
        return frame


class DedupSSIM:
    def __init__(
        self,
        ssimThreshold=0.99,
        sampleSize=224,
    ):
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None

        from skimage.metrics import structural_similarity as ssim
        from skimage import color

        self.ssim = ssim
        self.color = color

    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.ssim(self.prevFrame, frame, data_range=frame.max() - frame.min())
        self.prevFrame = frame.copy()
        

        return score > self.ssimThreshold

    def processFrame(self, frame):
        frame = frame.cpu().numpy()
        frame = np.resize(frame, (self.sampleSize, self.sampleSize, 3))
        frame = self.color.rgb2gray(frame)

        return frame


class DedupMSE:
    def __init__(
        self,
        mseThreshold=1000,
        sampleSize=224,
    ):
        self.mseThreshold = mseThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None

        from skimage.metrics import mean_squared_error as mse
        from skimage import color

        self.mse = mse
        self.color = color

    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.mse(self.prevFrame, frame)
        self.prevFrame = frame.copy()

        return score < self.mseThreshold

    def processFrame(self, frame):
        frame = frame.cpu().numpy()
        frame = np.resize(frame, (self.sampleSize, self.sampleSize, 3))
        frame = self.color.rgb2gray(frame)

        return frame
    
if __name__ == "__main__":
    main()