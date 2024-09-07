import modal

if not modal.is_local():
    import cv2
    import tempfile
    import subprocess
    import os
    import requests
    import shutil
    from datetime import datetime
    import sys
    
    import boto3
    import firebase_admin
    from firebase_admin import credentials
    from firebase_admin import firestore

    cred = credentials.Certificate("toona-firebase.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    firebase_collection = "vsgan-trt-jobs"

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    region_name = os.environ.get('AWS_REGION_NAME')
    bucket_name = os.environ.get('BUCKET_NAME')
    
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    s3 = session.client('s3')

class Job:
    def __init__(self, key="test", url="", task="", scale=None, multi=None, slowmotion=None):
        self.key = key
        self.video_s3_url = url
        self.task = task
        self.scale = scale
        self.multi = multi
        self.slowmotion = slowmotion

def remove_duplicate_frames(file_path: str):
    try:
        command = f"python dedup.py {file_path}"
        print(f"Executing command: {command}")
        
        # Run the command and capture output in real-time
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print stdout in real-time
        for line in process.stdout:
            print(f"STDOUT: {line.strip()}")
        
        # Print stderr in real-time
        for line in process.stderr:
            print(f"STDERR: {line.strip()}", file=sys.stderr)
        
        # Wait for the process to complete and get the return code
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        
        print(f"Command completed successfully with return code {return_code}")
        
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    
def resize_video_in_place(input_file: str, key: str, width=1920, height=1080):
    """
    Resizes the input video to 1920x1080 and replaces the original video with the resized version.

    :param input_file: Path to the input video file.
    """
    # Create a temporary file to store the resized video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=f"./{key}")
    temp_file.close()

    try:
        # Resize the video and save it to the temporary file
        command = [
            'ffmpeg',
            '-i', input_file,        # Input file
            '-vf', f'scale={width}:{height}', # Video filter to scale the video to 1920x1080
            '-c:a', 'copy',          # Copy the audio without re-encoding
            temp_file.name, "-y"           # Temporary output file
        ]
        
        subprocess.run(command, check=True)

        # Replace the original file with the resized video
        os.replace(temp_file.name, input_file)
    finally:
        # Clean up the temporary file if it still exists
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

def get_number_of_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Error opening video file {video_path}")
    
    # Get the number of frames
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture object
    video.release()
    
    return frame_count

def get_video_dimensions(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not video.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get the dimensions
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video capture object
    video.release()
    
    return width, height


def cleanup(job: Job, output_path: str, status: str):
    global running
    
    if (os.path.exists(f"./{job.key}")):
        
        print(os.listdir(f"./{job.key}"))
        
        # Upload the video to S3
        url = upload_video_s3(job.key, output_path)
        
        # Update the Firebase object with the S3 URL
        update_firebase_object(job.key, status, url, 100)
        
        # Remove this threads working directory
        shutil.rmtree(f"{job.key}")

        # Remove the job from the running queue
        running = None
        
# S3
    
# download video from s3    
def download_video_s3(url, download_path):
    try:
        # Send an HTTP GET request to the Discord attachment URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Write the video content to the specified path
        with open(download_path, 'wb') as file:
            file.write(response.content)
        
        print(f"Video downloaded successfully to {download_path}")
    except Exception as e:
        print(f"Error downloading video: {e}")

# upload video to s3
def upload_video_s3(key, video_path):
    file_extension = video_path.split('.')[-1]
    object_name = f"outputs/{key}.{file_extension}"
    
    try:
        # Upload the file
        s3.upload_file(video_path, bucket_name, object_name)
        print(f"File uploaded successfully to {bucket_name}/{object_name}")
        url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
        return url
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        
        
# Firebase

# update firebase object
def update_firebase_object(key, status, video_url="", progress=0):
    object_ref = db.collection(firebase_collection).document(key)
    object_ref.update({
        "status": status,
        "video_out": video_url,
        "progress": progress,
    })

# add end timestamp to firebase object
def tag_firebase_object(key: str):
    timestamp = datetime.now().timestamp()
    object_ref = db.collection(firebase_collection).document(key)  
    object_ref.update({
        "timestamp_end": timestamp
    })

# create firebase object for tooncrafter results
def create_firebase_object(key: str, video: str, user_id: str, scale: int, multi: int, task: str, slowmotion: str):
    object_ref = db.collection(firebase_collection).document(key)
    
    timestamp = datetime.now().timestamp()

    if task == "upscale":
        object_ref.set({
            "status": "queued",
            "task": task,
            "video_in": video,
            "video_out": "",
            "user_id": user_id,
            "scale": scale,
            "taskId": key,
            "progress": 0,
            "timestamp": timestamp
        })
        
    else:
        object_ref.set({
            "status": "queued",
            "task": task,
            "video_in": video,
            "video_out": "",
            "user_id": user_id,
            "multi": multi,
            "taskId": key,
            "progress": 0,
            "timestamp": timestamp,
            "slowmotion": slowmotion
        })
        
def upload_rife_engine():
    
    if os.path.exists(f"/root/volumes/rife.engine"):
        print("RIFE engine generated successfully")
        try:
            # Upload the file
            s3.upload_file("/root/volumes/rife.engine", bucket_name, "rife.engine")
            print(f"File uploaded successfully to {bucket_name}/rife.engine")
            url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/rife.engine"
            print(url)
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
    else:
        print("!!! RIFE engine generation failed !!!")
    
def generate_rife_engine():
    output = []
    sys.path.append("/workspace/tensorrt/")
    output.append(f"Generating RIFE engine. ONNX file exists: {os.path.exists('/root/engines/rife422_v2_ensembleFalse_op20_fp16_clamp.onnx')}")

    command = (
        "trtexec --verbose --bf16 --fp16 "
        "--onnx=engines/rife422_v2_ensembleFalse_op20_fp16_clamp.onnx "
        "--minShapes=input:1x7x1080x1920 --optShapes=input:1x7x1080x1920 "
        "--maxShapes=input:1x7x1080x1920 --saveEngine=rife.engine "
        "--tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference "
        "--useCudaGraph --noDataTransfers --builderOptimizationLevel=5"
    )

    try:
        # Run the command and capture output in real-time
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Print and store output in real-time
        for line in process.stdout:
            print(line.strip())
            output.append(line.strip())

        # Wait for the process to complete and get the return code
        return_code = process.wait()

        if return_code != 0:
            output.append(f"Command failed with return code {return_code}")
            return "\n".join(output), None

        output.append("RIFE engine generated successfully")

        # After generating the engine, upload it to S3
        if os.path.exists("/root/rife.engine"):
            try:
                s3 = boto3.client('s3')
                bucket_name = "your-bucket-name"  # Replace with your actual bucket name
                region_name = "your-region-name"  # Replace with your actual region name

                # Upload the file
                s3.upload_file("/root/rife.engine", bucket_name, "model.engine")
                output.append(f"File uploaded successfully to {bucket_name}/model.engine")
                url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/model.engine"
                return "\n".join(output), url
            except Exception as e:
                output.append(f"Error uploading file: {str(e)}")
                return "\n".join(output), None
        else:
            output.append("!!! RIFE engine generation failed !!!")
            return "\n".join(output), None

    except Exception as e:
        output.append(f"An unexpected error occurred: {str(e)}")
        return "\n".join(output), None