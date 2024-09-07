import subprocess
import uuid
import modal
import os
import threading
import re
from helper import cleanup, create_firebase_object, download_video_s3, get_number_of_frames, get_video_dimensions, remove_duplicate_frames, resize_video_in_place, tag_firebase_object, update_firebase_object, upload_rife_engine

app = modal.App("video-interpolation-test")
vsgan_image = modal.Image.from_registry("styler00dollar/vsgan_tensorrt:minimal").pip_install_from_requirements("requirements.txt").copy_local_dir("engines", "/root/engines").copy_local_file("inference.py", "/root/inference.py").copy_local_file("process_video.py", "/root/process_video.py").copy_local_dir("src", "/root/src").copy_local_file("helper.py", "/root/helper.py").copy_local_file("dedup.py", "/root/dedup.py").copy_local_file("toona-firebase.json", "/root/toona-firebase.json")

class Job:
    def __init__(self, key="test", url="", task="", scale=None, multi=None, slowmotion=None):
        self.key = key
        self.video_s3_url = url
        self.task = task
        self.scale = scale
        self.multi = multi
        self.slowmotion = slowmotion

@app.function(image=vsgan_image)
@modal.web_endpoint(method="POST")
def entry(file: dict):
    
    user_id = file["user_id"]
    video = file['video']
    task = file['task'] # upscale or interpolate
    
    multi = None
    slowmotion = None
    scale = None
    
    if task == "upscale":
        scale = file['scale'] # scale factor for upscaling
    else:
        multi = file['multi'] # multiplication factor for interpolation
        slowmotion = file['slowmotion']
        
    # generate unique key for this run
    key = str(uuid.uuid4())
    print("key ", key)

    create_firebase_object(key=key, video=video, user_id=user_id, scale=scale, multi=multi, task=task, slowmotion=slowmotion)
    
    job = Job(key=key, url=video, task=task, scale=scale, multi=multi, slowmotion=slowmotion)
    
    run_interpolation.spawn(key=key, job=job)
    
    return {"key": key}

@app.function(image=vsgan_image, timeout=20000, gpu="A10G")
async def run_interpolation(key: str, job: Job):
    # Execute vspipe command
    subprocess.run(["nvidia-smi"])
    
    # upload_rife_engine()
    # print("generating rife engine for rife", os.path.exists("/root/engines/rife422_v2_ensembleFalse_op20_fp16_clamp.onnx"))
    # generate_rife_engine()
    run_command_on_thread(job)
    
def run_vsgan_command(command, job, total_frames):
    try:
        # Run the command and capture both stdout and stderr
        # Start the process
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Regular expression to match frame number
        frame_pattern = re.compile(r'frame=\s*(\d+)')

        prev = 0
        # Read and process the output stream
        for line in process.stdout:
            # Print output to the console
            print(line, end='')

            # Search for the frame number in the line
            match = frame_pattern.search(line)
            
            if match:
                frame_number = int(match.group(1))
                
                progress = 0
                if job.task == "upscale":
                    progress = round((frame_number / total_frames) * 100)
                else:
                    # interpolate
                    progress = round(frame_number / (total_frames * int(job.multi)) * 100)
                
                if progress > 10 + prev:
                    update_firebase_object(job.key, "running", "", progress)
                    prev = progress

        # Wait for the process to complete and get the return code
        return_code = process.wait()
        
        return return_code, None, None  # We don't capture separate stdout and stderr here


    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None
    
    

def run_command_on_thread(job):
    def execute_command():
        try:
            os.makedirs(f"./{job.key}", exist_ok=True)
            print("task", job.task)
                
            update_firebase_object(job.key, "running", "", 0)
            file_extension = (job.video_s3_url.split('.')[-1]).split('?')[0]
            input_path = f"./{job.key}/input.{file_extension}"
            output_path = f"./{job.key}/output.{file_extension}"

            # get s3 input
            download_video_s3(job.video_s3_url, input_path)
            
            original_width, original_height = get_video_dimensions(input_path)        
            
            command = ""
            
            if job.task == "upscale":
                # ensure min size of video
                longest_side = max(original_width, original_height)
                
                size = "small"
                if (longest_side * int(job.scale)) >= 7680:
                    size = "large"
                    print("DESTINATION OUTPUT SIZE IS 8K")

                command = f'''
                vspipe -c y4m 
                --arg "args=upscale \"{input_path}\" {job.scale} {size}" "/root/inference.py" - | ffmpeg -i pipe: {output_path}'''
            
            else: 
                # ensure min size of video
                resize_video_in_place(input_path, job.key)
                print("resized video")
            
                # if interpolate deduplicate frames
                remove_duplicate_frames(input_path)
                
                # interpolation
                command = f'''
                vspipe -c y4m 
                --arg "args=interpolate \"{input_path}\" {job.multi} {job.slowmotion}" "/root/inference.py" - | ffmpeg -i pipe: {output_path}'''
                
            # Remove newlines and extra spaces
            command = ' '.join(command.split())
            print("Running command:", command)
            
            # calculate total frames
            total_frames = get_number_of_frames(input_path)
            print("total frames:", total_frames)

            # Run the subprocess and capture output
            exit_code, stdout, stderr = run_vsgan_command(command, job, total_frames)

            if exit_code != 0:
                print(f"Command failed with exit code {exit_code}")

        except Exception as e:
            print(f"Error occurred in thread: {e}")
        
        # Check if output file was created
        if not os.path.exists(output_path):
            print("Output file not found. Exiting...")
            cleanup(job, output_path, "error")
            return
        
        if (job.task == "interpolate"):
            # resize if necessary
            new_width, new_height = get_video_dimensions(output_path)
            if (not new_width == original_width) or (not new_height == original_height):
                resize_video_in_place(output_path, job.key, original_width, original_height)
        
        # upload to s3
        cleanup(job, output_path, "complete")
        
        tag_firebase_object(job.key)

    thread = threading.Thread(target=execute_command)
    thread.start()
    thread.join()
    return thread
