import os
import time
import json
import glob
import subprocess
import shlex
import cv2
from PIL import Image, ImageDraw, ImageFont
import imagehash
import numpy as np
from matplotlib import pyplot as plt 
from pathlib import Path
from urllib.parse import urlparse
import faiss
from lib import util
from lib.frame import Frame
from lib import frame_utils
import re

config = {
   "FRAME_PATH": "frames",
   "LAPLACIAN_PATH": "laplacian",
   "LAPLACIAN_THRESHOLD": 0.2,
   "LAPLACIAN_SIZE": 18,
   "LAPLACIAN_DISTANCE": 0.2,
   "PHASH_THRESHOLD": 0.2,
   "PHASH_SIZE": 18,
   "PHASH_DISTANCE": 4
}

# Must be h.264 video
class VideoFrames:
  def __init__(self, video_file, bucket, max_res=(750, 500), sample_rate_fps=1, force=False):
    self.frames = []
    self.video_file = video_file
    self.filtered = False
    self.filtered_config = None 
    self.stream_info = self.get_stream_info(video_file, force)
    self.start_ms = int(float(self.stream_info['format']['start_time'])*1000)
    self.duration_ms = self.stream_info['video_stream']['duration_ms']
    self.end_ms = self.start_ms + self.duration_ms
    self.object = util.upload_object(bucket, self.video_stem(), video_file)
    self.extract_frames(force, max_res=max_res, sample_rate_fps=sample_rate_fps)  
    dimension = len(self.frames[0]['titan_multimodal_embedding'])
    self.vector_store = self.create_index(dimension)
    self.calculate_similar_frames()

  def video_stem(self):
      return Path(self.video_file).stem

  def video_dir(self):
      return Path(self.video_file).parent

  def video_asset_dir(self):
      return os.path.join(self.video_dir(), self.video_stem())

  def frame_dir(self):
      return os.path.join( self.video_asset_dir(), config["FRAME_PATH"])

  def get_stream_info(self, video_file, force):
    """
      Extract and return detailed information about a video file. 
      The function uses the `ffprobe` command-line tool (part of FFmpeg) to extract detailed metadata about the given video, including 
       - Whether the video is progressive
       - Width and height
       - Duration in milliseconds
       - Number of frames
       - Frame rate
       - Sample aspect ratio
       - Display aspect ratio
       - Display width
       The information extract is saved in a JSON file for downstream consumption.
       Input: 
           video_file - video file locally on disk
           force - flag to indicate whether to recreate the video metadata.
        Output: 
           None
    """
    
    stream_info_file = os.path.join(self.video_asset_dir(), 'stream_info.json')

    # check to see if stream info already exists
    if os.path.exists(stream_info_file) and force == False:
        with open(stream_info_file, 'r', encoding="utf-8") as f:
            stream_info = json.loads(f.read())
        print(f"  get_stream_info: stream_info.json already exists. Use force=True to if you want to overwrite the existing stream_info. SKIPPING...")
        return stream_info
    
    video = urlparse(self.video_file)

    # input check: video is a file or https 
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # input check: file scheme video exists
    if video.scheme == 'file' and not os.path.exists(self.video_file):
        raise Exception('input video does not exist')
        
    util.mkdir(self.video_asset_dir())
    util.mkdir(self.frame_dir())

    command_string = f'ffprobe -v quiet -print_format json -show_format -show_streams {shlex.quote(self.video_file)}'
    
    # shlex.quote will place harmful input in quotes so it can't be executed by the shell
    # nosemgrep Rule ID: dangerous-subprocess-use-audit
    child_process = subprocess.Popen(
        shlex.split(command_string),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = child_process.communicate()

    stream_info = json.loads(str(stdout, 'utf-8'))
    stream_info['format']['filename'] = self.video_file

    # parse video_info
    video_stream = list(filter(lambda x: (x['codec_type'] == 'video'), stream_info['streams']))[0]

    progressive = bool('field_order' not in video_stream or video_stream['field_order'] == 'progressive')
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    duration_ms = int(float(video_stream['duration']) * 1000)
    num_frames = int(video_stream['nb_frames'])
    framerate = util.to_fraction(video_stream['r_frame_rate'])
    sample_aspect_ratio = util.to_fraction(video_stream['sample_aspect_ratio'])
    display_aspect_ratio = util.to_fraction(video_stream['display_aspect_ratio'])
    display_width = int((width * sample_aspect_ratio.numerator) / sample_aspect_ratio.denominator)

    stream_info['video_stream'] = {
        'duration_ms': duration_ms,
        'num_frames': num_frames,
        'framerate': (framerate.numerator, framerate.denominator),
        'progressive': progressive,
        'sample_aspect_ratio': (sample_aspect_ratio.numerator, sample_aspect_ratio.denominator),
        'display_aspect_ratio': (display_aspect_ratio.numerator, display_aspect_ratio.denominator),
        'encoded_resolution': (width, height),
        'display_resolution': (display_width, height),
    }

    util.save_to_file(stream_info_file, stream_info)

    return stream_info
  
  def extract_frames(self, force, max_res=(750, 500), sample_rate_fps=1):
    """
    Extract individual frames from a video file as JPEG images.

    Args:
        max_res (tuple, optional): The maximum resolution for the extracted frames. Defaults to (750, 500).
        sample_rate_fps (int, optional): The desired frame rate (frames per second) for the extracted frames.
            Defaults to 1.

    Returns:
        list: A list of file paths for the extracted JPEG frames.

    Raises:
        Exception: If the input video URL is not a valid local file path or HTTPS URL.
        Exception: If the input video file does not exist (for local file paths).

    Description:
        This function extracts individual frames from a video file and saves them as JPEG images
        in a 'frames' directory within the same directory as the video file.

        The function takes the video URL, the previously extracted stream_info (from the probe_stream function),
        an optional max_res parameter to specify the maximum resolution for the extracted frames, and an optional
        sample_rate_fps parameter to specify the desired frame rate for the extracted frames.

        If the video is interlaced, the function applies deinterlacing using the 'yadif' filter from FFmpeg.
        It then downscales the frames to the specified max_res using the 'scale' filter.

        The extracted frames are saved with sequential filenames (e.g., frames.0000001.jpg, frames.0000002.jpg, etc.)
        in the 'frames' directory. If the 'frames' directory already exists, the function skips the extraction process
        and returns the existing JPEG files.

        The function performs input validation to ensure that the provided video URL is a valid local file path
        or HTTPS URL, and that the file exists if it's a local file path.
    """
    video = urlparse(self.video_file)
    video_path = video.path
    video_dir = self.video_dir()
    frames_file = os.path.join(self.video_asset_dir(), 'frames.json')

    # check to see if the frame info already exists
    if os.path.exists(frames_file) and force == False:
        with open(frames_file, 'r', encoding="utf-8") as f:
            self.frames = json.loads(f.read())
        print(f"  extract_frames: frames.json already exists. Use force=True to if you want to force running frame extraction. SKIPPING...")
        return
    
    # input check: video is a file or https 
    if video.scheme not in ['https', 'file', '']:
        raise Exception('input video must be a local file path or use https')
    
    # input check: file scheme video exists
    if video.scheme == 'file' and not os.path.exists(self.video_file):
        raise Exception('input video does not exist')
    
    frame_dir = self.frame_dir()
    # Can't skip becasue we need the timestamps from the ffmpage command even if we have frames 
    #if len(self.frames) > 0 and force == False:
    #   print(f"  extract_frames: found frames. SKIPPING... Use the force=True option to regenerate frames")
    
    t0 = time.time()
    video_filters = []
    video_stream = self.stream_info['video_stream']
    
    # This is a video filter option that sets the frame rate to the specified frames per second. 
    # The fps filter adjusts the frame rate by duplicating or dropping frames as necessary. 
    video_filters.append(f"fps={sample_rate_fps},showinfo")
    
    # need deinterlacing
    progressive = video_stream['progressive']
    if not progressive:
        video_filters.append('yadif')
    
    # downscale image
    dw, dh = video_stream['display_resolution']
    factor = max((max_res[0] / dw), (max_res[1] / dh))
    w = round((dw * factor) / 2) * 2
    h = round((dh * factor) / 2) * 2
    video_filters.append(f"scale={w}x{h}")

    command = [
        'ffmpeg',
        #'-v',
        #'quiet',
        '-i',
        shlex.quote(self.video_file),
        # DEBUG: Just get the first 60 seconds
        #'-t',
        #str(60),
        '-vf',
        f"{','.join(video_filters)}",
        '-r',
        str(sample_rate_fps),
        '-f',
        'image2',
        f"{shlex.quote(frame_dir)}/frames%07d.jpg"
    ]
    
    print(f"  Resizing: {dw}x{dh} -> {w}x{h} (Progressive? {progressive})")
    print(f"  Command: {command}")
    
    self.frames = []
    try:
        # user input is wrapped in shlex above
        # nosemgrep: dangerous-subprocess-use-audit
        result = subprocess.run(command, shell=False, stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT, universal_newlines=True, check=True)
        
        jpeg_frames = sorted(glob.glob(f"{frame_dir}/*.jpg"))
        
        index = 0
        for line in result.stdout.splitlines():
            if "pts_time" in line:
                timestamp = float(line.split("pts_time:")[1].split()[0])
                newFrame = Frame(index, jpeg_frames[index], timestamp * 1000)
                frameDict = newFrame.__dict__
                self.frames.append(frameDict)
                # FIXME
                #self.frames.append(Frame(jpeg_frames[index], int(timestamp * 1000)))
                index+=1
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stdout}")

    util.save_to_file(frames_file, self.frames)
    
    t1 = time.time()
    print(f"  extract_frames: elapsed {round(t1 - t0, 2)}s")
    
    return 

  def composite_image(self, start_index, end_index, cols, variation="image", folder='composite-images'):
        """
        Create an image grid using the image frames from the given start and end index 
        and organized in the given number of columns.
        Args:
           start_index - start index of the picture frames in the current object
           end_index - end index of the picture frames in the current object
           cols - number of columns to organize the image grid
           variation - variation for the image grid
           folder - the folder to store the generated image grid.
        Returns:
           the file name path for generated image grid.
        """
        image_files = []
        frames = self.frames[start_index:end_index]
    
        composite_folder = os.path.join(self.video_asset_dir(), folder)
        os.makedirs(composite_folder, exist_ok=True)
        
        composite_image, grid_layout = frame_utils.create_grid_image(frames,
                         cols,
                         border_width = 2,
                         border_outline = (0, 0, 0),
                         burn_timecode = False)
        
        
        composite_file_name = os.path.basename(frames[0]["image_file"]).split(".")[0] + os.path.basename(frames[-1]["image_file"]).split(".")[0] + '.jpg'
        composite_image_file = os.path.join(composite_folder, composite_file_name)
        composite_image.save(composite_image_file)
        composite_image.close()
        
        return composite_image_file

  def calculate_similar_frames(self):
      
      frame_embeddings = [frame['titan_multimodal_embedding'] for frame in self.frames]

      for idx, frame in enumerate(self.frames):
          similar_frames = self.search_similarity(self.vector_store, frame, idx)
          frame['similar_frames'] = similar_frames
      
    
  def create_index(self, dimension):
        index = faiss.IndexFlatIP(dimension) # cosine similarity

        self.index_frames(index)
        return index

  def index_frames(self, index):
        for frame in self.frames:
            embedding = np.array([frame['titan_multimodal_embedding']])
            index.add(embedding)
      
        return index
    
  def cosine_similarity(self, a, b):
        cos_sim = dot(a, b) / (norm(a) * norm(b))
        return cos_sim
    
  


  def search_similarity(self, index, frame, idx, k = 20, min_similarity = 0.80, time_range = 30):
        
        embedding = np.array([frame['titan_multimodal_embedding']])
    
        D, I = index.search(embedding, k)
    
        similar_frames = [
            {
                'idx': int(i),
                'similarity': float(d)
            } for i, d in zip(I[0], D[0])
        ]
    
        # filter out lower similiarity
        similar_frames = list(
            filter(
                lambda x: x['similarity'] > min_similarity,
                similar_frames
            )
        )
    
        similar_frames = sorted(similar_frames, key=lambda x: x['idx'])
    
        # filter out frames that are far apart from the current frame idx
        filtered_by_time_range = [{
            'idx': idx,
            'similarity': 1.0
        }]
        # filtered_by_time_range = [similar_frames[0]]
    
        for i in range(0, len(similar_frames)):
            prev = filtered_by_time_range[-1]
            cur = similar_frames[i]
    
            if abs(prev['idx'] - cur['idx']) < time_range:
                   filtered_by_time_range.append(cur)
    
        return filtered_by_time_range

  def collect_similar_frames(self, start_frame_id, end_frame_id):
        """
        Creates a sorted list of frames between the given frame IDs based on similarity.
        The similar frames for each frame is precalculated in the Frames constructor.
        Args: 
           start_frame_id - the start of the frame index as a boundary
           end_frame_id - the end of the frame index as a boundary
        Returns:
           sorted list of all similar frames.
        
        """
        similar_frames = []
        for frame_id in range(start_frame_id, end_frame_id+1):
            similar_frames_ids = [frame['idx'] for frame in self.frames[frame_id]['similar_frames']]
            similar_frames.extend(similar_frames_ids)
        # unique frames in shot
        return sorted(list(set(similar_frames)))

  def set_shot_ids(self, shots):
        frames_file = os.path.join(self.video_asset_dir(), 'frames.json')
        for s, shot in enumerate(shots):
              for i in range(shot['start_frame_id'], shot['end_frame_id']+1):
                  self.frames[i]['shot_id'] = s
        util.save_to_file(frames_file, self.frames)
      
          
  def collect_related_shots(self, frame_ids):
        """
        Creates a sorted list of shots ids based on the similar image frames within a shot.
        Args:
           frame_ids - collection of frame ids to find the unique shots from.
        Returns:
           sorted list of unique short IDs.
        """
        related_shots = []
        for frame_id in frame_ids:
            if ('shot_id' in self.frames[frame_id]):
                related_shots.append(self.frames[frame_id]['shot_id'])
        # unique frames in shot
        return sorted(list(set(related_shots)))


  