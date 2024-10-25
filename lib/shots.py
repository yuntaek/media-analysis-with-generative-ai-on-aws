from lib import frame_utils
from PIL import Image, ImageDraw, ImageFont
import boto3
import time
import json
import os
from lib import util

class Shots():
    def __init__(self, videoframes, method="RekognitionShots", force=False):
        self.shots = []
        shots_file = os.path.join(videoframes.video_asset_dir(), 'shots.json')

        if os.path.exists(shots_file) and force == False:
            pass # keep the existing data
        else:
            self.shots_with_no_frames = []
        
        # check to see if the frame info already exists
        if os.path.exists(shots_file) and force == False:
            with open(shots_file, 'r', encoding="utf-8") as f:
                self.shots = json.loads(f.read())
            print(f"  Shots: shots.json already exists. Use force=True to if you want to force detecting shots. SKIPPING...")
            return
        
        if method=="RekognitionShots":
            self.rekognition_shots = self.rekognition_shot_detection(videoframes)
            self.shots = self.group_frames_by_rekognition_shots(videoframes, self.rekognition_shots)
            util.save_to_file(shots_file, self.shots)
        elif method == "SimilarFrames":
            raise NotImplementedError
        else:
            raise NameError

        videoframes.set_shot_ids(self.shots)

        return


    def rekognition_shot_detection(self, videoframes):
        """
        Starts a rekognition start_segment_detection API (https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/rekognition/client/start_segment_detection.html) to extract shots from the given video object. 
        Args: 
            videoframes: The video object that contains the information about the video to be processed.
        Returns:
            object that presents the shot segment returned from the Rekognition API.
        """

        rekognition = boto3.client('rekognition') 
        #Make the API Call to start shot detection
        startSegmentDetection = rekognition.start_segment_detection(
            Video={
                'S3Object': {
                    'Bucket': videoframes.object["Bucket"],
                    'Name': videoframes.object["Key"]
                },
            },
            SegmentTypes=['SHOT']
        )
        
        #Grab the ID of our job
        segmentationJobId = startSegmentDetection['JobId']

        #Grab the segment detection response
        getSegmentDetection = rekognition.get_segment_detection(
            JobId=segmentationJobId
        )

        # Determine the state. If the job is still processing we will wait and check again
        while(getSegmentDetection['JobStatus'] == 'IN_PROGRESS'):
            # intentional polling loop for async operation
            # nosemgrep: arbitrary-sleep
            time.sleep(5)
            print(f"Amazon Rekognition Shot Detection JobStatus = {getSegmentDetection['JobStatus']}")
            getSegmentDetection = rekognition.get_segment_detection(
                JobId=segmentationJobId)

        print(f"JobStatus = {getSegmentDetection['JobStatus']}")

        if (getSegmentDetection['JobStatus'] != "FAILED"):
            return getSegmentDetection['Segments']
        else:
            print(f"GetSegmentDetection JobStatus == FAILED: {getSegmentDetection['StatusMessage']}")
            raise RuntimeError

    def group_frames_by_rekognition_shots(self, videoframes, rekognition_shots):
        """
        Creates a group of image collections consist of video frames within the shot.
        Args:
           videoframes - the video object
           rekognition_shots - technical ques that represents shots for the given video.
        
        Returns:
           a list of unique Shot objects for the video.
        """
        current_shot_frames = []
        shots = []
        shot_id = 0
        
        for technicalCue in rekognition_shots:
            for frame in videoframes.frames:
                start_ms = technicalCue['StartTimestampMillis']

                # Find the start point of the scene
                end_ms = technicalCue['EndTimestampMillis']

                if frame['timestamp_millis'] >= start_ms and frame['timestamp_millis'] <= end_ms:
                    current_shot_frames.append(frame)

            if current_shot_frames:
                current_shot = Shot("RekognitionShots", shot_id, videoframes, current_shot_frames, start_ms, end_ms, technicalCue['DurationMillis'], technicalCue)
                shots.append(current_shot.__dict__)
                current_shot_frames = []
                shot_id = shot_id+1
            else:
                self.shots_with_no_frames.append(technicalCue["ShotSegment"]["Index"])

        return shots


class Shot():

     def __init__(self, method, id, frames, shot_frames, start_ms, end_ms, duration_ms, rekognition_shot=None):
        
        self.method = method
        self.id = id
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.duration_ms = duration_ms
        self.video_asset_dir = frames.video_asset_dir()
        self.start_frame_id = shot_frames[0]['id']
        self.end_frame_id = shot_frames[-1]['id']
        self.create_composite_images(shot_frames)

     def create_composite_images(self, frames):
        folder = os.path.join(self.video_asset_dir, 'shots')
        os.makedirs(folder, exist_ok=True) 
        self.composite_images = frame_utils.create_composite_images(
                frames,
                output_dir = folder,
                prefix = 'shot_',
                max_dimension = (1568, 1568),
                burn_timecode = False
                )
        return
        