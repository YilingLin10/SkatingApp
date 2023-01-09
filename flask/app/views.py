from app import app
from flask import request, jsonify, Response, send_file, make_response, after_this_request
import requests
from requests.exceptions import ConnectionError
import shutil
import os
import uuid
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from app.config import CONFIG

disable_warnings(InsecureRequestWarning)

requests.DEFAULT_RETRIES = 5
session = requests.Session()
session.keep_alive = False

@app.route("/")
def index():
    return "hello from flask"

def download_video(id):
    local_video_id = uuid.uuid1().hex
    request_url = f"https://4be4-140-109-16-205.ap.ngrok.io/uploads/video/{id}.mp4"
    try:
        response = session.get(request_url, stream=True, verify=CONFIG.CA_path, timeout=200)
        if response.status_code == 200:
            with open(f"{CONFIG.video_buffer_dir}/{id}.mp4", 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            if (os.path.isfile(f"{CONFIG.video_buffer_dir}/{id}.mp4")):
                os.rename(f"{CONFIG.video_buffer_dir}/{id}.mp4", f"{CONFIG.video_buffer_dir}/{local_video_id}.mp4")
            else:
                print("Downloaded file does not exist")
                return "error"
        else:
            print(f"Cannot download video {id}")
            return "error"
    except ConnectionError as e:
        print(e)
        return "error"
    return local_video_id
    
@app.route("/jump", methods=["GET"])
def getJumpDuration():
    video_id = request.args.get("id")
    local_video_id = download_video(video_id)
    if local_video_id == "error":
        return f"Failed to download video {video_id}", 500
    else:
        print("Successfully downloaded the video.")
    
    # RUN ALPHAPOSE
    os.system(f"bash {CONFIG.alphapose_script} jump {local_video_id}")
    alphapose_result_path = f"{CONFIG.jump_alphapose_result_dir}/alpha_pose_{local_video_id}"
    
    if (not os.path.exists(f"{alphapose_result_path}/alphapose-results.json")):
        os.remove(f"{CONFIG.video_buffer_dir}/{local_video_id}.mp4")
        shutil.rmtree(alphapose_result_path)
        return "Failed to run AlphaPose", 500
    
    # DATA PREPROCESSING & INFERENCE
    os.system(f"bash {CONFIG.jump_script} {local_video_id}")
    
    pkl_data_path = f"{CONFIG.jump_pkl_result_dir}/{local_video_id}.pkl"
    result_video_path = f"{CONFIG.result_dir}/jump_{local_video_id}.mp4"
    jump_duration_list_path = f"{CONFIG.result_dir}/jump_{local_video_id}.txt"
    if os.path.exists(result_video_path) and os.path.exists(jump_duration_list_path):
        response = make_response(send_file(result_video_path, 'video/mp4'))
        jump_duration_list = []
        with open(jump_duration_list_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                if l.strip(): #Checking Non-Empty Line
                    jump_duration_list.append(l.strip())
        response.headers['jump_duration_list'] = ';'.join(duration for duration in jump_duration_list)
    else:
        os.remove(f"{CONFIG.video_buffer_dir}/{local_video_id}.mp4")
        os.remove(result_video_path)
        os.remove(jump_duration_list_path)
        shutil.rmtree(alphapose_result_path)
        os.remove(pkl_data_path)
        return "Failed to run prediction.", 500
    
    @after_this_request
    def delete(response):
        os.remove(f"{CONFIG.video_buffer_dir}/{local_video_id}.mp4")
        os.remove(result_video_path)
        os.remove(jump_duration_list_path)
        shutil.rmtree(alphapose_result_path)
        os.remove(pkl_data_path)
        return response
    return response

@app.route("/align", methods=["GET"])
def getAlignedVideo():
    video_id_1 = request.args.get("id1")
    video_id_2 = request.args.get("id2")
    
    local_video_id_1 = download_video(video_id_1)
    if local_video_id_1 == "error":
        return f"Failed to download video {video_id_1}", 500
    else:
        print("Successfully downloaded video #1")
        
    local_video_id_2 = download_video(video_id_2)
    if local_video_id_2 == "error":
        return f"Failed to download video {video_id_2}", 500
    else:
        print("Successfully downloaded video #2")
    
    # Generate a unique identifier for this alignment
    dir_id = uuid.uuid1().hex
    alignment_data_path = f'{CONFIG.align_data_dir}/{dir_id}'
    if os.path.exists(alignment_data_path):
        shutil.rmtree(alignment_data_path, ignore_errors=True)
    else:
        os.mkdir(alignment_data_path)
    
    # RUN ALPHAPOSE
    os.system(f"bash {CONFIG.alphapose_script} align {local_video_id_1} {dir_id}")
    os.system(f"bash {CONFIG.alphapose_script} align {local_video_id_2} {dir_id}")
    
    if (not os.path.exists(f"{alignment_data_path}/alpha_pose_{local_video_id_1}/alphapose-results.json") or 
        not os.path.exists(f"{alignment_data_path}/alpha_pose_{local_video_id_2}/alphapose-results.json")):
        os.remove(f"{CONFIG.video_buffer_dir}/{local_video_id_1}.mp4")
        os.remove(f"{CONFIG.video_buffer_dir}/{local_video_id_2}.mp4")
        shutil.rmtree(f"{alignment_data_path}")
        return "Failed to run AlphaPose", 500
        
    # RUN ALIGNMENT
    os.system(f"bash {CONFIG.alignment_script} {dir_id}")
    
    result_video_path = f"{CONFIG.result_dir}/align_{dir_id}.mp4"
    if os.path.exists(result_video_path):
        response = make_response(send_file(result_video_path, 'video/mp4'))
    else:
        return "Failed to generate the aligned video.", 500
    @after_this_request
    def delete(response):
        if os.path.exists(f"{CONFIG.video_buffer_dir}/{local_video_id_1}.mp4"):
            os.remove(f"{CONFIG.video_buffer_dir}/{local_video_id_1}.mp4")
        if os.path.exists(f"{CONFIG.video_buffer_dir}/{local_video_id_2}.mp4"):
            os.remove(f"{CONFIG.video_buffer_dir}/{local_video_id_2}.mp4")
        if os.path.exists(result_video_path):
            os.remove(result_video_path)
        if os.path.exists(f"{alignment_data_path}"):
            shutil.rmtree(f"{alignment_data_path}")
        return response
    return response

## TESTING ENDPOINTS
@app.route("/download", methods=["GET"])
def testDownload():
    video_id_1 = request.args.get("id1")
    video_id_2 = request.args.get("id2")
    
    local_video_id_1 = download_video(video_id_1)
    if local_video_id_1 == "error":
        return f"Failed to download video {video_id_1}", 500
    local_video_id_2 = download_video(video_id_2)
    if local_video_id_2 == "error":
        return f"Failed to download video {video_id_2}", 500
    
    return "ok", 200
    